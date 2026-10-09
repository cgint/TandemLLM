#!/usr/bin/env bash
# TandemLLM container entrypoint: reproduce install.sh's runtime contract inside a container.
#
# It does what the native path's install.sh + ops/start.sh do at start time:
#   1. check the GPU and the CUDA compiler (install.sh step 1, condensed -- this is where the
#      CUDA-availability assert lives, not in the Dockerfile, so the image builds without a driver)
#   2. build the lookup corpus on first start, into the corpus volume (install.sh step 5), unless
#      TANDEM_NO_CORPUS=1 (the installer's --no-corpus: the lookup drafter copies from the
#      conversation only)
#   3. source the serve.env (the repo's ops/serve.env by default, SERVE_ENV to override) and run
#      the server with the EXACT argument list of ops/qwen38-engine.service's ExecStart.
#
# Weights are never in the image: the Hugging Face cache root is a volume mounted at /var/tandemllm/hf
# (the install.sh convention ${HF_HOME:-$HOME/.cache/huggingface}/hub; see docs/docker.md).
set -euo pipefail

REPO="${REPO:-/opt/tandemllm}"
PY="${PY:-/opt/venv/bin/python}"
DATA="${TANDEM_DATA_DIR:-/var/tandemllm/data}"   # the persistent volume: corpus, kernels, logs, state
SERVE_ENV="${SERVE_ENV:-$REPO/ops/serve.env}"
TZ="${TZ:-Europe/Berlin}"
export TZ
cd "$REPO"

# The paths every volume-mounted piece of state lives under: $DATA is the named volume,
# $HF_HOME the (read-only) weights mount.
export HF_HOME="${HF_HOME:-/var/tandemllm/hf}"          # the hub cache lands in $HF_HOME/hub (install.sh's convention)
export TORCH_EXTENSIONS_DIR="$DATA/torch_extensions"   # the sm_121a skinny kernel's JIT cache; must persist
mkdir -p "$TORCH_EXTENSIONS_DIR" "$DATA/corpus" "$DATA/logs" "$DATA/state/usage"

log() { printf '[tandem-docker] %s\n' "$*"; }

# ---------------------------------------------------------------------------------------------
# 1. Checks (the installer's GPU/nvcc asserts, condensed; no driver -> no engine)
# ---------------------------------------------------------------------------------------------
if ! command -v nvidia-smi >/dev/null 2>&1 || ! nvidia-smi -L >/dev/null 2>&1; then
    echo "[tandem-docker] no NVIDIA GPU reachable (nvidia-smi missing or driverless)." >&2
    echo "                The image builds without one, but the engine cannot start." >&2
    exit 1
fi
CC="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)"
log "GPU $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1), compute capability $CC"
if [ "$CC" != "12.1" ]; then
    log "WARNING: not a DGX Spark (GB10, sm_121); other GPUs are best effort (install.sh's wording)."
fi
# The skinny NVFP4 kernel is JIT-compiled with nvcc on first use: the base image carries the
# CUDA toolkit; this assert is the installer's step-1 check, moved here.
if [ ! -x /usr/local/cuda/bin/nvcc ]; then
    echo "[tandem-docker] no nvcc at /usr/local/cuda/bin/nvcc: the sm_121a kernel cannot be built." >&2
    exit 1
fi
command -v g++ >/dev/null 2>&1 || { echo "[tandem-docker] no g++ (the kernel build needs it)." >&2; exit 1; }
# install.sh asserts torch sees the GPU; the venv's torch is the cu130 wheel.
"$PY" - <<'EOF' || exit 1
import torch, triton, transformers
assert torch.cuda.is_available(), "torch.cuda.is_available() is False"
cc = ".".join(map(str, torch.cuda.get_device_capability(0)))
print(f"[tandem-docker] ok torch {torch.__version__} (CUDA {torch.version.cuda}), triton {triton.__version__}, "
      f"transformers {transformers.__version__}; {torch.cuda.get_device_name(0)} sm_{cc.replace('.', '')}")
EOF

# Unified-memory check (install.sh: the served stack takes ~64 GiB of the Spark at load).
MEM_AVAIL="$(awk '$1 == "MemAvailable:" { printf "%d", $2 / 1048576 }' /proc/meminfo)"
MIN_FREE="${TANDEM_MIN_FREE_GIB:-64}"
[ "$MEM_AVAIL" -ge "$MIN_FREE" ] || log "WARNING: ${MEM_AVAIL} GiB available, the engine needs about ${MIN_FREE} GiB to load; stop other GPU work first"

# ---------------------------------------------------------------------------------------------
# 2. The lookup corpus (install.sh step 5, CPU-only) -- built into the volume on first start.
#    TANDEM_NO_CORPUS=1 skips it (the installer's --no-corpus).
#    Snapshot resolution is done here, before the corpus build, because the build reads the
#    base tokenizer from the already-mounted weights -- no live re-download.
# ---------------------------------------------------------------------------------------------
HF_HUB_DIR="${HF_HUB_CACHE:-$HF_HOME/hub}"
hf_snap() {
    # the snapshot directory of a hub repository cache entry. The cache keeps an authoritative
    # revision pointer in refs/ (a symlink the hub client maintains); the snapshot dir is named
    # by that revision, not by date, so it is read rather than sorted. install.sh resolves the
    # same directory through snapshot_download's return value; here the cache is pre-acquired, so
    # the refs/ pointer is the equivalent source of truth.
    local repo_dir="$HF_HUB_DIR/models--${1//\//--}"
    local sha=""
    for ref in main main-v2 master; do
        [ -f "$repo_dir/refs/$ref" ] && { sha="$(cat "$repo_dir/refs/$ref")"; break; }
    done
    if [ -n "$sha" ] && [ -d "$repo_dir/snapshots/$sha" ]; then
        echo "$repo_dir/snapshots/$sha"
    else
        # no refs/ pointer (partial or hand-assembled cache): fall back to the newest-by-mtime
        ls -1dt "$repo_dir"/snapshots/* 2>/dev/null | head -n 1
    fi
}
SNAP_BASE="$(hf_snap "Qwen/Qwen3.8-27B-FP8")"
SNAP_NV="$(hf_snap "0xBakeer/TandemLLM-Qwen3.8-27B-NVFP4")"
SNAP_B8="$(hf_snap "0xBakeer/TandemLLM-Qwen3.8-27B-DFlash2-b8")"
SNAP_B16="$(hf_snap "0xBakeer/TandemLLM-Qwen3.8-27B-DFlash2-b16")"
for d in "$SNAP_BASE" "$SNAP_NV" "$SNAP_B8" "$SNAP_B16"; do
    [ -n "$d" ] && [ -d "$d" ] || {
        echo "[tandem-docker] the Hugging Face cache under $HF_HUB_DIR is missing a snapshot: $d" >&2
        echo "                (re-point it to a pre-acquired cache, or see docs/docker.md)." >&2
        exit 1
    }
done

CORPUS_DIR="$DATA/corpus"
if [ "${TANDEM_NO_CORPUS:-0}" = "1" ]; then
    CORPUS_VALUE=""
    log "corpus: skipped (TANDEM_NO_CORPUS=1): the lookup drafter copies from the conversation only"
elif [ -f "$CORPUS_DIR/meta.json" ]; then
    CORPUS_VALUE="$CORPUS_DIR"
    log "corpus: already built: $CORPUS_DIR ($(sed -n 's/.*"n_tokens": \([0-9]*\).*/\1/p' "$CORPUS_DIR/meta.json") tokens)"
else
    log "corpus: building on first start (CPU only, a few minutes)"
    WIKI_REPO="wikimedia/wikipedia"
    WIKI_EN="20231101.en/train-00000-of-00041.parquet"
    WIKI_DE="20231101.de/train-00000-of-00020.parquet"
    # The base tokenizer comes from the mounted weights (SNAP_BASE, resolved above) -- no live
    # re-download of the base repository's config.
    WIKI_PATHS="$("$PY" - "$WIKI_REPO" "$WIKI_EN" "$WIKI_DE" <<'EOF'
import sys
from huggingface_hub import hf_hub_download
repo = sys.argv[1]
for f in sys.argv[2:]:
    print(hf_hub_download(repo, f, repo_type="dataset"))
EOF
)" || { echo "[tandem-docker] Wikipedia download failed; re-run with TANDEM_NO_CORPUS=1 to skip the corpus" >&2; exit 1; }
    EN_PQ="$(sed -n 1p <<<"$WIKI_PATHS")"; DE_PQ="$(sed -n 2p <<<"$WIKI_PATHS")"
    SITE="$("$PY" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
    rm -rf "$CORPUS_DIR.partial"
    (nice -n 10 "$PY" tools/build_corpus.py --model "$SNAP_BASE" --out "$CORPUS_DIR.partial"
        --parquet "$EN_PQ:text::12000000" --parquet "$DE_PQ:text::7000000"
        --src "$SITE/transformers:6000000" --src "$SITE/torch:5000000" --src "$SITE/triton:1500000"
        --src "$REPO/engine" --src "$REPO/tools" --max-tokens 40000000) 2>&1 | sed 's/^/    | /' \
        || { echo "[tandem-docker] the corpus build failed; re-run with TANDEM_NO_CORPUS=1 to skip it" >&2; exit 1; }
    [ -f "$CORPUS_DIR.partial/meta.json" ] || { echo "[tandem-docker] the corpus build wrote no meta.json" >&2; exit 1; }
    mv "$CORPUS_DIR.partial" "$CORPUS_DIR"
    CORPUS_VALUE="$CORPUS_DIR"
    log "corpus: built: $CORPUS_DIR"
fi

# ---------------------------------------------------------------------------------------------
# 3. The server: the exact ExecStart of ops/qwen38-engine.service (which itself reads the
#    serve.env the unit sets via EnvironmentFile). Same contract as the native path: the env
#    file is the single source of every runtime knob.
#    The repo's template points NV/HEAD/CKPT8/CKPT16/CORPUS/QSE_USAGE_LEDGER at the author's
#    host install; those are re-pointed at the container's volumes here (a host-mounted
#    SERVE_ENV may already carry container paths -- in which case this is a no-op for them).
# ---------------------------------------------------------------------------------------------
set -a
# shellcheck source=/dev/null
. "$SERVE_ENV"
set +a

# The container's paths (after the env file: the file's own values win only if it already
# carries container paths).
export PY="$PY" REPO="$REPO"
export QWEN38_MODEL="$SNAP_BASE"
export NV="${SNAP_NV}/mlp.safetensors,${SNAP_NV}/gdn.safetensors,${SNAP_NV}/attn.safetensors"
export HEAD="${SNAP_NV}/head-fp8.safetensors"
export CKPT8="$SNAP_B8"
export CKPT16="$SNAP_B16"
export CORPUS="$CORPUS_VALUE"
[ -n "${QSE_USAGE_LEDGER:-}" ] && export QSE_USAGE_LEDGER="$DATA/state/usage/ledger.sqlite3"
# --corpus: the systemd unit's exec line leaves $CORPUS unquoted, so an empty value is removed
# by bash word-splitting and the argument is absent (the server then takes its own default: the
# lookup drafter copies from the conversation only). A quoted "--corpus ''" would instead pass a
# real empty string to a different code path. Match the unit's word-splitting exactly.
CORPUS_FLAG=""
[ -n "$CORPUS" ] && CORPUS_FLAG="--corpus $CORPUS"
[ -n "${CUDA_HOME:-}" ] || export CUDA_HOME="/usr/local/cuda"

# The exact argument list of ops/qwen38-engine.service's ExecStart:
export QWEN38_FUSE_PROJ="$FUSE_PROJ" PYTHONPATH="$PYTHONPATH"
LATCH_FLAG=""
[ "${LEN_LATCH:-0}" = "1" ] && LATCH_FLAG="--len-latch"

LOG="$DATA/logs/engine-$(date +%Y%m%d-%H%M%S).log"
ln -sfn "$LOG" "$DATA/logs/engine.log"
log "serving on :$PORT (log: $LOG)"
log "exec: $PY -u server/app.py --host $HOST --port $PORT ..."

exec "$PY" -u server/app.py \
    --host "$HOST" --port "$PORT" --served-model "$SERVED_MODEL" \
    --max-len "$MAX_LEN" --default-max-tokens "$DEFAULT_MAX_TOKENS" \
    --reasoning-format "$REASONING_FORMAT" --reasoning-effort "$REASONING_EFFORT" \
    --drafter lenrouter --len-fixed "$LEN_FIXED" $LATCH_FLAG \
    --dflash2-path greedy --tree --budget "$BUDGET" $CORPUS_FLAG \
    --dflash2-ckpt "$CKPT8" --dflash2-ckpt16 "$CKPT16" \
    --nvfp4 "$NV" --fp8-head "$HEAD" --cache-budget-gb "$CACHE_GB" \
    --request-timeout "$REQUEST_TIMEOUT" --max-queue "$MAX_QUEUE" \
    --queue-timeout "$QUEUE_TIMEOUT" --verbose \
    >>"$LOG" 2>&1
