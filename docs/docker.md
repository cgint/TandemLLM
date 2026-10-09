# Running TandemLLM in Docker

A side-by-side, optional path next to `install.sh`: the same engine, the same pinned
dependencies, the same `ops/serve.env` contract -- packaged as an image whose weights are
never baked in. The primary target is one NVIDIA DGX Spark (GB10, aarch64, sm_121), exactly
like `install.sh`; the image also builds on x86_64 (every pin has both arches' wheels), but
the served profile was measured on the Spark.

Nothing here replaces the native path: `install.sh` stays the canonical installer. This
document and the five new files (`Dockerfile`, `docker/entrypoint.sh`, `docker-compose.yml`,
`docker/smoke.sh`, `docker/requirements.lock`) add the container path without touching
`install.sh`, `server/`, `engine/` or `ops/serve.env`.

## What goes where

| | image | host volume |
|---|---|---|
| engine source (`server/`, `engine/`, `tools/`, `ops/`) | `/opt/tandemllm` (baked in at build time) | -- |
| Python 3.11 venv, pinned to `docker/requirements.lock` | `/opt/venv` | -- |
| CUDA toolkit (nvcc, the NVIDIA runtime stack) | base image `nvcr.io/nvidia/pytorch:26.07-py3` | driver (host) |
| weights (the four HF repositories) | **never** | HF cache root, mounted read-only at `/var/tandemllm/hf` |
| lookup corpus | -- | built on first start into `tandem-data` |
| kernel JIT cache (`torch_extensions`, sm_121a skinny kernel) | -- | `tandem-data` (must persist: a rebuild is silent and slow) |
| logs, usage ledger | -- | `tandem-data` |
| `serve.env` | the repo's own, as default | mount a deployment file to override |

## The base image, and why Python 3.11 still comes from uv

`nvcr.io/nvidia/pytorch:26.07-py3` is multi-arch (amd64 and arm64) and carries the CUDA 13.x
toolkit -- the `nvcc` the sm_121a skinny kernel is JIT-compiled with on first use
(`tools/nvfp4_skinny.py` via `torch.utils.cpp_extension.load_inline`), cached under
`$TORCH_EXTENSIONS_DIR`. Its default Python is 3.12 (`/usr/local/lib/python3.12/...`), so
the 3.11 venv the engine needs (`tests/test_grammar.py` passes under 3.11 and fails under
3.12, the same reason `install.sh` pins 3.11) is created inside by `uv` 0.12.12 -- the
same pinned version `install.sh` fetches when the host has neither `uv` nor `python3.11`.
The image takes its CUDA from the base, its Python from the venv.

Check at build time (`[unverified]` until the first build):

```
docker run --rm --entrypoint nvcc tandemllm:latest --version
docker run --rm --entrypoint cat  tandemllm:latest /etc/os-release
```

`install.sh` only *warns* when `nvcc` is not a 13.x release; if the base's toolkit is
13.3, the first start is the moment the skinny-kernel compile is actually verified --
watch for `$TORCH_EXTENSIONS_DIR` to fill under the `tandem-data` volume.

## Build

The build needs no GPU and no driver: the dependency layer only downloads wheels, and the
CUDA-availability assert deliberately lives in the entrypoint, not the image.

```
git clone https://github.com/0xBakeer/TandemLLM && cd TandemLLM
# on the Spark (or any arm64 host):
docker build --platform linux/arm64 -t tandemllm:latest .
```

On an x86_64 build host, build for the Spark with `docker buildx` / a remote arm64
builder; the target arch of the served image is `linux/arm64`.

## Run

`docker-compose.yml` (repo root):

```
# the weights must already be in the host's HF cache (install.sh step 4, or pre-acquired):
docker compose up -d
docker compose logs -f tandem          # first start: corpus build (a few minutes), then load
docker compose exec -T tandem bash docker/smoke.sh   # /health + one generation, install.sh's smoke
```

The API is on the host's `:8000` (change the mapping to taste). The compose file sets,
mirroring the native path's needs:

- `ipc: host` -- the engine allocates shared memory across processes;
- all GPUs (`deploy.resources.reservations.devices`, `driver: nvidia`, `count: all`);
- the HF cache root mounted **read-only** at `/var/tandemllm/hf` (the entrypoint's
  `HF_HOME`; the hub cache lands in `<mount>/hub`, the `install.sh` convention);
- `tandem-data`, a persistent volume at `/var/tandemllm/data` = `TANDEM_DATA_DIR`:
  the corpus, `torch_extensions`, logs, and the usage ledger.

The engine takes ~64 GiB of the Spark's unified memory at load (the same number
`install.sh` checks): **stop the host's other GPU work before `up`**. The entrypoint
checks `MemAvailable` and warns (override the floor with `TANDEM_MIN_FREE_GIB`).

## The volume contract

1. **weights** (`/var/tandemllm/hf`, read-only) -- the host's HF cache root, with the four
   repositories pre-acquired, exactly as `install.sh` step 4 leaves them:
   `Qwen/Qwen3.8-27B-FP8` (the base, without `mtp.safetensors`),
   `0xBakeer/TandemLLM-Qwen3.8-27B-NVFP4`,
   `0xBakeer/TandemLLM-Qwen3.8-27B-DFlash2-b8`,
   `0xBakeer/TandemLLM-Qwen3.8-27B-DFlash2-b16`.
   The entrypoint points `QWEN38_MODEL`, `NV`, `HEAD`, `CKPT8`, `CKPT16` at the snapshot
   directory of each cache entry and refuses to start if one is missing. The hub cache
   names snapshots by *revision* (a git-commit SHA, not a date) and keeps an authoritative
   pointer in `<cache>/models--<org>--<repo>/refs/`; the entrypoint reads that pointer
   (falling back to newest-by-mtime for a partial, hand-assembled cache) rather than sorting
   directory names -- the same directory `install.sh`'s `snapshot_download` returns, resolved
   from the pre-acquired cache.
   Weights are never in the image.
2. **corpus** (`$TANDEM_DATA_DIR/corpus`) -- built CPU-only on the container's first start
   (a few minutes), into the volume, from the venv's own `site-packages` (transformers,
   torch, triton) plus one English and one German Wikipedia shard
   (`wikimedia/wikipedia`, CC BY-SA 4.0) -- the same invocation `install.sh` step 5 makes
   (`--max-tokens 40000000`, the same `--parquet`/`--src` split). It *cannot* be pre-built
   elsewhere: it depends on the image's venv. The base tokenizer is read from the **mounted**
   weights (no live re-download of the base repository). Re-runs find `meta.json` and skip.
   `TANDEM_NO_CORPUS=1` skips it entirely (the installer's `--no-corpus`: the lookup drafter
   then copies from the conversation only; the entrypoint omits `--corpus` in that case,
   matching the unit's word-splitting of an empty `$CORPUS`).
3. **kernels** (`$TANDEM_DATA_DIR/torch_extensions`) -- the `nvcc` cache of the sm_121a
   skinny kernel, keyed by source hash. It must be the *same volume across container
   recreations*: a silent rebuild costs minutes and is exactly the kind of thing that
   should not happen invisibly. `TORCH_EXTENSIONS_DIR` is set to it explicitly (not left
   at `~/.cache/torch_extensions`).
4. **logs + ledger** (`$TANDEM_DATA_DIR/logs`, `$DATA/state/usage/ledger.sqlite3`) -- the
   engine log (`engine.log` symlink, one file per start) and the usage ledger.

## The serve.env contract

The entrypoint sources `SERVE_ENV` (default: the repo's own `ops/serve.env`, baked in) and
then runs the server with the **exact** argument list of `ops/qwen38-engine.service`'s
`ExecStart` (`--drafter lenrouter --len-fixed $LEN_FIXED $LATCH_FLAG --dflash2-path greedy
--tree --budget $BUDGET --corpus $CORPUS --dflash2-ckpt $CKPT8 --dflash2-ckpt16 $CKPT16
--nvfp4 $NV --fp8-head $HEAD --cache-budget-gb $CACHE_GB --request-timeout ... --max-queue
... --queue-timeout ... --verbose`, plus `QWEN38_FUSE_PROJ=$FUSE_PROJ` and
`PYTHONPATH` exported, `LATCH_FLAG` set from `LEN_LATCH` -- the unit's `bash -lc` body,
line for line).

One difference from the unit, by necessity: the repo's `ops/serve.env` points
`QWEN38_MODEL`, `NV`, `HEAD`, `CKPT8`, `CKPT16`, `CORPUS`, `QSE_USAGE_LEDGER` at the
author's *host* install paths, which do not exist in the container. The entrypoint
re-points those seven at the container's snapshot directories and volumes **after**
sourcing the file, so every other line of `serve.env` (StairCut, the tree, the kernels,
every `QWEN38_*` switch, the measured values) reaches the server byte for byte. A
deployment can instead mount its own `serve.env` carrying container paths and set
`SERVE_ENV` to it -- the entrypoint's re-pointing then becomes a no-op for those keys.

## What is different vs `install.sh`

| | `install.sh` | Docker |
|---|---|---|
| engine source | `git clone` at a pinned ref into `$DIR/src` | baked into the image at build time |
| Python venv | `$DIR/venv`, 3.11 via uv/venv | `/opt/venv`, 3.11 via uv inside the image |
| dependency pins | the `LOCK` block of `install.sh` | `docker/requirements.lock` -- the `LOCK` block copied verbatim; drift shows as a diff in one file |
| weights | downloaded by the installer into the HF cache | **never in the image**; mounted read-only from the host's cache |
| corpus | built by the installer into `$DIR/corpus` | built by the entrypoint on first start into the `tandem-data` volume |
| kernel cache | `~/.cache/torch_extensions` on the box | `tandem-data` volume (`TORCH_EXTENSIONS_DIR`), so it survives container recreation |
| serve.env | rendered into `$DIR/run/ops/serve.env` | the repo's file, sourced as-is; host-path keys re-pointed by the entrypoint (or mount your own) |
| supervisor | the `bin/tandem` control script / the cron line | `docker compose` (one container; `Restart=on-failure` equivalent via compose if you add it) |
| memory check | refuses to start under 64 GiB free | warns (the entrypoint); the compose file notes it too |
| first-start cost | installer steps 2-5 on the box | image build + corpus build + kernel JIT (first generation only) |

Unchanged by design: the pins (a different engine is a different engine), the 3.11
requirement, the exec-line arguments, the `serve.env` semantics, the ~64 GiB need, and
the "one engine per board" rule (a container does not change that: stop other GPU work
first, and do not run the native engine on `:8000` alongside the container's).

## HF download environment (for any in-container fetch)

The container normally only *mounts* pre-fetched weights; the only in-container fetches
are the two Wikipedia shards and the base repository's tokenizer/config at first start.
Download performance is governed by the host's environment (`HF_TOKEN`,
`HF_HUB_DISABLE_XET`, `HF_HUB_ENABLE_HF_TRANSFER`, ...); nothing is baked in here -- a
homelab may prefer `HF_HUB_DISABLE_XET=1` / `HF_HUB_ENABLE_HF_TRANSFER=0`, and either
passes through as the container's environment.

<!-- fleet-note: begin (excise this section for upstream) -->

## Fleet note (homelab)

- **Weights pre-acquired on pluto** (the build box): the four repositories land on pluto's
  NFS-mounted HF cache, which is the host-side path the compose file mounts at
  `/var/tandemllm/hf`. The Spark pulls the cache from NFS; nothing weight-shaped is
  rebuilt per board.
- **Image via pluto's registry**: build once on pluto, `docker tag` and `docker push` to
  `pluto:5001`, and let the Spark `docker compose pull` from there instead of building
  (`build:` in the compose file becomes `image: pluto:5001/tandemllm:latest`).
- The `tandem-data` volume is local to the Spark: the kernel JIT cache and the corpus do
  not NFS well (they are written once and read constantly).
<!-- fleet-note: end -->
