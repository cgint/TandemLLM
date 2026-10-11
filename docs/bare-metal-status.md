# Bare-metal (native) run — status

## North star

**One engine, full power, on the target GB10 DGX Spark, serving its weights in place from
the on-disk cache (never downloaded, never baked into an image) — reachable and usable.
**

Everything in this file and `docs/docker-status.md` serves that. "Full power" is the
default recipe: `--max-len 262144`, ~56–69 GiB footprint, spec-decoding (StairCut) on.
Two delivery paths, same target: (1) **native** `install.sh` (this file) and (2)
**Docker** (`docs/docker-status.md`). The two never run at once on the board (one engine
per board, ~69 GiB each of 128).

- **Phase 1 — native:** DONE. Proven at full power (below). This also answered the open
  `sm_121a` JIT unknown and produced the reference numbers.
- **Phase 2 — Docker:** IN PROGRESS. Make the *exact* bare-metal run that is currently
  working on the board runnable as a container — same weights source, same full-power
  exec line, reachable from the network. The container must reproduce the native proven
  run, not merely "start."

## The bare-metal run (native)

The canonical `install.sh` path (no container): the same engine, the same pinned
dependencies, the same `ops/serve.env` contract, driven by `install.sh` →
`ops/start.sh`. Weights and the lookup corpus are read in place from the host's
Hugging Face cache; nothing is baked or copied locally.

## Why native first

The one thing neither the native nor the Docker path has exercised yet is the
`sm_121a` skinny-kernel JIT on first load. Running the canonical installer first
isolates that: if it comes up at full power, the engine + weights + corpus are
proven on the board and the Docker layer reduces to its remaining wrapper concerns;
if it fails, the engine itself is the problem, not the container.

## Verified before the run

- The four model repositories are publicly readable (the installer's Hugging Face
  pre-flight passes with no token).
- The upstream repository is cloneable; the stock `install.sh` pins `v0.3.0`.
- The weight snapshots are complete in the Hugging Face cache (~50 GiB of
  `safetensors`, not cache pointers), so the install reuses them — the installer
  reports "about 0 GiB of downloads" into the cache.

## Run state

- Installer: stock `v0.3.0` from the upstream repository (not a fork).
- Hugging Face cache pointed at the existing on-disk cache, so weights are read
  in place (read-only) and not re-downloaded.
- Fresh install directory, separate from the Docker data volume (no collision).
- One engine per board: any previous engine/container is stopped before the run.

## Run outcome (first full-power attempt)

The stock `v0.3.0` run got through steps 1–6 and **failed at step 7 (start), before
`/health`**, on a missing image dependency:

```
ImportError: AutoImageProcessor requires the PIL library ... and the Torchvision
             library ... (server/app.py -> _load_vision -> server/images.py)
```

Steps that **succeeded**:

- 1/7 checks, 2/7 engine source at `v0.3.0`, 3/7 venv (`torch 2.13.0+cu130`,
  `triton 3.7.1`, `transformers 5.12.1`).
- 4/7 weights — **read in place from the on-disk cache, no download**; the NVFP4
  projections quantised (`mlp 17.11 GB fp8 -> 9.63 GB nvfp4`, etc.) and RAM rebounded
  after load. This validates the weights-served-from-the-share goal.
- 5/7 corpus — built host-side, `32,181,359 tokens, 257 MB`.
- 6/7 config — `serve.env` (StairCut on) + the `bin/tandem` control script.

The `sm_121a` kernel JIT and the full-context `/health` have **not** been exercised yet;
the vision crash happens first.

## Full-power success (after the `--vision off` workaround)

Re-launched the engine with the stock `v0.3.0` flags **plus `--vision off`** (the repo's
supported text-only path; no repo or venv edits). It got past the vision step and came
up at full power:

- **`/health` → HTTP 200**, `"max_len": 262144` (the full 262k context is active),
  `status: ok`, `engine_busy: false`.
- **Memory**: `allocated_gb 56.1`, `reserved_gb 56.6` — the expected full-power stack
  (weights + 262k KV + 8 GB cache). Available RAM bottomed at ~56 GiB and **held**
  (no OOM, no wedge); the guard's free-RAM poller stayed healthy the whole load.
- **`sm_121a` kernels JIT-compiled and running**: `verify graphs: 120 captured in 19.1 s`,
  drafter `lenrouter depth=31` reporting live verify/draft timings (~99 / ~25 ms). The
  skinny NVFP4 kernels used fallback tiles for a few un-profiled shapes (expected on a
  fresh load; not an error).
- **Weights read in place from the on-disk cache** (NVFP4 + FP8-head quantised from the
  `/mnt/...` snapshot paths), no download. The `--vision off` flag was the only deviation
  from the stock exec line.

This validates the native path end-to-end: engine + weights + corpus + full 262k power
all work on the board. The `sm_121a` first-load JIT — the open unknown — is now confirmed
to work.

## Smoke test + benchmark (full power, 512→512, c1/2/4)

- **Smoke test:** a real chat completion returned coherent, correct text (~42 tok/s
  decode on a 120-token answer; reasoning-tags path working, `usage` reported correctly).
- **llama-benchy** (uvx, `--pp 512 --tg 512 --concurrency 1 2 4 --depth 0 --runs 2
  --latency-mode generation --no-cache`), saved to `result.json`:

  | | c1 | c2 | c4 |
  |---|---|---|---|
  | prompt processing (pp, t/s) | 804.9 ± 57 | 456.9 ± 413 (per-req) | 226.7 ± 345 (per-req) |
  | token generation (tg, t/s, total) | 63.1 ± 3.6 | 51.0 ± 1.5 | 51.1 ± 1.7 |
  | tg per-request (t/s) | 63.1 | 52.8 ± 2.5 | 57.2 ± 6.9 |
  | peak tg (t/s) | 63.5 | 55.0 | 68.0 |
  | e2e TTFT (ms) | 653 ± 36 | 5,948 ± 5,305 | 16,020 ± 11,716 |

  Read: **fast decode, strong single-stream** (~63 tok/s solo, holding ~51-57 tok/s per
  request at c4, peak 68 — above the CHANGELOG ~44 baseline; spec-decoding working).
  **Prefill-bound under concurrency:** this is a single-stream engine (`max_queue 8`,
  serial queue by design), so c4 mainly overlaps decode with the next prefill; 4×512-token
  prefills queue up → TTFT ~16 s. Aggregate `pp` (total) is misleading with `--no-cache`
  + non-locked-step prefills; the per-request pp and the TTFT are the honest signals.
  **Memory held** ~54-56 GiB allocated (59.5 post-bench), no OOM/wedge under c4.

## Findings (for later)

- **Root cause: a repo bug, not the setup.** `server/app.py` loads the vision tower
  unconditionally at startup (`_load_vision`), and that path calls
  `AutoImageProcessor.from_pretrained`, which needs `PIL` + `torchvision`. But the
  installer's pinned `LOCK` (`requirements.lock`) pins **neither** — confirmed absent
  from the built venv and from `pip list` (not even a transitive dep). So any bare
  `install.sh` at full power hits this `ImportError` on the current repo.
- **`--vision off` is the clean workaround (verified).** In `_load_vision`, when
  `--vision off`, `vcfg` is `None` and the function `return`s *before* the
  `AutoImageProcessor` call. The two module-level imports the running process takes —
  `server/app.py:62 from server import images` and the lazy `from engine.vision import`
  — do **not** require PIL/torchvision (`server/images.py` imports PIL only inside a
  function, guarded; `engine/vision.py` needs only `torch`). Direct test:
  `import server.images, engine.vision` succeeds without those deps. So `--vision off`
  reaches full power today without editing the repo or the venv. (Text-only; image input
  disabled.)
- **Alternative workaround:** `pip install pillow torchvision` into the existing venv
  then restart — keeps image input, but is a local venv change, not the repo's path.
- **No public in-flight fix to pull.** `main == v0.3.0 == c11aeca` (2026-09-30); the
  author's "reworked setup" / "huge update" is **not pushed yet** (he says Mon/Tue).
  GitHub has **0 open PRs and 0 open issues**; the only other branches are
  `kolibri-experimental` (kernels/drafter, explicitly stopped/not merged) and
  `router-eng169` (StairCut docs, already in main). Neither pins the missing deps.
  CHANGELOG stops at 0.3.0 with no forward note.
- **Next step to actually test full power:** re-run the engine with `--vision off`
  (via `bin/tandem` or the exec line) under the memory guard; watch for the `sm_121a`
  JIT + first `/health` at `--max-len 262144`. Drop the page cache first if the
  weights are on a network share.

## Phase 2 — Docker: goal and plan

**Goal (the north star's current phase):** make the *exact* bare-metal run that is
currently working on the board runnable as a container — same weights source, same
full-power exec line, reachable from the network (so a tunnel from the other host works).
"Runnable" means it comes up at full power like native did, not just that the container
starts. Honest framing: Docker differs from native in one *deliberate* way — the engine
binds `0.0.0.0` (not the native `127.0.0.1`) so it is reachable from the host network;
loopback would put it in the container's own netns, unreachable. Everything else must
match the proven native line.

**Verified state before starting:**
- The deployed board image's `docker/entrypoint.sh` (203 lines) is the same as the local
  file, and **both lack the three flags the proven native run used** (see blockers).
- The image's baked `ops/serve.env` already carries the full-power env
  (`MAX_LEN=262144`, `LEN_LATCH=1`, `DROP_IDLE=1`, `SAMPLED_TREE=det`, `BUDGET=16`,
  `CACHE_GB=8`, `QWEN38_DEEP=32`, `QWEN38_LEN_SWITCH=1`, all QWEN38_* kernel flags), so
  no custom `serve.env` mount is needed.
- Weights: `/var/tandemllm/hf` is a real dir; `/var/tandemllm/hf/hub` is a **symlink** to
  the real NFS dir backing the weights (e.g. `/mnt/<weights-share>/common/models/hub-tandem/hub`). Bind-mounts do **not**
  follow symlinks, so the container must mount the *real* NFS dir.

**The three blockers** (entrypoint exec line vs the proven native `ops/start.sh` line —
confirmed by direct comparison; all fixes live in `docker/entrypoint.sh`, which is fair
play — not on the gold-rule list):
1. `--drop-idle` — not translated from `DROP_IDLE=1`; without it the container runs a
   non-StairCut served profile.
2. `--sampled-tree=det` — not translated from `SAMPLED_TREE=det`; sampled requests would
   use the q-aware chain instead of the exact det tree.
3. `--vision off` — absent, no way to pass it; the stock start crashes at `_load_vision`
   (pillow/torchvision missing), exactly like the native run did.

**Plan (ordered):**
1. Fix `docker/entrypoint.sh`: add `DROP_FLAG` (`--drop-idle`), `SAMPLED_FLAG`
   (`--sampled-tree=det` from `SAMPLED_TREE`), and `VISION_FLAG` (default `--vision off`;
   `VISION=on` only once the image ships the image deps) — mirroring `ops/start.sh`.
2. Rebuild the arm64 image → save tar → transfer → `docker load` on the board.
3. Stop the native engine (kill its PID; confirm RAM frees to ≥ 64 GiB — one-engine rule).
4. `docker compose up` with the **real NFS bind-mount**:
   `- /mnt/<weights-share>/common/models/hub-tandem:/var/tandemllm/hf:ro`   # the real NFS dir, not the symlink
   (`HF_HOME` at its default → `/var/tandemllm/hf/hub` resolves to the NFS weights),
   keep the `tandem-data` volume (holds the `sm_121a` JIT cache), `ipc: host`, GPU all.
5. Watch the container log to `/health` at `max_len 262144`; then a smoke test (as native).

**Risks (mitigable):** NFS startup latency (the ~50 GiB load over NFS is slow — don't
mistake a slow read for a dead process); page-cache pressure (the wrapper lacks native's
post-health `drop_page_cache.py` — run it in-container after health, or keep requests
conservative); JIT cache (keep the `tandem-data` volume); one-engine overlap (native PID
dead + ≥ 64 GiB free before `up`).

## Open

- **Full power not yet confirmed** — the first run died at the vision step; the
  `sm_121a` JIT + 262,144-token load are still the untested unknown.
- **Memory guard** — the run is watched with a free-RAM poller; at full context the
  engine reserves ~69 GiB of the board's 128 GiB. One engine per board (see
  `docs/operations.md`).
- **Weights over a network share** — read on every load; the page-cache drop
  (`tools/drop_page_cache.py`) matters when the cache is on a network share.
- **Report to the author** — the `LOCK`-vs-`_load_vision` mismatch so it's fixed in the
  reworked setup; `--vision off` is a stopgap until then.
