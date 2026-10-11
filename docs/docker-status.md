# Docker support — status

Side-by-side, optional path next to `install.sh`: the same engine, the same pinned
dependencies, the same `ops/serve.env` contract, packaged as an image whose weights
are never baked in.

> **North star (shared with `docs/bare-metal-status.md`):** one engine, full power,
> on the target GB10 DGX Spark, serving its weights in place from the on-disk cache,
> reachable and usable. This file is **Phase 2**: make the *exact* bare-metal run that
> is currently working on the board runnable as a container — same weights source,
> same full-power exec line. The native path (Phase 1) is done and is the reference.

## State

- **The wrapper is ready but not yet full-power-correct** — the `Dockerfile`,
  `docker/entrypoint.sh`, `docker-compose.yml`, and `docker/smoke.sh` exist and the
  image (`tandemllm:latest`, arm64, 29.4 GB) is already on the board. But the
  entrypoint's exec line was missing three flags the proven native run uses (below);
  the fix is applied to `docker/entrypoint.sh` and pending a rebuild.
- **Image** — built from `nvcr.io/nvidia/pytorch:26.07-py3`; a Python 3.11 venv pinned
  to `install.sh`'s `LOCK` (byte-identical in `docker/requirements.lock`); engine source
  at `/opt/tandemllm`; CUDA toolkit from the base. The baked `ops/serve.env` already
  carries the full-power env (`MAX_LEN=262144`, `LEN_LATCH`, `DROP_IDLE`, `SAMPLED_TREE`,
  all QWEN38_* kernel flags), so no custom `serve.env` mount is needed.
- **Entrypoint** — reproduces the `ops/qwen38-engine.service` exec line; GPU / nvcc /
  unified-memory checks at start time; corpus built into its volume on first start; weight
  snapshots resolved via the HF `refs/main` pointer (mtime fallback). The full-power
  flags `--drop-idle` / `--sampled-tree=det` / `--vision off` are now added (see below).
- **Run contract** — `docker-compose.yml` (host IPC, all GPUs, read-only weights mount,
  persistent `tandem-data` volume for corpus + kernel-JIT cache + logs), `docker/smoke.sh`
  (`/health` + one generation).
- **Reachability** — unlike native (`127.0.0.1`), the container engine binds `0.0.0.0` so
  it is reachable from the host network; a tunnel from the other host reaches it. This is
  the one *deliberate* difference from the native run.

## The three full-power flags the entrypoint was missing (now fixed)

The proven native exec line carries these; the entrypoint's exec line did not. All three
are in `docker/entrypoint.sh` (fair play — not on the gold-rule list), mirroring
`ops/start.sh`:

1. `--drop-idle` — from `DROP_IDLE=1` (served-profile / StairCut correctness).
2. `--sampled-tree=det` — from `SAMPLED_TREE=det` (exact det verify tree for sampled
   requests).
3. `--vision off` — the release's `_load_vision` needs pillow+torchvision (absent from the
   lock), so a stock start crashes there (exactly as the native run did); default off,
   `VISION=on` only once the image ships the image deps.

## Open (the run, in order)

1. **Rebuild + redeploy** — the deployed image still has the old entrypoint; rebuild
   arm64, save tar, transfer, `docker load`.
2. **Stop the native engine** — kill its PID, confirm RAM frees to ≥ 64 GiB (one-engine
   rule; native and Docker cannot both run at full power on 128 GiB).
3. **`docker compose up`** with the **real NFS bind-mount** (the weights' `hub` is a
   *symlink* to the real NFS dir, and bind-mounts don't follow symlinks):
   `- /mnt/<weights-share>/common/models/hub-tandem:/var/tandemllm/hf:ro`   # the real NFS dir, not the symlink.
4. **Watch to `/health`** at `max_len 262144`, then the smoke test (as native).

### Risks
- **NFS startup latency** — the ~50 GiB weight load over NFS is slow; a slow read is not a
  dead process (watch the log).
- **Page-cache drop** — the container entrypoint does not yet run native's post-health
  `tools/drop_page_cache.py`; a later large prefill over a network-share cache can exhaust
  memory. Mitigate by running it in-container after health, or keeping requests conservative.
- **JIT cache** — keep the `tandem-data` volume (it holds the `sm_121a` kernel cache; the
  JIT itself is proven to work from the native run).
- **One engine per board** — ~56 GiB (measured native) to 69 GiB reserved at full context;
  the board must be dedicated to whichever engine is up (see `docs/operations.md`).
