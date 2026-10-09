#!/usr/bin/env bash
# First-run verification, the container mirror of install.sh's smoke test:
#   1. /health answers
#   2. /v1/models lists the served model
#   3. one short streamed chat completion comes back, with first-token latency and tok/s
#
# Runs on the host: `docker compose exec -T tandem bash docker/smoke.sh`
# (or `docker compose run --rm client bash docker/smoke.sh` -- the compose file has no
# client service; exec into the engine container, which carries curl and the venv).
set -euo pipefail
URL="http://127.0.0.1:${PORT:-8000}"
MODEL="${SERVED_MODEL:-qwen3.8-27b-tandem}"

# 1. health (the engine binds its port only once the weights are in; wait for that)
for i in $(seq 1 120); do
    if curl -sf -m 3 "$URL/health" >/dev/null 2>&1; then
        echo "[smoke] /health ok after ${i}0s"
        break
    fi
    [ "$i" = 120 ] && { echo "[smoke] not healthy after 1200s" >&2; exit 1; }
    sleep 10
done

# 2 + 3. one streamed completion (the installer's prompt, byte for byte)
python3 - "$URL" "$MODEL" <<'PY'
import json, sys, time, urllib.request
url, model = sys.argv[1], sys.argv[2]
with urllib.request.urlopen(url + "/v1/models", timeout=30) as r:
    models = [m["id"] for m in json.load(r)["data"]]
print(f"[smoke] /v1/models: {', '.join(models)}")
body = {"model": model, "stream": True, "stream_options": {"include_usage": True},
        "temperature": 0, "max_tokens": 256, "chat_template_kwargs": {"enable_thinking": False},
        "messages": [{"role": "user", "content":
                      "Write a Python function that returns the n-th Fibonacci number iteratively, "
                      "with a docstring and two doctests."}]}
req = urllib.request.Request(url + "/v1/chat/completions", data=json.dumps(body).encode(),
                             headers={"Content-Type": "application/json"})
t0 = time.perf_counter(); first = None; n = 0; usage = None; text = []
with urllib.request.urlopen(req, timeout=600) as r:
    for raw in r:
        line = raw.decode().strip()
        if not line.startswith("data:") or line == "data: [DONE]":
            continue
        chunk = json.loads(line[5:])
        if chunk.get("usage"):
            usage = chunk["usage"]
        for ch in chunk.get("choices") or []:
            piece = (ch.get("delta") or {}).get("content")
            if piece:
                if first is None:
                    first = time.perf_counter()
                n += 1
                text.append(piece)
t1 = time.perf_counter()
tokens = (usage or {}).get("completion_tokens") or n
if first is None or tokens < 2:
    sys.exit("[smoke] FAILED: no content came back")
print("[smoke] reply: " + "".join(text)[:160].replace("\n", " ") + " ...")
print(f"[smoke] {tokens} tokens, first token after {1000 * (first - t0):.0f} ms, "
      f"{(tokens - 1) / (t1 - first):.1f} tok/s decode (one request, a cold engine's first)")
PY
echo "[smoke] PASS"
