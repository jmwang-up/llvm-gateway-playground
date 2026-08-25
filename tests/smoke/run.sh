#!/usr/bin/env bash
set -euo pipefail

smoke_tmp_dir="$(mktemp -d)"
cleanup() {
  docker compose --profile smoke down -v
  rm -rf -- "${smoke_tmp_dir}"
}
trap cleanup EXIT

export CLIENT_API_KEYS="smoke:smoke-key"
export DEEPSEEK_API_KEY="mock-key"
export DEEPSEEK_BASE_URL="http://mock-provider:9000/v1"

docker compose --profile smoke up -d --build redis mock-provider gateway

for attempt in $(seq 1 60); do
  if curl --fail --silent http://localhost:8000/ready > /dev/null; then
    break
  fi
  if [ "${attempt}" -eq 60 ]; then
    docker compose logs gateway mock-provider
    exit 1
  fi
  sleep 1
done

curl --fail --silent \
  -H 'X-API-Key: smoke-key' \
  -H 'Content-Type: application/json' \
  -d '{"model":"auto","messages":[{"role":"user","content":"hello"}],"stream":false}' \
  http://localhost:8000/chat > "${smoke_tmp_dir}/non-stream.json"

.venv/bin/python -c 'import json,sys; data=json.load(open(sys.argv[1])); assert data["provider"] == "deepseek"; assert data["message"]["content"] == "mock response"' "${smoke_tmp_dir}/non-stream.json"

curl --fail --silent --no-buffer \
  -H 'X-API-Key: smoke-key' \
  -H 'Content-Type: application/json' \
  -d '{"model":"auto","messages":[{"role":"user","content":"hello"}],"stream":true}' \
  http://localhost:8000/chat > "${smoke_tmp_dir}/stream.txt"

grep -q 'event: done' "${smoke_tmp_dir}/stream.txt"
echo "Smoke test passed: non-streaming and streaming /chat"

