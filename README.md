# Local LLM Inference Server (Flask + GPU)

Runs Whisper ASR and Qwen LLM locally and exposes OpenAI-like endpoints with API key auth.

## Endpoints

- `GET /healthz`: liveness
- `GET /readyz`: readiness (ensures models are loaded)
- `POST /v1/chat/completions`: OpenAI-style chat completion
- `POST /v1/audio/transcriptions`: multipart upload (field `file`) -> transcribe+LLM

## Environment

- `API_KEY`: required to authorize requests (default `changeme`)
- `ASR_MODEL`: default `openai/whisper-small`
- `LLM_MODEL`: default `Qwen/Qwen3-0.6B`

## Run locally (no Docker)

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
$env:API_KEY = "mysecret"; $env:ASR_MODEL = "openai/whisper-small"; $env:LLM_MODEL = "Qwen/Qwen3-0.6B"
python server.py
```

## Test with PowerShell

```powershell
# Health
curl http://localhost:8000/healthz

# Chat
$body = @{
  messages = @(@{ role = "user"; content = "Who are you?" })
  max_tokens = 128
  temperature = 0.7
} | ConvertTo-Json

curl -Method Post `
  -Uri http://localhost:8000/v1/chat/completions `
  -Headers @{ Authorization = "Bearer mysecret"; 'Content-Type' = 'application/json' } `
  -Body $body

# Transcribe (multipart)
curl -Method Post `
  -Uri http://localhost:8000/v1/audio/transcriptions `
  -Headers @{ Authorization = "Bearer mysecret" } `
  -Form @{ file = Get-Item .\sample.wav }
```

## Docker (GPU)

Build and run with NVIDIA runtime:

```powershell
# Build
docker build -t local-llm:latest .

# Run (requires NVIDIA Container Toolkit on host)
docker run --rm -it `
  -p 8000:8000 `
  -e API_KEY=mysecret `
  --gpus all `
  local-llm:latest
```

Then call the endpoints as above against `http://localhost:8000`.

## Notes

- First run will download models to container cache (`/app/.hf_cache`).
- Adjust `--workers` in Docker CMD if you need concurrency; each worker loads the models into VRAM.
- Ensure sufficient GPU memory for the chosen models.
