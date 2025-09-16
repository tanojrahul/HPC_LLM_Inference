import os
import uuid
from pathlib import Path
from typing import Any, Dict

from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename

from model import ensure_loaded, chat_with_audio, ENGINES, build_prompt, clean_output


API_KEY = os.environ.get("API_KEY", "changeme")
UPLOAD_DIR = Path(os.environ.get("UPLOAD_DIR", "./uploads"))
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)


def require_api_key() -> bool:
	# Support both header styles: Authorization: Bearer <key> or x-api-key: <key>
	auth = request.headers.get("Authorization", "")
	if auth.startswith("Bearer "):
		token = auth.split(" ", 1)[1].strip()
		return token == API_KEY
	x_key = request.headers.get("x-api-key")
	if x_key and x_key == API_KEY:
		return True
	return False


@app.before_request
def check_key():
	if request.path in ("/healthz", "/readyz", "/"):
		return  # no auth for health endpoints
	if not require_api_key():
		return jsonify({"error": "Unauthorized"}), 401


@app.route("/")
def root():
	return jsonify({"status": "ok"})


@app.route("/healthz", methods=["GET"])  # liveness
def healthz():
	return jsonify({"status": "alive"})


@app.route("/readyz", methods=["GET"])  # readiness (models loaded)
def readyz():
	try:
		ensure_loaded()
		return jsonify({"status": "ready"})
	except Exception as e:
		return jsonify({"status": "loading", "error": str(e)}), 503


@app.route("/v1/chat/completions", methods=["POST"])
def chat_completions():
	ensure_loaded()
	data: Dict[str, Any] = request.get_json(force=True, silent=True) or {}

	messages = data.get("messages", [])
	max_tokens = int(data.get("max_tokens", 256))
	temperature = float(data.get("temperature", 0.7))
	# Optional: language hint
	lang = data.get("language")

	# Build a simple prompt from messages
	user_texts = [m.get("content", "") for m in messages if m.get("role") == "user"]
	user_text = "\n".join(user_texts).strip() or "Hello"
	prompt = build_prompt(user_text, lang or "en")

	raw = ENGINES.generate(prompt, max_new_tokens=max_tokens, temperature=temperature)
	content = clean_output(raw)

	# OpenAI-like response shape
	resp = {
		"id": f"chatcmpl-{uuid.uuid4().hex}",
		"object": "chat.completion",
		"model": os.environ.get("LLM_MODEL", "Qwen/Qwen3-0.6B"),
		"choices": [
			{
				"index": 0,
				"message": {"role": "assistant", "content": content},
				"finish_reason": "stop",
			}
		],
	}
	return jsonify(resp)


@app.route("/v1/audio/transcriptions", methods=["POST"])
def audio_transcriptions():
	ensure_loaded()

	# Expect multipart/form-data with a file field named 'file'
	if "file" not in request.files:
		return jsonify({"error": "Missing file field 'file'"}), 400
	file = request.files["file"]
	filename = secure_filename(file.filename or f"audio-{uuid.uuid4().hex}.wav")
	path = UPLOAD_DIR / filename
	file.save(path)

	# Optional language override
	language = request.form.get("language") or request.args.get("language")

	result = chat_with_audio(str(path), lang_code=language)

	# OpenAI Whisper-like shape (basic)
	return jsonify(
		{
			"text": result["text"],
			"language": result["language"],
			"transcription": result["transcription"],
			"safe_text": result["safe_text"],
		}
	)


def create_app() -> Flask:
	ensure_loaded()
	return app


if __name__ == "__main__":
	# For local dev only; production uses gunicorn
	port = int(os.environ.get("PORT", 8000))
	app.run(host="0.0.0.0", port=port, debug=False)

