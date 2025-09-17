import os
from dotenv import load_dotenv
load_dotenv()
import re
from pathlib import Path
from typing import Dict, Any

from transformers import (
	pipeline,
	AutoTokenizer,
	AutoModelForCausalLM,
)
from io import BytesIO


# Ensure all models/cache live under a local ./models directory unless overridden
MODELS_DIR = Path(os.environ.get("MODELS_DIR", "./models")).resolve()
MODELS_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("HF_HOME", str(MODELS_DIR))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(MODELS_DIR))
os.environ.setdefault("TRANSFORMERS_CACHE", str(MODELS_DIR))

ASR_MODEL = os.environ.get("ASR_MODEL", "openai/whisper-small")
LLM_MODEL = os.environ.get("LLM_MODEL", "Qwen/Qwen3-0.6B")


def mask_pii(text: str) -> str:
	text = re.sub(r"\b\d{10}\b", "[PHONE_NUMBER]", text)
	text = re.sub(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", "[EMAIL]", text)
	return text


def build_prompt(user_text: str, lang_code: str) -> str:
	system_instruction = f"You are a helpful assistant. Always respond ONLY in {lang_code}."
	return (
		f"### System:\n{system_instruction}\n\n"
		f"### User:\n{user_text}\n\n"
		f"### Assistant:\n"
	)


def clean_output(output_text: str) -> str:
	cleaned_text = re.sub(r"<think>.*?</think>", "", output_text, flags=re.DOTALL)
	return cleaned_text.strip()


class InferenceEngines:
	def __init__(self) -> None:
		self.asr = None
		self.tokenizer = None
		self.model = None
		self.text_gen = None

	def load(self) -> None:
		# ASR
		self.asr = pipeline("automatic-speech-recognition", model=ASR_MODEL)

		# LLM
		self.tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL, trust_remote_code=True)
		self.model = AutoModelForCausalLM.from_pretrained(
			LLM_MODEL, trust_remote_code=True, device_map="auto"
		)
		self.text_gen = pipeline(
			"text-generation", model=self.model, tokenizer=self.tokenizer, device_map="auto"
		)


	def transcribe_audio(self, audio_file_path: str) -> str:
		result = self.asr(audio_file_path)
		return result.get("text", "")

	def generate(self, prompt: str, max_new_tokens: int = 256, temperature: float = 0.7) -> str:
		output = self.text_gen(
			prompt, max_new_tokens=max_new_tokens, do_sample=True, temperature=temperature
		)[0]["generated_text"]
		# Remove the prompt prefix
		return output[len(prompt) :].strip()



ENGINES = InferenceEngines()


def ensure_loaded() -> InferenceEngines:
	if ENGINES.asr is None:
		ENGINES.load()
	return ENGINES


def chat_with_audio(audio_path: str, lang_code: str | None = None) -> Dict[str, Any]:
	from langdetect import detect

	engines = ensure_loaded()
	transcription = engines.transcribe_audio(audio_path)
	safe_text = mask_pii(transcription)
	if not lang_code:
		try:
			lang_code = detect(safe_text) or "en"
		except Exception:
			lang_code = "en"
	prompt = build_prompt(safe_text, lang_code)
	raw_reply = engines.generate(prompt)
	cleaned = clean_output(raw_reply)
	return {
		"text": cleaned,
		"transcription": transcription,
		"safe_text": safe_text,
		"language": lang_code,
	}


def chat_with_text(user_text: str, lang_code: str | None = None, max_new_tokens: int = 256, temperature: float = 0.7) -> Dict[str, Any]:
	from langdetect import detect

	engines = ensure_loaded()
	safe_text = mask_pii(user_text or "")
	if not lang_code:
		try:
			lang_code = detect(safe_text) or "en"
		except Exception:
			lang_code = "en"
	prompt = build_prompt(safe_text, lang_code)
	raw_reply = engines.generate(prompt, max_new_tokens=max_new_tokens, temperature=temperature)
	cleaned = clean_output(raw_reply)
	return {"text": cleaned, "safe_text": safe_text, "language": lang_code}


def tts_synthesize(text: str, lang_code: str = "en") -> BytesIO:  # deprecated
	raise NotImplementedError("TTS disabled: using Whisper (ASR) and Qwen (LLM) only.")

