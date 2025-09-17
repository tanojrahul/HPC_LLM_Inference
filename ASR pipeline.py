import os
import re
import sys
from pathlib import Path
from typing import List, Dict

from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from langdetect import detect

MODELS_DIR = Path(os.environ.get("MODELS_DIR", "./models")).resolve()
MODELS_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("HF_HOME", str(MODELS_DIR))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(MODELS_DIR))
os.environ.setdefault("TRANSFORMERS_CACHE", str(MODELS_DIR))

ASR_MODEL = "openai/whisper-small"
LLM_MODEL = "Qwen/Qwen3-0.6B"


def mask_pii(text: str) -> str:
    text = re.sub(r"\b\d{10}\b", "[PHONE_NUMBER]", text)
    text = re.sub(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", "[EMAIL]", text)
    return text


def clean_output(output_text: str) -> str:
    cleaned_text = re.sub(r"<think>.*?</think>", "", output_text, flags=re.DOTALL)
    return cleaned_text.strip()


def llm_generate_chat(tokenizer: AutoTokenizer, model: AutoModelForCausalLM, messages: List[Dict], max_new_tokens: int = 256, temperature: float = 0.7) -> str:
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        pad_token_id=tokenizer.eos_token_id,
    )
    generated = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
    return clean_output(generated)


def audio_to_llm_response(audio_file_path: str) -> Dict[str, str]:
    asr = pipeline("automatic-speech-recognition", model=ASR_MODEL)
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(LLM_MODEL, trust_remote_code=True, device_map="auto")

    transcription = asr(audio_file_path).get("text", "").strip()
    safe_text = mask_pii(transcription)
    try:
        lang_code = detect(safe_text) or "en"
    except Exception:
        lang_code = "en"

    messages = [
        {"role": "system", "content": f"You are a helpful assistant. Always respond ONLY in {lang_code}."},
        {"role": "user", "content": safe_text or "Hello"},
    ]
    reply = llm_generate_chat(tokenizer, model, messages, max_new_tokens=256, temperature=0.7)
    return {"transcription": transcription, "safe_text": safe_text, "llm_reply": reply}


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python ASR pipeline.py <audio_file_path>")
        sys.exit(1)
    audio_path = sys.argv[1]
    out = audio_to_llm_response(audio_path)
    print("Done:", out)

