from transformers import pipeline

pipe = pipeline("text-generation", model="Qwen/Qwen3-0.6B")
messages = [
    {"role": "user", "content": "Who are you?"},
]
pipe(messages)
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B")
messages = [
    {"role": "user", "content": "Who are you?"},
]
inputs = tokenizer.apply_chat_template(
  messages,
  add_generation_prompt=True,
  tokenize=True,
  return_dict=True,
  return_tensors="pt",
).to(model.device)
outputs = model.generate(**inputs, max_new_tokens=40)
print(tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:]))
import os
import re
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from gtts import gTTS
from langdetect import detect

ASR_MODEL = "openai/whisper-small"
LLM_MODEL = "Qwen/Qwen3-0.6B"

def mask_pii(text: str) -> str:
    text = re.sub(r'\b\d{10}\b', '[PHONE_NUMBER]', text)
    text = re.sub(r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}', '[EMAIL]', text)
    return text

def build_prompt(user_text: str, lang_code: str) -> str:
    system_instruction = f"You are a helpful assistant. Always respond *ONLY* in {lang_code}."
    return f"### System:\n{system_instruction}\n\n### User:\n{user_text}\n\n### Assistant:\n"

def clean_output(output_text: str) -> str:
  """Removes text within <think> ... </think> tags using regex.

  Args:
    output_text: The string containing the model's output.

  Returns:
    The cleaned string with the text within <think> ... </think> tags removed.
  """
  cleaned_text = re.sub(r'<think>.*?</think>', '', output_text, flags=re.DOTALL)
  return cleaned_text

print("Loading ASR...")
asr_pipe = pipeline("automatic-speech-recognition", model=ASR_MODEL)

print("Loading Qwen locally...")
tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(LLM_MODEL, trust_remote_code=True, device_map="auto")

gen_pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device_map="auto")

def audio_to_llm_response(audio_file_path: str, output_tts_file="final_response.mp3"):
    
    transcription = asr_pipe(audio_file_path)["text"]
    print("ASR Output:", transcription)


    safe_text = mask_pii(transcription)
    print("Anonymized Text:", safe_text)


    try:
        lang_code = detect(safe_text)
    except:
        lang_code = "en"
    print("Detected Language:", lang_code)


    prompt = build_prompt(safe_text, lang_code)


    output = gen_pipe(prompt, max_new_tokens=256, do_sample=True, temperature=0.7)[0]["generated_text"]
    llm_reply = output[len(prompt):].strip()
    print("LLM Reply:", llm_reply)


    cleaned_llm_reply = clean_output(llm_reply)
    print("Cleaned LLM Reply:", cleaned_llm_reply)

    try:
        tts = gTTS(text=cleaned_llm_reply, lang=lang_code)
        tts.save(output_tts_file)
        print(f"Final safe speech saved as {output_tts_file}")
    except Exception as e:
        print("TTS failed:", e)
        with open(output_tts_file + ".txt", "w", encoding="utf-8") as f:
            f.write(cleaned_llm_reply)

    return {"transcription": transcription, "safe_text": safe_text, "llm_reply": cleaned_llm_reply}


if _name_ == "_main_":
    audio_file_path = "/content/hindi (1).mp3"
    result = audio_to_llm_response(audio_file_path, "safe_reply.mp3")
    print("Done:", result)
