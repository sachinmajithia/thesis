"""
Standalone IndicTrans2 (indic-indic) translation worker.

Runs in its OWN Python environment (see requirements-indictrans2.txt),
separate from the main app's environment (which loads NLLB-200). This is
necessary because AI4Bharat's trust_remote_code=True model/tokenizer code
for IndicTrans2 was written against transformers 4.x, and is incompatible
with the transformers 5.x the main app otherwise uses - isolating this
worker in its own environment (with transformers<5.0 installed) sidesteps
that entirely instead of patching around each individual incompatibility.

Set up once with:
    python3 -m venv indictrans2_env
    indictrans2_env/bin/pip install -r requirements-indictrans2.txt

app.py's start_indictrans2_worker() spawns this script under that venv's
Python interpreter once at app startup, and talks to it for the lifetime
of the app - it is NOT re-spawned per request.

Protocol: one JSON object per line on stdin -> one JSON object per line on
stdout, flushed immediately after each response so the parent process
never blocks waiting on buffered output.
  - On startup, once the model is loaded: {"ready": true}
  - Each request:  {"hindi_sentence": "..."}
  - Each response: {"translation": "...", "confidence": 0.95}
                 or {"error": "<message>"}
All logging goes to stderr, never stdout, since stdout is reserved
entirely for the JSON response protocol.
"""

import sys
import json

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from IndicTransToolkit.processor import IndicProcessor

MODEL_NAME = "ai4bharat/indictrans2-indic-indic-1B"
SRC_LANG, TGT_LANG = "hin_Deva", "pan_Guru"


def log(message):
    print(message, file=sys.stderr, flush=True)


def translate(tokenizer, model, device, ip, hindi_sentence):
    batch = ip.preprocess_batch([hindi_sentence], src_lang=SRC_LANG, tgt_lang=TGT_LANG)
    inputs = tokenizer(batch, truncation=True, padding="longest", return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Same repetition-loop protection as the main app's NLLB path: beam
    # search plus repetition controls, and an output length capped
    # relative to the input instead of a large flat max_length.
    input_len = inputs["input_ids"].shape[1]
    max_new_tokens = min(200, max(20, input_len * 4))

    with torch.no_grad():
        generated_tokens = model.generate(
            **inputs,
            num_beams=4,
            no_repeat_ngram_size=3,
            repetition_penalty=1.3,
            max_new_tokens=max_new_tokens,
            early_stopping=True,
        )

    decoded = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
    translations = ip.postprocess_batch(decoded, lang=TGT_LANG)
    return translations[0]


def main():
    log(f"[indictrans2_worker] Loading {MODEL_NAME}...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"[indictrans2_worker] Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = model.to(device)
    ip = IndicProcessor(inference=True)

    log("[indictrans2_worker] Ready.")
    print(json.dumps({"ready": True}), flush=True)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        try:
            request = json.loads(line)
        except json.JSONDecodeError as e:
            print(json.dumps({"error": f"Invalid request JSON: {e}"}), flush=True)
            continue

        hindi_sentence = request.get("hindi_sentence", "")
        if not hindi_sentence:
            print(json.dumps({"error": "No hindi_sentence provided"}), flush=True)
            continue

        try:
            translation = translate(tokenizer, model, device, ip, hindi_sentence)
            print(json.dumps({"translation": translation, "confidence": 0.95}), flush=True)
        except Exception as e:
            log(f"[indictrans2_worker] Translation error: {type(e).__name__}: {e}")
            print(json.dumps({"error": f"{type(e).__name__}: {e}"}), flush=True)


if __name__ == "__main__":
    main()
