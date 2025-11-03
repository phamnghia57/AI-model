import sys
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import json

model_dir = sys.argv[1] if len(sys.argv) > 1 else "outputs/bartpho-finetuned"
input_file = sys.argv[2] if len(sys.argv) > 2 else None

tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

def summarize(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024).to(device)
    out = model.generate(inputs["input_ids"], attention_mask=inputs.get("attention_mask"), max_length=256, num_beams=4)
    return tokenizer.decode(out[0], skip_special_tokens=True)

if input_file:
    with open(input_file, "r", encoding="utf-8") as f:
        items = json.load(f)
    outputs = []
    for item in items:
        s = summarize(item.get("text",""))
        outputs.append({"id": item.get("id"), "summary": s})
    print(json.dumps(outputs, ensure_ascii=False, indent=2))
else:
    for line in sys.stdin:
        text = line.strip()
        if not text:
            continue
        print(summarize(text))
