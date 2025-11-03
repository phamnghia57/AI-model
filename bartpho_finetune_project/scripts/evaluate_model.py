import sys
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from datasets import load_dataset, Dataset
import pandas as pd
import os
import evaluate
import numpy as np

model_dir = 'vinai/bartpho-word'
test_csv = sys.argv[2] if len(sys.argv) > 2 else "data/processed/test.csv"

tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
device = "cuda" if __import__('torch').cuda.is_available() else "cpu"
model = model.to(device)

df = pd.read_csv(test_csv)
texts = df['text'].astype(str).tolist()
refs = df['summary'].astype(str).tolist()

batch_size = 8
rouge = evaluate.load("rouge")
preds = []
for i in range(0, len(texts), batch_size):
    batch = texts[i:i+batch_size]
    inputs = tokenizer(batch, return_tensors="pt", truncation=True, padding=True, max_length=1024).to(device)
    output = model.generate(inputs["input_ids"], attention_mask=inputs["attention_mask"], max_length=256, num_beams=4)
    decoded = tokenizer.batch_decode(output, skip_special_tokens=True)
    preds.extend([d.strip() for d in decoded])

result = rouge.compute(predictions=preds, references=refs, use_stemmer=True)
print({k: round(v*100,4) for k,v in result.items()})
