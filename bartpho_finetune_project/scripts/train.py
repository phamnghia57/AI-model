import os
import sys
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq
from datasets import load_dataset, Dataset
import torch
import evaluate
import numpy as np
import pandas as pd
from typing import Dict

model_name = sys.argv[1] if len(sys.argv) > 1 else "vinai/bartpho-word"
data_dir = sys.argv[2] if len(sys.argv) > 2 else "data/processed"
output_dir = sys.argv[3] if len(sys.argv) > 3 else "outputs/bartpho-finetuned"
os.makedirs(output_dir, exist_ok=True)

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

def load_csv(path):
    df = pd.read_csv(path)
    return Dataset.from_pandas(df)

train_ds = load_csv(os.path.join(data_dir,"train.csv"))
val_ds = load_csv(os.path.join(data_dir,"val.csv"))
test_ds = load_csv(os.path.join(data_dir,"test.csv"))

max_input_length = 1024
max_target_length = 256
def preprocess_function(examples):
    inputs = examples["text"]
    targets = examples["summary"]
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=max_target_length, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

train_block = train_ds.map(preprocess_function, batched=True, remove_columns=train_ds.column_names)
val_block = val_ds.map(preprocess_function, batched=True, remove_columns=val_ds.column_names)
test_block = test_ds.map(preprocess_function, batched=True, remove_columns=test_ds.column_names)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

rouge = evaluate.load("rouge")
def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]
    return preds, labels

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
    result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    result = {k: round(v * 100, 4) for k, v in result.items()}
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
    result["gen_len"] = np.mean(prediction_lens)
    return result

training_args = Seq2SeqTrainingArguments(
    output_dir=output_dir,
    eval_strategy="epoch",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    predict_with_generate=True,
    do_train=True,
    do_eval=True,
    logging_strategy="steps",
    logging_steps=200,
    save_strategy="epoch",
    save_total_limit=3,
    num_train_epochs=4,
    learning_rate=3e-5,
    weight_decay=0.01,
    warmup_steps=500,
    fp16=torch.cuda.is_available(),
    gradient_accumulation_steps=4,
    generation_max_length=256,
    generation_num_beams=4
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_block,
    eval_dataset=val_block,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

trainer.train()
trainer.save_model(output_dir)
tokenizer.save_pretrained(output_dir)
