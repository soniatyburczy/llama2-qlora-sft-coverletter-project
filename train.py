import re
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model
from huggingface_hub import login

# Config
MODEL_NAME = "meta-llama/Llama-2-7b-chat-hf"
DATASET_NAME = "ShashiVish/cover-letter-dataset"
OUTPUT_DIR = "./llama2-7b-chat-coverletter-final"
MAX_LENGTH = 768
BATCH_SIZE = 1
GRAD_ACCUM = 4
NUM_EPOCHS = 4
LEARNING_RATE = 2e-4
USE_CHAT_FORMAT = False
CLEAN_DATASET = True

# To login:
# huggingface-cli login

device = "cuda" if torch.cuda.is_available() else "cpu"
# print(f"Using device: {device}", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# QLoRA config to reduce memory use
bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    bnb_8bit_use_double_quant=True,
    bnb_8bit_quant_type="nf4",
    llm_int8_threshold=6.0
)

# Load base model in 8-bit
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.float16
)

# Enable memory-efficient training
model.gradient_checkpointing_enable()
model.enable_input_require_grads()

# Apply LoRA adapters
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
# model.print_trainable_parameters()

# Load dataset
raw_train = load_dataset(f'{DATASET_NAME}', split="train")
raw_test  = load_dataset(f'{DATASET_NAME}', split="test")

# Remove trash 
def clean_cover_letter(text):
    if text is None:
        return ""

    parts = re.split(r"\n###\s*Sample", text)

    for p in parts:
        if "Dear" in p or len(p.strip()) > 100:
            cleaned = re.sub(r"#{2,}.*\n", "", p).strip()
            cleaned = re.sub(r"\n\s*\n+", "\n\n", cleaned)
            return cleaned

    cleaned = re.sub(r"\n\s*\n+", "\n\n", text).strip()
    return cleaned

# Prompt construction
def format_example(ex):
    system_prompt = (
        "You are an expert in professional communication. "
        "Write a polished, personalized cover letter based only on the following job description and resume."
    )

    base_prompt = (
        "### Job Description\n"
        f"Job Title: {ex.get('Job Title','')}\n"
        f"Company: {ex.get('Hiring Company','')}\n"
        f"Preferred Qualifications: {ex.get('Preferred Qualifications','')}\n\n"

        "### Applicant Resume\n"
        f"Name: {ex.get('Applicant Name','')}\n"
        f"Current Experience: {ex.get('Current Working Experience','')}\n"
        f"Past Experience: {ex.get('Past Working Experience','')}\n"
        f"Skills: {ex.get('Skillsets','')}\n"
        f"Qualifications: {ex.get('Qualifications','')}\n\n"

        "### Cover Letter\n"
        "Using the information above, write a professional, personalized cover letter."
    )

    completion_raw = ex.get("Cover Letter", "") or ""
    completion = clean_cover_letter(completion_raw) if CLEAN_DATASET else completion_raw.strip()

    # Optional: produce chat-style input
    if USE_CHAT_FORMAT:
        prompt = f"<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{base_prompt} [/INST] "
        return {"prompt": prompt, "completion": completion}

    prompt = base_prompt + "\n"
    return {"prompt": prompt, "completion": completion}

train_dataset = raw_train.map(format_example, remove_columns=raw_train.column_names)
test_dataset  = raw_test.map(format_example, remove_columns=raw_test.column_names)

# print(train_dataset[0])

# Tokenization
def tokenize_fn(ex):
    text = ex["prompt"] + ex["completion"] + tokenizer.eos_token

    tokens = tokenizer(
        text,
        truncation=True,
        max_length=MAX_LENGTH,
    )

    tokens["labels"] = tokens["input_ids"].copy()
    return tokens


train_dataset = train_dataset.map(tokenize_fn, remove_columns=["prompt", "completion"])
test_dataset  = test_dataset.map(tokenize_fn, remove_columns=["prompt", "completion"])

# Data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

# Training configuration
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    num_train_epochs=NUM_EPOCHS,
    learning_rate=LEARNING_RATE,
    logging_steps=20,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    fp16=True,
    optim="paged_adamw_32bit",
    warmup_ratio=0.03,
    report_to="none",
    gradient_checkpointing=True,
)

# Trainer
trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    data_collator=data_collator
)

print("Starting training...") 
trainer.train() 

# Save model
print("Saving final model...") 
model.save_pretrained(OUTPUT_DIR, safe_serialization=True) 
tokenizer.save_pretrained(OUTPUT_DIR) 
print("Model saved to:", OUTPUT_DIR)