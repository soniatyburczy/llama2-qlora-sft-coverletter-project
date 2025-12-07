from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from tqdm import tqdm
from datasets import load_dataset
import pandas as pd
import numpy as np

# To login:
# huggingface-cli login

MERGED_PATH = "where merged model lives"
DATASET = "ShashiVish/cover-letter-dataset"

# Structure prompt
def format_example(ex):
    prompt = (
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
    
    completion = ex.get("Cover Letter", "").strip()

    return {
        "prompt": prompt,
        "completion": completion
    }

# Load tokenizer and both models
base_name = "meta-llama/Llama-2-7b-chat-hf"
merged_path = MERGED_PATH

tokenizer = AutoTokenizer.from_pretrained(base_name)

base_model = AutoModelForCausalLM.from_pretrained(
    base_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

ft_model = AutoModelForCausalLM.from_pretrained(
    merged_path,
    torch_dtype=torch.float16,
    device_map="auto"
)

# Prepare the test set
raw_test = load_dataset(DATASET, split="test")
test_dataset = raw_test.map(format_example, remove_columns=raw_test.column_names)

# Generate output
def generate_cover_letter(model, tokenizer, prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,
        )

    text = tokenizer.decode(output[0], skip_special_tokens=True)
    return text

# Run inference for each example and store results
results = []

for ex in tqdm(test_dataset):
    prompt = ex["prompt"]

    base_output = generate_cover_letter(base_model, tokenizer, prompt)
    ft_output = generate_cover_letter(ft_model, tokenizer, prompt)

    results.append({
        "prompt": prompt,
        "ground_truth": ex["completion"],
        "base_output": base_output,
        "fine_tuned_output": ft_output
    })

# Save predictions
df = pd.DataFrame(results)
df.to_csv('eval.csv', index=False)