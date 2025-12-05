import pandas as pd

CSV_PATH = "evaluation.csv"
COLUMN1 = "base_output"
COLUMN2 = "fine_tuned_output"

df = pd.read_csv(CSV_PATH)

def clean_text(text):
    if not isinstance(text, str):
        return ""

    parts = text.split("### Cover Letter", 1)

    if len(parts) == 1:
        return text.strip()

    cleaned = parts[1].strip()

    instruction = "Using the information above, write a professional, personalized cover letter."
    cleaned = cleaned.replace(instruction, "").strip()

    return cleaned

df[COLUMN1] = df[COLUMN1].apply(clean_text)
df[COLUMN2] = df[COLUMN2].apply(clean_text)

df.to_csv(f"cleaned_{CSV_PATH}", index=False)
