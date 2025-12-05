import pandas as pd
from evaluate import load
import csv

# Load dataset
df = pd.read_csv("cleaned_evaluation.csv")

preds_columns = {
    "base": "base_output",
    "fine_tuned": "fine_tuned_output"
}

refs = df['ground_truth'].astype(str).tolist()

# Load metrics
rouge = load("rouge")
bertscore = load("bertscore")

# Repetition ratio
def repetition_ratio(text):
    text = str(text).lower()
    words = text.split()
    if len(words) == 0:
        return 0.0
    return 1 - (len(set(words)) / len(words))


# Per-example results file
per_example_file = "results_per_example.csv"

with open(per_example_file, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "model", "prediction", "reference",
            "rougeL_f",
            "bert_precision", "bert_recall", "bert_f1",
            "repetition_ratio",
        ]
    )
    writer.writeheader()

    for model_name, col_name in preds_columns.items():

        preds = df[col_name].astype(str).tolist()

        # Can batch BERTScore
        bert_results = bertscore.compute(
            predictions=preds,
            references=refs,
            lang="en"
        )

        bert_prec = bert_results["precision"]
        bert_rec  = bert_results["recall"]
        bert_f1   = bert_results["f1"]

        rep_scores = [repetition_ratio(p) for p in preds]

        for i in range(len(preds)):

            # Compute ROUGE-L per example
            rouge_score = rouge.compute(
                predictions=[preds[i]],
                references=[refs[i]]
            )["rougeL"]

            writer.writerow({
                "model": model_name,
                "prediction": preds[i],
                "reference": refs[i],
                "rougeL_f": rouge_score,
                "bert_precision": bert_prec[i],
                "bert_recall": bert_rec[i],
                "bert_f1": bert_f1[i],
                "repetition_ratio": rep_scores[i],
            })


df_results = pd.read_csv(per_example_file)
summary_rows = []

for model_name in df_results["model"].unique():
    df_m = df_results[df_results["model"] == model_name]

    summary_rows.append({
        "model": model_name,
        "rougeL_f_mean": df_m["rougeL_f"].mean(),
        "bert_precision_mean": df_m["bert_precision"].mean(),
        "bert_recall_mean": df_m["bert_recall"].mean(),
        "bert_f1_mean": df_m["bert_f1"].mean(),
        "repetition_ratio_mean": df_m["repetition_ratio"].mean(),

        "rougeL_f_std": df_m["rougeL_f"].std(),
        "bert_precision_std": df_m["bert_precision"].std(),
        "bert_recall_std": df_m["bert_recall"].std(),
        "bert_f1_std": df_m["bert_f1"].std(),
        "repetition_ratio_std": df_m["repetition_ratio"].std(),
    })

pd.DataFrame(summary_rows).to_csv("eval_summary.csv", index=False)

print("Per-example results saved to:", per_example_file)
print("Summary results saved to: eval_summary.csv")