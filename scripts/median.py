import pandas as pd
import csv

# Helper file for median from results per example

df = pd.read_csv("results_per_example.csv")

medians = df.groupby("model").median(numeric_only=True)

summary = pd.read_csv("eval_summary.csv")
summary = summary.set_index("model")

summary["rougeL_f_median"]           = medians["rougeL_f"]
summary["bert_precision_median"]     = medians["bert_precision"]
summary["bert_recall_median"]        = medians["bert_recall"]
summary["bert_f1_median"]            = medians["bert_f1"]
summary["repetition_ratio_median"]   = medians["repetition_ratio"]

summary.to_csv("summary_with_medians.csv")
