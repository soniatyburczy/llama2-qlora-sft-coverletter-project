import pandas as pd

# Compute without re-evaluating metrics

df = pd.read_csv("results_per_example_filtered.csv")
metrics = ["rougeL_f", "bert_precision", "bert_recall", "bert_f1", "repetition_ratio"]

stats = df.groupby("model")[metrics].agg(["mean", "median", "std"])
pd.DataFrame(stats).to_csv("fair_eval_summary.csv", index=False)
