import pandas as pd

# Compute without recalculating scores:
df = pd.read_csv("results_per_example_strict_filtered.csv")

# (in hindsight doing it this way is weird but...eh)
metrics = ["rougeL_f", "bert_precision", "bert_recall", "bert_f1", "repetition_ratio"]
stat_order = ["mean", "std", "median"]

stats = df.groupby("model")[metrics].agg(stat_order)

stats.columns = [f"{metric}_{stat}" for metric, stat in stats.columns]

ordered_cols = (
    [f"{m}_mean" for m in metrics] +
    [f"{m}_std" for m in metrics] +
    [f"{m}_median" for m in metrics]
)

stats = stats[ordered_cols]

# Save
stats.to_csv("strict_eval_summary.csv")
