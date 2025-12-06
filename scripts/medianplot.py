# Median Score Comparison
import matplotlib.pyplot as plt
import numpy as np

base_medians = {
    "rouge": 0.3229461756,
    "bert_precision": 0.8749441504,
    "bert_recall": 0.9205292463,
    "bert_f1": 0.9004762769,
    "repetition": 0.4261603376
}

ft_medians = {
    "rouge": 0.4950099800,
    "bert_precision": 0.9303727150,
    "bert_recall": 0.9355383515,
    "bert_f1": 0.9327492118,
    "repetition": 0.3644859813
}

metrics = ["ROUGE-L", "Precision", "Recall", "F1", "Repetition"]
base_vals = [
    base_medians["rouge"],
    base_medians["bert_precision"],
    base_medians["bert_recall"],
    base_medians["bert_f1"],
    base_medians["repetition"]
]
ft_vals = [
    ft_medians["rouge"],
    ft_medians["bert_precision"],
    ft_medians["bert_recall"],
    ft_medians["bert_f1"],
    ft_medians["repetition"]
]

x = np.arange(len(metrics))
width = 0.35

plt.figure(figsize=(10,5))
plt.bar(x - width/2, base_vals, width, label="Base", color="slategray")
plt.bar(x + width/2, ft_vals, width, label="Fine-tuned", color="cornflowerblue")

plt.xticks(x, metrics)
plt.title("Median Score Comparison")
plt.ylabel("Score")
plt.ylim(0, 1)
plt.legend()
plt.grid(axis="y", linestyle="--", alpha=0.4)

plt.show()
