import matplotlib.pyplot as plt
import numpy as np

base_scores = {
    "rouge": 0.3143679378916034,
    "bert_precision": 0.8842039019467154,
    "bert_recall": 0.9196792709451692,
    "bert_f1": 0.9013214738116223,
    "repetition": 0.4282253913894293
}

ft_scores = {
    "rouge": 0.5188970823223441,
    "bert_precision": 0.9266420913630707,
    "bert_recall": 0.939011920489008,
    "bert_f1": 0.932643420545966,
    "repetition": 0.36145704097118836
}

# ROUGE-L Bar Chart
plt.figure(figsize=(6,5))
labels = ["Base", "Fine-tuned"]
values = [base_scores["rouge"], ft_scores["rouge"]]

plt.bar(labels, values, color=["slategray", "cornflowerblue"])
plt.title("ROUGE-L Score Comparison")
plt.ylabel("ROUGE-L")
plt.ylim(0, 1)
plt.grid(axis="y", linestyle="--", alpha=0.4)

plt.show()


# BERTScore Precision / Recall / F1
plt.figure(figsize=(8,5))

metrics = ["Precision", "Recall", "F1"]
base_vals = [
    base_scores["bert_precision"],
    base_scores["bert_recall"],
    base_scores["bert_f1"]
]
ft_vals = [
    ft_scores["bert_precision"],
    ft_scores["bert_recall"],
    ft_scores["bert_f1"]
]

x = np.arange(len(metrics))
width = 0.35

plt.bar(x - width/2, base_vals, width, label="Base", color="slategray")
plt.bar(x + width/2, ft_vals, width, label="Fine-tuned", color="cornflowerblue")

plt.xticks(x, metrics)
plt.title("BERTScore Comparison")
plt.ylabel("Score")
plt.ylim(0.7, 1)
plt.legend()
plt.grid(axis="y", linestyle="--", alpha=0.4)

plt.show()


# Repetition Ratio
plt.figure(figsize=(6,5))
labels = ["Base", "Fine-tuned"]
values = [base_scores["repetition"], ft_scores["repetition"]]

plt.bar(labels, values, color=["slategray", "cornflowerblue"])
plt.title("Repetition Ratio Comparison")
plt.ylabel("Repetition Ratio")
plt.ylim(0, 1)
plt.grid(axis="y", linestyle="--", alpha=0.4)

plt.show()
