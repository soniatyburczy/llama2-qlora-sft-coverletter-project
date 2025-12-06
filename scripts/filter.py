import pandas as pd

# Script to exclude cover letters under a certain length
# This is for quality: by only including long cover letters,
# it's fairer for the base model, which only generates long cover letters.
# Thus quality can be directly compared.

df = pd.read_csv("cleaned_eval.csv")
MIN_LEN = 500 # 500 strict (only long), 350 fair (will include some shorter letters, but mainly medium to long)

filtered = df[
    (df["ground_truth"].str.len() > MIN_LEN) &
    (df["base_output"].str.len() > MIN_LEN) &
    (df["fine_tuned_output"].str.len() > MIN_LEN)
]

filtered.to_csv("cleaned_eval_strict_filtered.csv", index=False)

keep_indices = filtered.index[
    (filtered["ground_truth"].str.len() > MIN_LEN) &
    (filtered["base_output"].str.len() > MIN_LEN) &
    (filtered["fine_tuned_output"].str.len() > MIN_LEN)
].tolist()

df_results = pd.read_csv("results_per_example.csv")

N = len(filtered)

expanded_indices = []

for i in keep_indices:
    expanded_indices.append(i)
    expanded_indices.append(i + N)

filtered_results = df_results.loc[expanded_indices]

filtered_results.to_csv("results_per_example_strict_filtered.csv", index=False)