import pandas as pd

df1 = pd.read_csv("evaluation_outputs.csv")
df2 = pd.read_csv("evaluation-279.csv")

merged_df = pd.concat([df1, df2])
merged_df.to_csv("evaluation.csv", index=False)