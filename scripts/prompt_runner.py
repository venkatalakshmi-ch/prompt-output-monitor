import pandas as pd

df = pd.read_csv("data/labelbox_export.csv")

with open("prompts/sentiment_v1.txt", "r") as f:
    prompt_template = f.read()

for _, row in df.iterrows():
    prompt = f"{prompt_template}\n\nText: {row['text']}"
    print("-----")
    print(prompt)

