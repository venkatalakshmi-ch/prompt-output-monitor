import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

data_path = BASE_DIR / "data" / "labelbox_export.csv"
prompt_path = BASE_DIR / "prompts" / "sentiment_v1.txt"
output_path = BASE_DIR / "results" / "predictions.csv"

df = pd.read_csv(data_path)

with open(prompt_path, "r") as f:
    prompt_template = f.read()

def mock_llm(prompt):
    prompt_lower = prompt.lower()

    if "love" in prompt_lower or "amazing" in prompt_lower:
        return "positive"
    elif "bad" in prompt_lower or "terrible" in prompt_lower:
        return "negative"
    else:
        return "neutral"

results = []

for _, row in df.iterrows():
    prompt = f"{prompt_template}\n\nText: {row['text']}"
    prediction = mock_llm(prompt)

    results.append({
        "row_id": row["row_id"],
        "text": row["text"],
        "gold_label": row["gold_label"],
        "predicted_label": prediction,
        "batch_id": row["batch_id"]
    })

results_df = pd.DataFrame(results)
results_df.to_csv(output_path, index=False)

print("Mock LLM predictions saved successfully")
print(results_df)