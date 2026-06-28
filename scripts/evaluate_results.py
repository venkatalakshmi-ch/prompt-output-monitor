import pandas as pd
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix

BASE_DIR = Path(__file__).resolve().parent.parent

predictions_path = BASE_DIR / "results" / "predictions.csv"
metrics_path = BASE_DIR / "results" / "metrics.txt"
confusion_matrix_path = BASE_DIR / "results" / "confusion_matrix.csv"

df = pd.read_csv(predictions_path)

y_true = df["gold_label"]
y_pred = df["predicted_label"]

accuracy = accuracy_score(y_true, y_pred)

precision, recall, f1, _ = precision_recall_fscore_support(
    y_true,
    y_pred,
    average="macro",
    zero_division=0
)

report = classification_report(y_true, y_pred, zero_division=0)
cm = confusion_matrix(y_true, y_pred, labels=["positive", "negative", "neutral"])

cm_df = pd.DataFrame(
    cm,
    index=["actual_positive", "actual_negative", "actual_neutral"],
    columns=["pred_positive", "pred_negative", "pred_neutral"]
)

with open(metrics_path, "w") as f:
    f.write(f"Accuracy: {accuracy:.2f}\n")
    f.write(f"Macro Precision: {precision:.2f}\n")
    f.write(f"Macro Recall: {recall:.2f}\n")
    f.write(f"Macro F1: {f1:.2f}\n\n")
    f.write("Classification Report:\n")
    f.write(report)

cm_df.to_csv(confusion_matrix_path)

print("Evaluation completed successfully")
print(f"Accuracy: {accuracy:.2f}")
print(f"Macro F1: {f1:.2f}")
print(cm_df)