import pandas as pd

df = pd.read_csv("data/labelbox_export.csv")

print("Labelbox data loaded successfully")
print(df.head())
print(f"Total rows: {len(df)}")
print(f"Columns: {list(df.columns)}")
