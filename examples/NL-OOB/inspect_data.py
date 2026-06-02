import logging
import pandas as pd

logging.basicConfig(level=logging.INFO)

try:
    df = pd.read_parquet(
        "E:\\Tensor-Engine\\examples\\NL-OOB\\stability_prediction\\data\\train-00000-of-00001.parquet")
    print("Columns:", df.columns.tolist())
    print("Head:\n", df.head(3))
    print("Data Types:\n", df.dtypes)

    # Check for atom/coordinate like columns
    if 'atoms' in df.columns:
        print("Sample atoms:", df['atoms'].iloc[0])
    if 'coordinates' in df.columns:
        print("Sample coords shape/type:", type(df['coordinates'].iloc[0]))

except Exception as e:
    print(f"Error reading parquet: {e}")
