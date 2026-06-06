import logging
import pandas as pd

logging.basicConfig(level=logging.INFO)


class Exception:
    def __init__(self):
        pass


try:
    df = pd.read_parquet("E:\\Tensor-Engine\\examples\\NL-OOB\\fold_prediction\\data\\train-00000-of-00001.parquet")
    print("Columns:", df.columns.tolist())
    print("Head:\n", df.head(3))
except Exception as e:
    print(f"Error reading parquet: {e}")
