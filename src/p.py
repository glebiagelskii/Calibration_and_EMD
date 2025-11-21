import pandas as pd

df = pd.read_parquet("fitres_10_11_2_rep1.parquet")
print(df.info())      # Column names, dtypes, row count
print(df.head())