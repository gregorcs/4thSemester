import numpy as np
import pandas as pd

raw_dataFrame = pd.read_csv("profiles.csv")

num_rows = len(raw_dataFrame)
for i in raw_dataFrame.columns:
    col_data = raw_dataFrame[i]
    num_missing = col_data.apply(lambda x: pd.isnull(x)).sum()
    percent_missing = num_missing / num_rows * 100
    print(f"Column {i}: {num_missing} missing values ({percent_missing:.2f}%)")

clean_dataFrame = raw_dataFrame.replace('', np.nan).dropna()

# clean_dataFrame['income'] = clean_dataFrame['income'].replace(-1, pd.NA)
#
# clean_dataFrame = clean_dataFrame.dropna(subset=['income'])

num_rows = len(clean_dataFrame)
for i in clean_dataFrame.columns:
    col_data = clean_dataFrame[i]
    num_missing = col_data.apply(lambda x: x == '').sum()
    percent_missing = num_missing / num_rows * 100
    print(f"Column {i}: {num_missing} missing values ({percent_missing:.2f}%)")

clean_dataFrame.to_csv('cleaned_data.csv', index=False)
