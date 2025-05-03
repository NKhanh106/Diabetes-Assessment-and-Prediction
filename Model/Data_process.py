import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import os

base_path = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(base_path, '../Data/diabetes_dataset.csv')

df = pd.read_csv(data_path)
list_columns = df.columns.tolist()

# Tìm giá trị số trong các cột và chuyển đổi chúng thành kiểu số
numerical_columns = []

for col in df.columns:
    coerced = pd.to_numeric(df[col], errors='coerce')
    valid_ratio = coerced.notna().mean()
    if valid_ratio >= 1.0:
        numerical_columns.append(col)

category_columns = list(set(list_columns) - set(numerical_columns) - set(['Target']))
encoder = OneHotEncoder(sparse_output=False)
encoded = encoder.fit_transform(df[category_columns])
encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(category_columns))

df_processed = pd.concat([df[numerical_columns], encoded_df], axis=1)
df_processed.astype(float)
df_processed['Target'] = df['Target']
df_processed.head(10)

df_processed.to_csv(os.path.join(base_path, '../Data/diabetes_dataset_processed.csv'), index=False)