import pandas as pd
import numpy as np

df = pd.read_csv("D:/실습/T1/T1_11/basic1.csv")
print(df.head(5))
print(df.isnull().sum())

# min-max scaler 1
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
df['f5_1'] = scaler.fit_transform(df[['f5']])

# min-max scaler 2
df['f5_2'] = df['f5'].transform(lambda x: ((x - x.min()) / (x.max() - x.min())))

print(df.head)

# 하위 5%, 상위 5% 값 구하기
lower = df['f5_1'].quantile(0.05)
print(lower)

upper = df['f5_1'].quantile(0.95)
print(upper)

print(lower + upper)