import pandas as pd
import numpy as np

df = pd.read_csv("D:/실습/T1/T1_9/basic1.csv")

# 라이브러리 및 데이터 불러오기
from sklearn.preprocessing import StandardScaler
df['f5']

# 표준화
scaler = StandardScaler()
df['f5'] = scaler.fit_transform(df[['f5']])
print(df['f5'].median())