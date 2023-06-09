import pandas as pd
import numpy as np
from sklearn.preprocessing import power_transform

df = pd.read_csv("D:/실습/T1/T1_10/basic1.csv")
df.head(5)

# 조건에 맞는 데이터
df = df[df['age'] >= 20]
print(df.shape)

# 최빈값으로 'f1'컬럼 결측치 대체
print("결측치 처리 전 : \n", df.isnull().sum())
print("최빈값: ", df['f1'].mode()[0])
df['f1'] = df['f1'].fillna(df['f1'].mode()[0])
print("결측치 처리 후: \n", df.isnull().sum())

# 'f1'데이터 여-존슨 yeo-johnson값 구하기
df['y'] = power_transform(df[['f1']])
df['y'].head()

# 'f1'데이터 박스-콕스 box-cox값 구하기
df['b'] = power_transform(df[['f1']], method = 'box-cox')
df['b'].head()

round(sum(np.abs(df['y'] - df['b'])),2)