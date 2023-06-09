import pandas as pd
import numpy as np

df = pd.read_csv("D:/실습/T1/T1_3/basic1.csv")
df.head()

# 결측치 개수 확인
df.isnull().sum()
df.shape

# 결측치 비율 확인
df.isnull().sum() / df.shape[0]

# f3 컬럼 삭제
df = df.drop('f3', axis = 1)
df.isnull().sum() / df.shape[0]

# 도시별 중앙값
df['city'].unique()
s = df[df['city'] == '서울']['f1'].median()
b = df[df['city'] == '부산']['f1'].median()
d = df[df['city'] == '대구']['f1'].median()
k = df[df['city'] == '경기']['f1'].median()

# 도시별 중앙값으로 결측치 대체
df['f1'] = df['f1'].fillna(df['city'].map({'서울': s, '부산' : b, '대구' : d, '경기' : k}))
print(df['f1'].mean())