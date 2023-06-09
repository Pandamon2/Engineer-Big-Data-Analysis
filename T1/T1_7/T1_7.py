import pandas as pd
import numpy as np

df = pd.read_csv("D:/실습/T1/T1_7/basic1.csv")
df.shape

# f4가 ESFJ -> ISFJ로 변경
df['f4'] = df['f4'].replace('ESFJ','ISFJ')
df[df['f4'] == 'ESFJ']

# 조건에 맞는 값 추출
df[(df['city'] == '경기') & (df['f4'] == 'ISFJ')]['age'].max()