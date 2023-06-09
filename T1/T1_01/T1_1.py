import pandas as pd
import numpy as np

df = pd.read_csv("D:/실습/T1/train.csv")
df.shape
df.head(2)

df['Fare']
# IQR
Q1 = df['Fare'].quantile(0.25)
Q3 = df['Fare'].quantile(0.75)
IQR = Q3-Q1

Q1-1.5*IQR, Q3+1.5*IQR

# 이상치 데이터 확인
outdata1 = df[df['Fare'] < (Q1 - 1.5 * IQR)]
outdata2 = df[df['Fare'] > (Q3 + 1.5 * IQR)]
len(outdata1), len(outdata2)
outdata = df[(df['Fare'] < (Q1 - 1.5 * IQR)) | (df['Fare'] > (Q3 + 1.5 * IQR))]
#
sum(outdata['Sex'] == 'female')