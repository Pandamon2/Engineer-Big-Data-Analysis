import pandas as pd
import numpy as np

df = pd.read_csv("D:/실습/T1/T1_4/train.csv")
df.shape
df.head(2)
df['SalePrice']

# 왜도, 첨도 구하기
from scipy.stats import skew, kurtosis

s1 = skew(df['SalePrice'])
k1 = kurtosis(df['SalePrice'])

# 로그 변환
df['logSalePrice'] = np.log1p(df['SalePrice'])
s2 = skew(df['logSalePrice'])
k2 = kurtosis(df['logSalePrice'])

print(round(s1 + s2 + k1 + k2, 2))