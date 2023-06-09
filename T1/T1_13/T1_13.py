import pandas as pd
import numpy as np

df = pd.read_csv("D:/실습/T1/T1_13/winequality-red.csv")

# 상관관계 확인
df_corr = df.corr()
df_corr = df_corr[:-1]

max_corr = abs(df_corr['quality']).max()
min_corr = abs(df_corr['quality']).min()

round(max_corr + min_corr, 2)