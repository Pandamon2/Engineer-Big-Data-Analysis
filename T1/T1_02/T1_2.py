import pandas as pd
import numpy as np

df = pd.read_csv("D:/실습/T1/T1_2/basic1.csv")
df['age']

df = df[(df['age'] - np.floor(df['age'])) != 0]
df

m_ceil = np.ceil(df['age']).mean()
m_floor = np.floor(df['age']).mean()
m_trunc = np.trunc(df['age']).mean()

m_ceil, m_floor, m_trunc

print(m_ceil, m_floor, m_trunc)