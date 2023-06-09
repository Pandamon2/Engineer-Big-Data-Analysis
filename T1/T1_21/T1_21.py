import pandas as pd

df = pd.read_csv("D:/실습/T1/T1_21/basic1.csv")

df.head()

df.describe()

df = df[~(df['age'] <= 0)]

df = df[(df['age'] == round(df['age'],0))]
df.shape

pd.qcut(df['age'], q=3)
df['range'] = pd.qcut(df['age'], q=3, labels=['group1','group2','group3'])
df['range'].value_counts()

g1_med = df[df['range'] == 'group1']['age'].median()
g2_med = df[df['range'] == 'group2']['age'].median()
g3_med = df[df['range'] == 'group3']['age'].median()

print(g1_med + g2_med + g3_med)