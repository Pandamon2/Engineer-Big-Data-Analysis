import pandas as pd

df = pd.read_csv("D:/실습/T1/T1_15/basic1.csv")

# 나이 기준으로 내림차순 정렬
df = df.sort_values('age', ascending=False).reset_index(drop=True)
print(df)

# 상위 20개 값
df = df[:20]
print(df)

# f1의 중앙값으로 f1의 결측치 채우기
df['f1'] = df['f1'].fillna(df['f1'].median())

cond = (df['f4'] == 'ISFJ') & (df['f5'] >= 20)

df[cond]['f1'].mean()