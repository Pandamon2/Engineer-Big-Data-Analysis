import pandas as pd

df = pd.read_csv("D:/실습/T1/T1_14/basic1.csv")
df.head()

# city, f4를 기준으로 f5의 평균값
df = df.groupby(['city','f4'])['f5'].mean()

# f5를 기준으로 상위 7개 값 출력
df = df.reset_index().sort_values('f5', ascending=False).head(7)

round(df['f5'].sum(),2)