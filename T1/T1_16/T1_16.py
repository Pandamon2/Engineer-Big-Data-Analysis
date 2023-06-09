import pandas as pd


df = pd.read_csv("D:/실습/T1/T1_16/basic1.csv")
# 'f2'가 0인 데이터를 age를 기준으로 오름차순 정렬
df = df[df['f2'] == 0].sort_values('age',ascending=True).reset_index(drop=True)

# 앞에서부터 20개 데이터 추출
df = df[:20]

# 결측치 채우기 전과 후 분산 저장
df_var1 = df['f1'].var()
df['f1'] = df['f1'].fillna(df['f1'].min())
df_var2 = df['f1'].var()

# 분산 차이를 소수점 둘째 자리까지 계산
print(round(df_var1 - df_var2, 2))