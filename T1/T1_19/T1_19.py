import pandas as pd

df = pd.read_csv("D:/실습/T1/T1_19/basic2.csv", parse_dates = ['Date'])
df.info()

# 날짜 컬럼 생성
df['year'] = df['Date'].dt.year
df['month'] = df['Date'].dt.month
df['day'] = df['Date'].dt.day
df['dayofweek'] = df['Date'].dt.dayofweek

# 이벤트가 1인 sales값은 80%만 반영
def event_sale(x):
    if x['Events'] == 1:
        x['Sales2'] = x['Sales'] * 0.8

    else:
        x['Sales2'] = x['Sales']

    return x


df = df.apply(lambda x: event_sale(x), axis = 1)
df.head()

# 2022년 월별 합계 중 가장 큰 값
cond = df['year'] == 2022
df1 = df[cond]
sale1 = df1.groupby('month')['Sales2'].sum().max()
sale1


# 2023년 월별 합계 중 가장 큰 값
cond = df['year'] == 2023
df2 = df[cond]
sale2 = df2.groupby('month')['Sales2'].sum().max()
sale2

int(round(abs(sale1 - sale2),0))