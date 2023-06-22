import pandas as pd
import numpy as np

train = pd.read_csv("D:/실습/T2/T2_6/train.csv")
test = pd.read_csv("D:/실습/T2/T2_6/test.csv")

# 데이터 전처리
train.head()
test.head()

train.info()
test.info()

train.isnull().sum()
test.isnull().sum()

train['datetime'] = pd.to_datetime(train['datetime'])
test['datetime'] = pd.to_datetime(test['datetime'])

train['year'] = train['datetime'].dt.year
train['month'] = train['datetime'].dt.month
train['day'] = train['datetime'].dt.day

test['year'] = test['datetime'].dt.year
test['month'] = test['datetime'].dt.month
test['day'] = test['datetime'].dt.day

train = train.drop(['datetime'],axis= 1)
test = test.drop(['datetime'], axis = 1)

train.columns
test.columns

train = train.drop(['casual','registered'],axis=1)

target = train.pop('count')
target

# 데이터셋 분리
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(train, target, test_size = 0.2, random_state=2023)
X_train.shape, X_val.shape, y_train.shape, y_val.shape

# 선형회귀
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train, y_train)
pred = lr.predict(X_val)
r2_score(y_val, pred)

# 랜덤포레스트
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor()
rf.fit(X_train, y_train)
pred = rf.predict(X_val)
r2_score(y_val, pred)

# xgboost
from xgboost import XGBRegressor
xgb = XGBRegressor()
xgb.fit(X_train, y_train)
pred = xgb.predict(X_val)
r2_score(y_val, pred)

# test 데이터 예측
pred = rf.predict(test)
pred

