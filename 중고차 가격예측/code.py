import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

X_train = pd.read_csv("D:/실습/중고차 가격예측/X_train.csv")
X_test = pd.read_csv("D:/실습/중고차 가격예측/X_test.csv")
y_train = pd.read_csv("D:/실습/중고차 가격예측/y_train.csv")
y_test = pd.read_csv("D:/실습/중고차 가격예측/y_test.csv")

X_train.head()

# 결측치 확인
X_train.isnull().sum()
X_test.isnull().sum()

# 컬럼 중 object만 list로 변환하고 각각 고유값의 개수와 고유값의 빈도수 출력
for col in X_train.select_dtypes(object).columns.tolist():
    print(X_train[col].value_counts())
    print(X_train[col].nunique())
    print('------------')
for col in X_test.select_dtypes(object).columns.tolist():
    print(X_test[col].value_counts())
    print(X_test[col].nunique())
    print('----------------')


# train, test 데이터 합치고 원핫인코딩
X = pd.concat([X_train, X_test], axis=0)
get_dummy_X = pd.get_dummies(X)


# 인코딩 후 train, test로 나누기
X_train = pd.DataFrame(get_dummy_X).iloc[:4960,:]
X_test = pd.DataFrame(get_dummy_X).iloc[4960:,:]

X_test_id = X_test['carID']
X_train = X_train.drop(columns = 'carID')
X_test = X_test.drop(columns = 'carID')
y_train = y_train['price']


from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
X_train[['year','mileage','tax','mpg','engineSize']] = ss.fit_transform(X_train[['year','mileage','tax','mpg','engineSize']])
X_test[['year','mileage','tax','mpg','engineSize']] = ss.fit_transform(X_test[['year','mileage','tax','mpg','engineSize']])

from sklearn.model_selection import train_test_split
X_TRAIN, X_TEST, Y_TRAIN, Y_TEST = train_test_split(X_train, y_train, test_size=0.25, random_state=42)

from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor()
rf.fit(X_TRAIN, Y_TRAIN)
Y_TEST_predict = rf.predict(X_TEST)
y_test_predict = rf.predict(X_test)

from xgboost import XGBRegressor
xg = XGBRegressor()
xg.fit(X_TRAIN, Y_TRAIN)
Y_TEST2_predict = xg.predict(X_TEST)
y_test2_predict = xg.predict(X_test)

from sklearn.metrics import r2_score
print(r2_score(Y_TEST, Y_TEST_predict))
print(r2_score(Y_TEST, Y_TEST2_predict))

print(r2_score(y_test['price'], y_test_predict))
print(r2_score(y_test['price'], y_test2_predict))