import pandas as pd
import numpy as np

X_train = pd.read_csv("D:/실습/학생 성적 예측/X_train.csv")
X_test = pd.read_csv("D:\실습\학생 성적 예측\X_test.csv")
y_train = pd.read_csv("D:\실습\학생 성적 예측\y_train.csv")
y_test = pd.read_csv("D:\실습\학생 성적 예측\y_test.csv")

# 함수 이용해서 데이터셋 불러오기
col_i = ['X_train','X_test','y_train','y_test']
for i in col_i:
    globals()[i] = pd.read_csv(f"D:/실습/학생 성적 예측/{i}.csv")

X_train.head()
X_train.info()
y_train.info()

# 결측치 확인
X_train.isnull().sum()
X_test.isnull().sum()
y_train.isnull().sum()
y_test.isnull().sum()

X_train = X_train.drop(['StudentID'], axis=1)
X_test = X_test.drop(['StudentID'], axis=1)

y_train = y_train['G3']
y_test = y_test['G3']



# 스케일링
X_train.info()
n_cols = X_train.select_dtypes(exclude = 'object').copy().columns
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train[n_cols] = scaler.fit_transform(X_train[n_cols])
X_test[n_cols] = scaler.transform(X_test[n_cols])

X_train.head(), X_test.head()
# 라벨인코딩
from sklearn.preprocessing import LabelEncoder
o_cols = X_train.select_dtypes(include = 'object').copy().columns

for i in o_cols:
    le = LabelEncoder()
    X_train[i] = le.fit_transform(X_train[i])
    X_test[i] = le.transform(X_test[i])

X_train.head(), X_test.head()

# 모델링
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor
rf = RandomForestRegressor(random_state=2023)
rf.fit(X_train, y_train)

rf_pred = rf.predict(X_test)
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
rf_r2 = r2_score(y_test, rf_pred)

print("랜덤포레스트 rmse : " , rf_rmse)
print("랜덤포레스트 r2 : ", rf_r2)

lgbm = LGBMRegressor()
lgbm.fit(X_train, y_train)

lgbm_pred = lgbm.predict(X_test)
lgbm_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
lgbm_r2 = r2_score(y_test, rf_pred)

print("lgbm rmse: ", lgbm_rmse)
print("lgbm r2: ", lgbm_r2)