# 시험환경 세팅 (코드 변경 X)
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def exam_data_load(df, target, id_name="", null_name=""):
    if id_name == "":
        df = df.reset_index().rename(columns={"index": "id"})
        id_name = 'id'
    else:
        id_name = id_name

    if null_name != "":
        df[df == null_name] = np.nan

    X_train, X_test = train_test_split(df, test_size=0.2, random_state=2021)

    y_train = X_train[[id_name, target]]
    X_train = X_train.drop(columns=[target])

    y_test = X_test[[id_name, target]]
    X_test = X_test.drop(columns=[target])
    return X_train, X_test, y_train, y_test


df = pd.read_csv("D:/실습/T2/T2_1/train.csv")
X_train, X_test, y_train, y_test = exam_data_load(df, target='Survived', id_name='PassengerId')

X_train.shape, X_test.shape, y_train.shape, y_test.shape

X_train.head()
y_train.head()

# PassengerId 제외
X_train = X_train.drop(['PassengerId'], axis =1)
y_train = y_train.drop(['PassengerId'], axis =1)
X_test = X_test.drop(['PassengerId'], axis =1)
y_test = y_test.drop(['PassengerId'], axis = 1)

# 결측치 확인
X_train.isnull().sum()/X_train.shape[0]

# Cabin 제외
X_train = X_train.drop(['Cabin'], axis =1)
X_test = X_test.drop(['Cabin'], axis =1)

# Age 결측값 대체
X_train['Age'] = X_train['Age'].fillna(0)
X_test['Age'] = X_test['Age'].fillna(0)

# Embarked 결측치 최빈값으로 대체
X_train['Embarked'].value_counts()
X_train['Embarked'] = X_train['Embarked'].fillna('S')

X_test['Embarked'].value_counts()
X_test['Embarked'] = X_test['Embarked'].fillna('S')

X_train.info()

#원핫 인코딩
X = pd.get_dummies(X_train['Sex'])
test = pd.get_dummies(X_test['Sex'])

# 데이터 병합
X_train = pd.concat([X_train, X], axis = 1)
X_test = pd.concat([X_test, test], axis = 1)

# 컬럼 삭제
X_train = X_train.drop(['Sex','Name','Ticket'], axis =1)
X_test = X_test.drop(['Sex', 'Name','Ticket'], axis =1)

# 라벨인코딩
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
X_train['Embarked'] = le.fit_transform(X_train['Embarked'])
X_test['Embarked'] = le.transform(X_test['Embarked'])

# 스케일링
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train['Age'] = scaler.fit_transform(X_train[['Age']])
X_test['Age'] = scaler.transform(X_test[['Age']])

from sklearn.preprocessing import RobustScaler
Rs = RobustScaler()
X_train['Fare'] = Rs.fit_transform(X_train[['Fare']])
X_test['Fare'] = Rs.transform(X_test[['Fare']])

# 랜덤포레스트
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(max_depth=5)
model = rf.fit(X_train, y_train["Survived"])
model.score(X_train, y_train)
model.score(X_test, y_test)

# 로지스틱 회귀
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
model2 = lr.fit(X_train, y_train)
model2.score(X_train, y_train)
model2.score(X_test, y_test)

# Decision Tree
from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier()
model3 = dtc.fit(X_train, y_train)
model3.score(X_train, y_train)
model3.score(X_test, y_test)

# 모델 예측 확률
# model.predict_proba(X_test)
