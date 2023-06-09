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
df.isnull().sum()/len(df)

# Cabin 컬럼 제거
df = df.drop(['Cabin'], axis=1)
df.shape

# Embarked 결측치 제거
df.isnull().sum()
df1 = df[~(df['Embarked'].isnull())]
df1.shape

# age 결측치 평균대체법 사용
df1['Age'].mean()
df1.isnull().sum()
df1['Age'].fillna(df1['Age'].mean(), inplace=True)
df1.isnull().sum()

# 데이터셋 분리
X_train, X_test, y_train, y_test = exam_data_load(df1, target = 'Survived', id_name = 'PassengerId')
X_train.shape, X_test.shape, y_train.shape, y_test.shape

y_train['Survived'].value_counts()
y = y_train['Survived']

# 원핫 인코딩
features = ['Pclass','Sex','SibSp','Parch','Embarked','Age']
X = pd.get_dummies(X_train[features])
test = pd.get_dummies(X_test[features])

# 라벨 인코딩
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
for feature in features:
    X_train[feature] = label_encoder.fit_transform(X_train[feature])
    X_test[feature] = label_encoder.fit_transform(X_test[feature])

X.shape, test.shape

# 모델링 및 평가
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=10, max_depth=7, random_state=2021)
model.fit(X,y)
predictions = model.predict(test)

model.score(X, y)

model.score(test, y_test['Survived'])