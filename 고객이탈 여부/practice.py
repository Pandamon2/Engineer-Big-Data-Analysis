import numpy as np
import pandas as pd

# 데이터 불러오기
X_train = pd.read_csv("D:/실습/고객이탈 여부/X_train.csv")
X_test = pd.read_csv("D:/실습/고객이탈 여부/X_test.csv")
y_test = pd.read_csv("D:/실습/고객이탈 여부/y_test.csv")
y_train = pd.read_csv("D:/실습/고객이탈 여부/y_train.csv")

X_train.head()


# 목표 변수에 부적합한 설명 변수 제거
X_train = X_train.drop(['CustomerId','Surname'],axis =1)
X_test_id = X_test['CustomerId'].copy()
X_test = X_test.drop(['CustomerId'], axis = 1)
X_test = X_test.drop(['Surname'], axis = 1)

# 라벨인코딩
# object타입인 열만 선택
X_train.select_dtypes('object')

label_encoding_features = ['Geography', 'Gender']

from sklearn.preprocessing import LabelEncoder

for feature in label_encoding_features:
    le = LabelEncoder()
    X_train[feature] = le.fit_transform(X_train[feature])
    X_test[feature] = le.fit_transform(X_test[feature])

X_train.info()
X_train.head()

# 범주형 변수 더미변수로 변경
categorical_features = ['NumOfProducts', 'HasCrCard', 'IsActiveMember']

for feature in categorical_features:
    X_train[feature] = X_train[feature].astype('category')
    X_test[feature] = X_test[feature].astype('category')

X_train = pd.get_dummies(X_train)
X_test = pd.get_dummies(X_test)

X_test.head()
X_train.head()

# CreditScore를 5등분으로 나눠서 구간 인덱스를 나타냄
X_train['CreditScore_qcut'] = pd.qcut(X_train['CreditScore'], 5, labels = False)
X_test['CreditScore_qcut'] = pd.qcut(X_test['CreditScore'], 5, labels = False)


# 스케일링
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

from sklearn.model_selection import train_test_split
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train.iloc[:,1], random_state=2022, stratify =y_train.iloc[:,1])

# 모델링
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier

# 로지스틱 회귀
model1 = LogisticRegression()
model1.fit(X_train, y_train)
predicted1 = model1.predict_proba(X_valid)

# 랜덤포레스트
model2 = RandomForestClassifier()
model2.fit(X_train, y_train)
predicted2 = model2.predict_proba(X_valid)

# 의사결정나무
model3 = DecisionTreeClassifier()
model3.fit(X_train, y_train)
predicted3 = model3.predict_proba(X_valid)

# 소프트보팅
model4 = VotingClassifier(estimators = [('logistic', model1), ('random', model2)], voting = 'soft')
model4.fit(X_train, y_train)
predicted4 = model4.predict_proba(X_valid)

y_valid.shape, predicted1.shape, predicted2.shape, predicted3.shape

from sklearn.metrics import roc_auc_score

print("로지스틱 회귀분석 점수 : {}".format(roc_auc_score(y_valid, predicted1[:,1])))
print("랜덤포레스트 점수 : {}".format(roc_auc_score(y_valid, predicted2[:,1])))
print("의사결정나무 점수 : {}".format(roc_auc_score(y_valid, predicted3[:,1])))
print("앙상블 보팅 점수 : {}".format(roc_auc_score(y_valid, predicted4[:,1])))

from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators' : [50, 100, 150],
    'max_depth' : [5, 7]
}

gs = GridSearchCV(estimator = RandomForestClassifier(), param_grid = param_grid, cv = 10)
gs.fit(X_train, y_train)
gs.best_params_

# 적합한 초매개변수 설정 후 평가(랜덤포레스트)
model5 = RandomForestClassifier(max_depth = 7, n_estimators= 150)
model5.fit(X_train, y_train)
predicted5 = model5.predict_proba(X_valid)

print("로지스틱 회귀분석 점수 : {}".format(roc_auc_score(y_valid, predicted1[:,1])))
print("랜덤포레스트 점수 : {}".format(roc_auc_score(y_valid, predicted2[:,1])))
print("의사결정나무 점수 : {}".format(roc_auc_score(y_valid, predicted3[:,1])))
print("앙상블 보팅 점수 : {}".format(roc_auc_score(y_valid, predicted4[:,1])))
print("GridSearchCV 적용 랜덤포레스트 점수 : {}".format(roc_auc_score(y_valid, predicted5[:,1])))

X_test_id.shape

model6 = RandomForestClassifier(max_depth = 7, n_estimators= 150)
model6.fit(X_train, y_train)
predicted6 = model6.predict_proba(X_test)

result = pd.DataFrame({'CustomerId' : X_test_id, 'Exited' : predicted6[:,1]})

def result_validate(result):
    y_test = pd.read_csv("D:/실습/고객이탈 여부/y_test.csv")
    expected = y_test['Exited']
    predicted = result['Exited']

    print('ROC AUC score: ', roc_auc_score(expected, predicted))

result_validate(result)