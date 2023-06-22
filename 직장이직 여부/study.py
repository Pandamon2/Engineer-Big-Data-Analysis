import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

X_train = pd.read_csv("D:/실습/직장이직 여부/X_train.csv")
X_test = pd.read_csv("D:/실습/직장이직 여부/X_test.csv")
y_train = pd.read_csv("D:/실습/직장이직 여부/y_train.csv")
y_test = pd.read_csv("D:/실습/직장이직 여부/y_test.csv")

X_train.head()
y_train.head()

X_train.shape, X_test.shape, y_train.shape, y_test.shape
X_train.info()

for col in X_train.select_dtypes(object).columns.tolist():
    print('-----------------')
    print(col)
    print(X_train[col].nunique())
    print(X_train[col].value_counts())

for col in X_test.select_dtypes(object).columns.tolist():
    print('-----------------')
    print(col)
    print(X_test[col].nunique())
    print(X_test[col].value_counts())

# 데이터 전처리

# city는 많아서 뺌
X_train = X_train.drop(['enrollee_id', 'city'], axis= 1)
X_test_id = X_test['enrollee_id']
X_test = X_test.drop(['enrollee_id', 'city'], axis = 1)
y_train = y_train['target']
y_test = y_test['target']
print(y_train.value_counts())


print(X_train.isnull().sum())
print("------------------")
print(X_test.isnull().sum())

print(X_train.isnull().sum()/X_train.shape[0])
# 결측치 비율 30% 넘는 컬럼 제외
X_train = X_train.drop(['company_size','company_type'], axis=1)

X_train

X_train['gender']= X_train['gender'].fillna('Other')
X_train.isnull().sum()
X_train['enrolled_university'].value_counts()
X_train['enrolled_university'].mode()[0]
X_train.fillna({'enrolled_university': X_train['enrolled_university'].mode()[0],
               'education_level' : X_train['education_level'].mode()[0],
               'major_discipline' : X_train['major_discipline'].mode()[0],
               'experience' : X_train['experience'].mode()[0],
               'last_new_job' : X_train['last_new_job'].mode()[0]}, inplace = True)

X_train.isnull().sum()

X_test = X_test.drop(['company_size','company_type'], axis=1)

X_test

X_test['gender']= X_test['gender'].fillna('Other')
X_test.isnull().sum()
X_test['enrolled_university'].value_counts()
X_test['enrolled_university'].mode()[0]
X_test.fillna({'enrolled_university': X_test['enrolled_university'].mode()[0],
               'education_level' : X_test['education_level'].mode()[0],
               'major_discipline' : X_test['major_discipline'].mode()[0],
               'experience' : X_test['experience'].mode()[0],
               'last_new_job' : X_test['last_new_job'].mode()[0]}, inplace = True)

X_test.isnull().sum()

# AutoML
from pycaret.datasets import get_data
from pycaret.classification import *
train = pd.merge(X_train, y_train, on ='enrollee_id', how= 'inner')
test = pd.merge(X_test, y_test, on = 'enrollee_id', how= 'inner')
data = pd.concat([train,test], axis=0)
data["target"] = data["target"].astype("str")

train = pd.concat([X_train, y_train], axis = 1)
test = pd.concat([X_test, y_test], axis = 1)
data = pd.concat([train, test], axis = 0)

# <, > 값 변경
data.info()
data['experience'].value_counts()
data['experience'] = data['experience'].replace('>20', 'upper20')
data['experience'] = data['experience'].replace('<1', 'lower1')

data['last_new_job'].value_counts()
data['last_new_job'] = data['last_new_job'].replace('>4', 'upper4')

data.shape
data.info()
exp_clf101 = setup(data=data, target = 'target', train_size=0.65, index = False)
top5 = compare_models(sort = 'Accuracy', n_select = 5)


# 스케일링
X_train.shape, X_test.shape, y_train.shape, y_test.shape
X_train = X_train.drop(['enrollee_id'],axis=1)
X_test = X_test.drop(['enrollee_id'],axis=1)
X_train.shape, X_test.shape

from sklearn.preprocessing import RobustScaler
X_train['training_hours'].value_counts()
from matplotlib import pyplot as plt
plt.boxplot(X_train['training_hours'])
q1 = X_train['training_hours'].quantile(0.25)
q3 = X_train['training_hours'].quantile(0.75)
iqr = q3-q1
q1 - 1.5 * iqr, q3+ 1.5*iqr

X_train.info()
n_cols = ['city_development_index','training_hours']
scaler = RobustScaler()
X_train[n_cols] = scaler.fit_transform(X_train[n_cols])
X_test[n_cols] = scaler.transform(X_test[n_cols])


# Encoding
from sklearn.preprocessing import LabelEncoder

c_cols = ['gender','relevent_experience', 'enrolled_university', 'education_level', 'major_discipline', 'experience', 'last_new_job']

X_train = X_train.drop(['city'], axis = 1)
X_test = X_test.drop(['city'], axis = 1)

for col in c_cols:
    le = LabelEncoder()
    X_train[col] = le.fit_transform(X_train[col])
    X_test[col] = le.transform(X_test[col])

# modeling
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# Modeling
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train, y_train)
print(model.score(X_train, y_train))
print(model.score(X_test, y_test))