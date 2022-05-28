# -*- coding: utf-8 -*-
"""
titanic_problem.ipynb
## Data Analysis
"""

from google.colab import drive
drive.mount('/content/driver')

# Commented out IPython magic to ensure Python compatibility.
import pandas as pd
import matplotlib.pyplot as plt
# %matplotlib inline
train = pd.read_csv('/content/driver/MyDrive/Colab Notebooks/kaggle/titanic/data/train.csv')
train.head()

# Checking missing value
print(train.info())
print("\n")
print(train.isnull().sum())
'''
Missing value
>> Age, Cabin, Embarked
'''

# create chart for analysis
## correlation between 'Survived' & 'Colum of data'
# define functions
def show_pChart(df, col_name):
  colname_survived = crosstabMaker(train, col_name)
  pChartMaker(colname_survived)
  return colname_survived

def crosstabMaker(df, col_name):
  feature_survived = pd.crosstab(df[col_name], df['Survived'])
  feature_survived.columns = feature_survived.columns.map({0:"Dead", 1:"Alive"})
  return feature_survived

def pChartMaker(feature_survived):
  fRows, fCols = feature_survived.shape
  pCol = 3
  pRow = (fRows/pCol + fRows%pCol)
  plot_height = pRow * 2.5
  plt.figure(figsize=(8, plot_height))

  for row in range(fRows):
    plt. subplot(pRow, pCol, row+1)
    idx_name = feature_survived.index[row]
    plt.pie(feature_survived.loc[idx_name], labels=feature_survived.loc[idx_name], autopct='%1.1f%%')
    plt.title("{0:}' survived".format(idx_name))

  plt.show()

# Sex
s = show_pChart(train, 'Sex')
s
''' female > male '''

# Emaberked
e = show_pChart(train, "Embarked")
e
''' C > Q > S '''

# Name
## extract "Mr. Mrs. Miss...."
train['Title'] = train.Name.str.extract('([A-Za-z]+)\.')
train['Title'].value_counts()
# classify 'Title'
train['Title'] = train['Title'].replace(['Dr', 'Rev', 'Major', 'Col', 'Countess', 'Capt', 'Sir', 'Lady', 'Don', 'Dona', 'Jonkheer'], 'Other')
train['Title'] = train['Title'].replace('Mlle', 'Miss')
train['Title'] = train['Title'].replace('Mme', 'Mrs')
train['Title'] = train['Title'].replace('Ms', 'Miss')
train['Title'].value_counts()
t = show_pChart(train, 'Title')
t
''' Mrs > Miss > Master > Other > Mr '''

# Age
## fill the 'missing value' with 'average age' (calculate with 'Title' & 'Age')
## divide into 8-sections & create AgeCategory
### pd.qcut >>> equal-size buckets (same count // different length)
### pd.cut >>> equal-length buckets (same length // different count)
meanAge = train[['Title', 'Age']].groupby(['Title']).mean()
for index, row in meanAge.iterrows():
  nullIndex = train[(train.Title == index) & (train.Age.isnull())].index
  train.loc[nullIndex, 'Age'] = row[0]
train['AgeCategory'] = pd.qcut(train.Age, 8, labels=range(1,9))
train.AgeCagtegory = train.AgeCategory.astype(int)
a = show_pChart(train, 'AgeCategory')
a

# cabin
## fill the 'missing value' with 'N'
## create CabinCategory
train.Cabin.fillna('N', inplace=True)
train['CabinCategory'] = train['Cabin'].str.slice(start=0, stop=1)
train['CabinCategory'] = train['CabinCategory'].map({ "N":0, "C":1, "B":2, "D":3, "E":4, "A":5, "F":6, "G":7, "T":8 })
c = show_pChart(train, 'CabinCategory')
c
''' ??? '''

# Fare
## fill the 'missing value' with '0'
## divide into 8-sections & create FareCategory
train.Fare.fillna(0)
train['FareCategory'] = pd.qcut(train.Fare, 8, labels=range(1,9))
train.FareCategory = train.FareCategory.astype(int)
f = show_pChart(train, 'FareCategory')
f
''' propotional to the fare '''

# SibSp, Parch
## create 'family' & 'IsAlone'
train['Family'] = train['SibSp'] + train['Parch'] + 1
train.loc[train['Family'] > 4, 'Family'] = 5
train['IsAlone'] = 1
train.loc[train['Family'] > 1, 'IsAlone'] =0
f = show_pChart(train, 'Family')
f
''' 4 > 3 > 2 > 1 > 5 '''
isa = show_pChart(train, 'IsAlone')
isa
''' Alone > Family '''

# Ticket
## create 'TicketCategory
### factorize: mapping object to value (enumeration or category)
train['TicketCategory'] = train.Ticket.str.split()
train['TicketCategory'] = [i[-1][0] for i in train['TicketCategory']]
train['TicketCategory'] = train['TicketCategory'].replace(['8', '9', 'L'], '8')
train['TicketCategory'] = pd.factorize(train['TicketCategory'])[0] + 1
t = show_pChart(train, 'TicketCategory')
t
train.head()

"""
## Data preprocessing
"""

import pandas as pd
train = pd.read_csv('/content/driver/MyDrive/Colab Notebooks/kaggle/titanic/data/train.csv')
test = pd.read_csv('/content/driver/MyDrive/Colab Notebooks/kaggle/titanic/data/test.csv')

def feature_engineering(df):
  # Sex
  df['Sex'] = df['Sex'].map({'female': 0, 'male': 1})

  # Embarked
  df.Embarked.fillna('S', inplace=True)
  df['Embarked'] = df['Embarked'].map({'C': 0, 'Q':1, 'S':2})

  # Title
  df['Title'] = df.Name.str.extract('([A-Za-z]+)\.')
  df['Title'] = df['Title'].replace(['Dr', 'Rev', 'Major', 'Col', 'Countess', 'Capt', 'Sir', 'Lady', 'Don', 'Dona', 'Jonkheer'], 'Other')
  df['Title'] = df['Title'].replace('Mlle', 'Miss')
  df['Title'] = df['Title'].replace('Mme', 'Mrs')
  df['Title'] = df['Title'].replace('Ms', 'Miss')
  df['Title'] = df['Title'].replace({'Master': 0, 'Miss':1, 'Mr':2, 'Mrs':3, 'Other':4})

  # Age
  meanAge = df[['Title', 'Age']].groupby(['Title']).mean()
  for index, row in meanAge.iterrows():
    nullIndex = df[(df.Title == index) & (df.Age.isnull())].index
    df.loc[nullIndex, 'Age'] = row[0]

  df['AgeCategory'] = pd.qcut(df.Age, 8, labels=range(1,9))
  df.AgeCategory = df.AgeCategory.astype(int)

  # Cabin
  df.Cabin.fillna('N', inplace=True)
  df['CabinCategory'] = df['Cabin'].str.slice(start=0, stop=1)
  df['CabinCategory'] = df['CabinCategory'].map({ "N": 0, "C": 1, "B": 2, "D": 3, "E": 4, "A": 5, "F": 6, "G": 7, "T": 8 })

  # Fare
  df.Fare.fillna(0, inplace=True)
  df['FareCategory'] = pd.qcut(df.Fare, 8, labels=range(1,9))
  df.FareCategory = df.FareCategory.astype(int)

  #SibSp, Parch
  df['Family'] = df['SibSp'] + df['Parch'] + 1
  df.loc[df['Family'] > 4, 'Family'] = 5

  df['IsAlone'] = 1
  df.loc[df['Family'] > 1, 'IsAlone'] = 0

  # Ticket
  df['TicketCategory'] = df.Ticket.str.split()
  df['TicketCategory'] = [i[-1][0] for i in df ['TicketCategory']]
  df['TicketCategory'] = df['TicketCategory'].replace(['8', '9', 'L'], '8')
  df['TicketCategory'] = pd.factorize(df['TicketCategory'])[0] + 1

  df.drop(['PassengerId', 'Ticket', 'Cabin', 'Fare', 'Name', 'Age', 'SibSp', 'Parch'], axis=1, inplace=True)

  return df

train = feature_engineering(train)
test = feature_engineering(test)
train.info()
test.info()

"""
## Machine learning
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

feature = train.drop('Survived', axis=1).values
label = train['Survived'].values

# serparate train data set (train : validation = 6 : 4)
## <parameters>
## test_size: seperation rate
## stratify: seperatio point
## random_state: random seed
x_train, x_valid, y_train, y_valid = train_test_split(feature, label, test_size=0.4, stratify=label, random_state=0)

# Random Forest Classifier - 1
rf = RandomForestClassifier(n_estimators=50, criterion='entropy', max_depth=5, oob_score=True, random_state=10)
rf.fit(x_train, y_train)
prediction = rf.predict(x_valid)
length = y_valid.shape[0]
accuracy = accuracy_score(prediction, y_valid)
print(f'Total occupants: {length}')
print(f'Accuracy: {accuracy * 100:.3f}%')

# Random Forest Classifier - 2
## Run-time type : None (1hr 14min 7sec)
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold

RF_Classifier = RandomForestClassifier()

RF_Paramgrid = {
    'max_depth': [6, 8, 10, 15],
    'n_estimators': [50, 100, 300, 500, 700, 800, 900],
    'max_features': ['sqrt'],
    'min_samples_split': [2, 7, 15, 30],
    'min_samples_leaf': [1, 15, 30, 60],
    'bootstrap': [True, False],
    'criterion': ['gini', 'entropy']
}

k_fold = StratifiedKFold(n_splits=5)
RF_Classifiergrid = GridSearchCV(RF_Classifier, param_grid=RF_Paramgrid, cv=k_fold, scoring='accuracy', n_jobs=-1, verbose=1)

RF_Classifiergrid.fit(x_train, y_train)
rf = RF_Classifiergrid.best_estimator_


# Bset score
print("Best Score: ", RF_Classifiergrid.best_score_)
# Best Parameter
print("Best Parameter: ", RF_Classifiergrid.best_params_)
# Best Model
print("Best Model: ", RF_Classifiergrid.best_estimator_)

# Feature Importances
import matplotlib.pyplot as plt
from pandas import Series

feature_importances = rf.feature_importances_
fi = Series(feature_importances, index=train.drop(['Survived'], axis=1).columns)

plt.figure(figsize=(8,8))
fi.sort_values(ascending=True).plot.barh()
plt.xlabel('Feature importance')
plt.ylabel('Featrue')
plt.show()

"""## Algorithm Test
> RandomForestClassifier
> SVC
> LogisticRegression
> DecisionTreeClassifier
> KNeighborsClassifier
> GaussianNB
"""

# Algorithm for Supervised learning
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

feature = train.drop('Survived', axis=1).values
label = train['Survived'].values
x_train, x_valid, y_train, y_valid = train_test_split(feature, label, test_size=0.4, stratify=label, random_state=0)

def ml_fit(model):
  model.fit(x_train, y_train)
  prediction = model.predict(x_valid)
  accuracy = accuracy_score(prediction, y_valid)
  print(model)
  print(f'Total accupants: {y_valid.shape[0]}')
  print(f'Accuracy {accuracy * 100:.3f}%')
  return model
  

model = ml_fit(RandomForestClassifier(n_estimators=100))
model = ml_fit(LogisticRegression(solver='lbfgs'))
model = ml_fit(SVC(gamma='scale'))
model = ml_fit(KNeighborsClassifier())
model = ml_fit(GaussianNB())
model = ml_fit(DecisionTreeClassifier())

# Result
model = ml_fit(RandomForestClassifier(bootstrap=True, max_depth=8, oob_score=True, random_state=10, criterion='entropy', max_features='sqrt',
                       min_samples_split=7, n_estimators=800, min_samples_leaf=1))
