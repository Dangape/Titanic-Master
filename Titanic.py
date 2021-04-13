# Created by: Daniel Bemerguy 
# 08/04/2021 at 01:04
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import pickle

#Load train data
train = pd.read_csv("Dados/train.csv")
train = pd.DataFrame(train)
train = train.set_index('PassengerId')
train = train.drop(columns=['Ticket','Cabin'])
print(train.head())

#Load test data
test = pd.read_csv("Dados/test.csv")
test = pd.DataFrame(test)
test = test.set_index('PassengerId')
test = test.drop(columns=['Ticket','Cabin'])
print(test.head())

# Convert columns names to lowercase
train.columns = train.columns.str.lower()
test.columns = test.columns.str.lower()

#Check NaN
print(train.isna().any())
print(test.isna().any())


train["age"] = train["age"].fillna(train["age"].mean()) #mean
train["embarked"] = train["embarked"].fillna(train["embarked"].mode()[0]) #mode
test["age"] = test["age"].fillna(test["age"].mean()) #mean
test["embarked"] = test["embarked"].fillna(test["embarked"].mode()[0]) #mode
test["fare"] = test["fare"].fillna(test["fare"].mean()) #mean

#Generate dummies on train dataset
features = ['pclass', 'sex', 'sibsp', 'parch', 'embarked']
train[features] = train[features].astype('category')

def get_dummies(train,test,columns):
    df = pd.concat([train[columns], test[columns]])
    df = pd.get_dummies(df)
    X_train = df.iloc[:train.shape[0]]
    X_test = df.iloc[train.shape[0]:]
    return X_train, X_test

train['pclass'] = train['pclass'].astype('category')
test['pclass'] = test['pclass'].astype('category')

features = ['pclass', 'sex', 'sibsp', 'parch', 'embarked']
X_train, X_test = get_dummies(train, test, features)

y_train = train['survived']
print(type(y_train))

print("After generating dummies:",train)

#Feature Engineering
titles = train['name'].str.extract(' ([A-Za-z]+)\.', expand=False)
titles.unique() #Check unique titles

titles_map = {'Mr': 'Mr', 'Master': 'Master'}
titles_map.update(dict.fromkeys(['Mrs', 'Ms', 'Mme', 'Ms'], 'Mrs'))
titles_map.update(dict.fromkeys(['Miss', 'Mlle'], 'Miss'))
titles_map.update(dict.fromkeys(['Capt', 'Col', 'Major', 'Dr', 'Rev'], 'Officer'))
titles_map.update(dict.fromkeys(['Jonkheer', 'Don', 'Sir', 'Countess', 'Dona', 'Lady'], 'Royalty'))


def extract_title(df):
    df['title'] = df['name'].str.extract(' ([A-Za-z]+)\.', expand=False).map(titles_map)

extract_title(train)
extract_title(test)

features = ['pclass', 'sex', 'sibsp', 'parch', 'embarked', 'title']
X_train, X_test = get_dummies(train, test, features)

print(X_train)

#Save datasets to disk
X_test.to_csv("x_test.csv")
X_train.to_csv("x_train.csv")
y_train.to_csv("y_train.csv")

# #Visualize train
# # Check for correlations
# corr = train.corr(method = 'pearson')
# plt.figure(figsize=(12,10))
# sns.heatmap(corr, cmap="YlOrRd",vmin=-1., vmax=1., annot=True, fmt='.2f', cbar=True, linewidths=0.8)
# plt.title("Pearson correlation")
# plt.savefig("corr_matrix.png")
# plt.show()
#
#
# #Distributions
# fig, axarr = plt.subplots(nrows = 3, ncols = 3,figsize=(12,9))
# fig.subplots_adjust(hspace=0.5)
# fig.suptitle('Features\' Distributions')
#
#
# axarr[0,0].hist(train.iloc[:,0],density = 1, alpha = 0.65)
# axarr[0,0].title.set_text(train.columns[0])
#
# axarr[0,1].hist(train.iloc[:,1],density = 1, alpha = 0.65)
# axarr[0,1].title.set_text(train.columns[1])
#
# axarr[0,2].hist(train.iloc[:,2],density = 1, alpha = 0.65)
# axarr[0,2].title.set_text(train.columns[2])
#
# axarr[1,0].hist(train.iloc[:,3],density = 1, alpha = 0.65)
# axarr[1,0].title.set_text(train.columns[3])
#
# axarr[1,1].hist(train.iloc[:,4],density = 1, alpha = 0.65)
# axarr[1,1].title.set_text(train.columns[4])
#
# axarr[1,2].hist(train.iloc[:,5],density = 1, alpha = 0.65)
# axarr[1,2].title.set_text(train.columns[5])
#
# axarr[2,0].hist(train.iloc[:,6],density = 1, alpha = 0.65)
# axarr[2,0].title.set_text(train.columns[6])
#
# axarr[2,1].hist(train.iloc[:,7],density = 1, alpha = 0.65)
# axarr[2,1].title.set_text(train.columns[7])
# fig.tight_layout()
# plt.savefig('Grid_dist.png')
# plt.show()
#
# #Defining features and prediction variable

# train = train.drop(columns=['name','sex','sibsp','parch','embarked','title'])
# train["parch_9"] = np.zeros(len(train))
# X,Y = train.iloc[:,1:], train.iloc[:,0]
# print(X.head())
#
# x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size=0.7) #Use 100% as training data to upload results on kaggle

#Suport Vector Machine model
clf = SVC(kernel='linear')
clf.fit(X_train,y_train) #use x_train and y_train if not submitting the results

# y_pred = clf.predict(x_test)
# print("Printing accuracy:")
# print(accuracy_score(y_test,y_pred)) #Around 0.8 accuracy with 0.7 training data


# save the model to disk
filename = 'finalized_model_SVM.sav'
pickle.dump(clf, open(filename, 'wb'))


