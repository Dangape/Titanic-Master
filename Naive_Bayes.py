import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import pickle

#Load data
data = pd.read_csv("Dados/train.csv")
data = pd.DataFrame(data)
data = data.set_index('PassengerId')
data = data.drop(columns=['Name','Ticket','Cabin'])
print(data.head())

#Subset
survived = data[data['Survived']==1]
dead = data[data['Survived']==0]

#Recode data
data['Embarked'] = data['Embarked'].replace(['S'],int(0))
data['Embarked'] = data['Embarked'].replace(['C'],int(1))
data['Embarked'] = data['Embarked'].replace(['Q'],int(2))
data['Sex'] = data['Sex'].replace(['male'],int(1))
data['Sex'] = data['Sex'].replace(['female'],int(0))
print(data)

#Replace NaN values with mean (not best solution)
print("There are", len(data), "rows on training data, and",data.isnull().values.ravel().sum(),"rows have missing values")

print(data.isna().any())

data["Age"] = data["Age"].fillna(data["Age"].mean())
data["Embarked"] = data["Embarked"].fillna(0)
print("Mean:",data["Age"].mean())

#Defining features and prediction variable
X,Y = data.iloc[:,1:], data.iloc[:,0]
print(X.head())

#x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size=0.7) #Use 100% as training data to upload results on kaggle

#Naive Bayes Classifier
model = GaussianNB()
model.fit(X,Y)

#ypred = model.predict(x_test)
#print(accuracy_score(y_test,ypred))

# save the model to disk
filename = 'finalized_model_NBC.sav'
pickle.dump(model, open(filename, 'wb'))