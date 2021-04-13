import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import pickle

#Load training data
x_train = pd.read_csv("x_train.csv")
x_train = pd.DataFrame(x_train)
x_train = x_train.set_index('PassengerId')
print(x_train.head())

y_train = pd.read_csv("y_train.csv")
y_train = pd.DataFrame(y_train)
y_train = y_train.set_index("PassengerId")

print(y_train)


#Load testing data
x_test = pd.read_csv("x_test.csv")
x_test = pd.DataFrame(x_test)
x_test = x_test.set_index('PassengerId')
print(x_test.head())


#Naive Bayes Classifier
model = GaussianNB()
model.fit(x_train,y_train.values.ravel())

#ypred = model.predict(x_x_test)
#print(accuracy_score(y_x_test,ypred))

# save the model to disk
filename = 'finalized_model_NBC.sav'
pickle.dump(model, open(filename, 'wb'))