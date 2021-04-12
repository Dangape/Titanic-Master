# Created by: Daniel Bemerguy 
# 08/04/2021 at 01:04
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

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

#Visualize data
# Check for correlations
corr = data.corr(method = 'pearson')
plt.figure(figsize=(12,10))
sns.heatmap(corr, cmap="YlOrRd",vmin=-1., vmax=1., annot=True, fmt='.2f', cbar=True, linewidths=0.8)
plt.title("Pearson correlation")
plt.savefig("corr_matrix.png")
plt.show()

#Fare mean
print('Survived: ',survived['Fare'].mean())
print("Dead: ",dead['Fare'].mean())

#Distributions
fig, axarr = plt.subplots(nrows = 3, ncols = 3,figsize=(12,9))
fig.subplots_adjust(hspace=0.5)
fig.suptitle('Features\' Distributions')


axarr[0,0].hist(data.iloc[:,0],density = 1, alpha = 0.65)
axarr[0,0].title.set_text(data.columns[0])

axarr[0,1].hist(data.iloc[:,1],density = 1, alpha = 0.65)
axarr[0,1].title.set_text(data.columns[1])

axarr[0,2].hist(data.iloc[:,2],density = 1, alpha = 0.65)
axarr[0,2].title.set_text(data.columns[2])

axarr[1,0].hist(data.iloc[:,3],density = 1, alpha = 0.65)
axarr[1,0].title.set_text(data.columns[3])

axarr[1,1].hist(data.iloc[:,4],density = 1, alpha = 0.65)
axarr[1,1].title.set_text(data.columns[4])

axarr[1,2].hist(data.iloc[:,5],density = 1, alpha = 0.65)
axarr[1,2].title.set_text(data.columns[5])

axarr[2,0].hist(data.iloc[:,6],density = 1, alpha = 0.65)
axarr[2,0].title.set_text(data.columns[6])

axarr[2,1].hist(data.iloc[:,7],density = 1, alpha = 0.65)
axarr[2,1].title.set_text(data.columns[7])
fig.tight_layout()
plt.savefig('Grid_dist.png')
plt.show()

#Defining features and prediction variable

X,Y = data.iloc[:,1:], data.iloc[:,0]
print(X.head())

x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size=0.7) #Use 100% as training data to upload results on kaggle

#Suport Vector Machine model
clf = SVC(kernel='linear')
clf.fit(X,Y) #use x_train and y_train if not submitting the results
#y_pred = clf.predict(x_test)
#print(accuracy_score(y_test,y_pred)) #Around 0.8 accuracy with 0.7 training data

##Load test dataset
data_test = pd.DataFrame(pd.read_csv("Dados/test.csv"))

data_test = data_test.set_index('PassengerId')
data_test = data_test.drop(columns=['Name','Ticket','Cabin'])


#Recode data_test
data_test['Embarked'] = data_test['Embarked'].replace(['S'],int(0))
data_test['Embarked'] = data_test['Embarked'].replace(['C'],int(1))
data_test['Embarked'] = data_test['Embarked'].replace(['Q'],int(2))
data_test['Sex'] = data_test['Sex'].replace(['male'],int(1))
data_test['Sex'] = data_test['Sex'].replace(['female'],int(0))
print("Printing data test:")
print(data_test)

#Replace NaN values
print("There are", len(data_test), "rows on test data, and",data_test.isnull().values.ravel().sum(),"rows have missing values")

print(data_test.isna().any()) #Check columns with NaN values

data_test["Age"] = data_test["Age"].fillna(data_test["Age"].mean())
data_test["Fare"] = data_test["Fare"].fillna(data_test["Fare"].mean())

## Predict
prediction = clf.predict(data_test)
print("Printing prediction:")
print(prediction)

##Generate result file
submission = pd.DataFrame()
submission["PassengerId"] = list(range(892,1310))
submission["Survived"] = prediction
submission = submission.set_index("PassengerId")
print(submission)

submission.to_csv("submission.csv")
