# Created by: Daniel Bemerguy 
# 08/04/2021 at 01:04
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import seaborn as sns

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
print(data)

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

# print(data.Pclass.unique())