import pandas as pd
import pickle

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

# load the model from disk
SVM = 'finalized_model_SVM.sav' #Support vector Machine Model
NBC = 'finalized_model_NBC.sav' #Naive Bayes Classifier Model
loaded_model = pickle.load(open(SVM, 'rb'))

## Predict
prediction = loaded_model.predict(data_test)
print("Printing prediction:")
print(prediction)


##Generate result file
submission = pd.DataFrame()
submission["PassengerId"] = list(range(892,1310))
submission["Survived"] = prediction
submission = submission.set_index("PassengerId")
print(submission)

submission.to_csv("submission.csv")