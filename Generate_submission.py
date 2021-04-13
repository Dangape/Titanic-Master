import pandas as pd
import pickle

#Load test data
test = pd.read_csv("x_test.csv")
test = pd.DataFrame(test)
test = test.set_index('PassengerId')

print(test.columns)
# load the model from disk
SVM = 'finalized_model_SVM.sav' #Support vector Machine Model
NBC = 'finalized_model_NBC.sav' #Naive Bayes Classifier Model
loaded_model = pickle.load(open(NBC, 'rb'))

## Predict
prediction = loaded_model.predict(test)
print("Printing prediction:")
print(prediction)


##Generate result file
submission = pd.DataFrame()
submission["PassengerId"] = list(range(892,1310))
submission["Survived"] = prediction
submission = submission.set_index("PassengerId")
print(submission)

submission.to_csv("submission.csv")