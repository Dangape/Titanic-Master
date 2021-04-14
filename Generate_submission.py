import pandas as pd
import pickle
from keras.models import model_from_json

#Load test data
test = pd.read_csv("x_test.csv")
test = pd.DataFrame(test)
test = test.set_index('PassengerId')

print(test.columns)
# load the model from disk
SVM = 'finalized_model_SVM.sav' #Support vector Machine Model
NBC = 'finalized_model_NBC.sav' #Naive Bayes Classifier Model
MLP = 'finalized_model_MLP.sav' #Multi-layer Perceptron classifier
#loaded_model = pickle.load(open(MLP, 'rb'))

# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")

## Predict
prediction = loaded_model.predict(test)
prediction = (prediction > 0.5).astype(int).ravel()
print("Printing prediction:")
print(prediction)


##Generate result file
submission = pd.DataFrame()
submission["PassengerId"] = list(range(892,1310))
submission["Survived"] = prediction
submission = submission.set_index("PassengerId")
print(submission)

submission.to_csv("submission.csv")