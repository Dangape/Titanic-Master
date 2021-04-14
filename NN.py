import pandas as pd
import numpy as np
import tensorflow as tf
import keras
from keras.layers import Dense, Dropout, Input
from keras.models import Sequential
from sklearn.model_selection import GridSearchCV
from keras.layers import Dropout
from keras.wrappers.scikit_learn import KerasRegressor


#Load training data
x_train = pd.read_csv("x_train.csv")
x_train = pd.DataFrame(x_train)
x_train = x_train.set_index('PassengerId')
print(x_train.head())
print(x_train.shape)

y_train = pd.read_csv("y_train.csv")
y_train = pd.DataFrame(y_train)
y_train = y_train.set_index("PassengerId")
print(y_train)

#Load testing data
x_test = pd.read_csv("x_test.csv")
x_test = pd.DataFrame(x_test)
x_test = x_test.set_index('PassengerId')
print(x_test.head())

model = Sequential()
model.add(Dense(16, input_dim=16, activation='relu',kernel_initializer='he_normal'))
model.add(Dense(32, activation='relu',kernel_initializer='he_normal'))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu', kernel_initializer='he_normal'))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu',kernel_initializer='he_normal'))
model.add(Dense(1, activation='sigmoid'))
# compile the keras model
model.compile(loss = tf.keras.losses.binary_crossentropy, optimizer = tf.keras.optimizers.Adam(), metrics = ['acc'])
model.fit(x_train, y_train, batch_size = 128, epochs = 50)

####################################
#Hyper-parameter tuning
#Deep learning
#define the keras model
# def create_model(neurons,dropout_rate):
#     model = Sequential()
#     model.add(Dense(16, input_dim=16, activation='relu',kernel_initializer='he_normal'))
#     model.add(Dense(neurons, activation='relu',kernel_initializer='he_normal'))
#     model.add(Dropout(dropout_rate))
#     model.add(Dense(neurons, activation='relu', kernel_initializer='he_normal'))
#     model.add(Dropout(dropout_rate))
#     model.add(Dense(neurons, activation='relu',kernel_initializer='he_normal'))
#     model.add(Dense(1, activation='sigmoid'))
#     # compile the keras model
#     model.compile(loss = tf.keras.losses.binary_crossentropy, optimizer = tf.keras.optimizers.Adam(), metrics = ['acc'])
#     return model
# model = KerasRegressor(build_fn=create_model, verbose=True)
#
# batch_size = [10, 20,32,128]
# epochs = [50, 100, 200]
# neurons = [18,32,64,128]
# dropout_rate = [0.0, 0.1, 0.2, 0.3]
# #activation = ['relu', 'tanh', 'linear']
# #init_mode = ['uniform', 'normal']
# param_grid = dict(batch_size=batch_size, epochs=epochs, neurons = neurons, dropout_rate = dropout_rate)
# grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3)
# grid_result = grid.fit(x_train, y_train)
#
# print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
# means = grid_result.cv_results_['mean_test_score']
# stds = grid_result.cv_results_['std_test_score']
# params = grid_result.cv_results_['params']
# for mean, stdev, param in zip(means, stds, params):
#     print("%f (%f) with: %r" % (mean, stdev, param))
############################################

# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")
