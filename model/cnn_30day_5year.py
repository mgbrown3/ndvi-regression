#!/usr/bin/python

# cnn regression
# based on CNN regression example from https://www.datatechnotes.com/2019/12/how-to-fit-regression-data-with-cnn.html

# Initialize
from sklearn.datasets import load_boston
from keras.models import Sequential
from keras.layers import Dense, Conv1D, Flatten
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import csv
import os, sys
import numpy as np


### Prepare the data
##boston = load_boston()
##x, y = boston.data, boston.target
##print(x.shape) 
##
### Reshape data
##x = x.reshape(x.shape[0], x.shape[1], 1)
##print(x.shape)

# Read in 2006NDVI_LCin_noZeros.csv
with open('/Volumes/haraguchi/2006NDVI_LCin_noZeros.csv', newline='') as f:
    reader = csv.reader(f)
    data = []
    for row in reader:
#        print(row)
        data.append(row)

data_all = np.array(data[1:-1])    # make the list an array, exlude headers
makeFloat = np.vectorize(np.float) # make the data type float
data_all = makeFloat(data_all)


# remove NAN
data_clean=data_all[~np.isnan(data_all).any(axis=1)]

# get inputs and outputs for model
y=data_clean[:,1]
x=data_clean[:,2:9]
print(x.shape)
print(y.shape)


# Reshape
x = x.reshape(x.shape[0], x.shape[1], 1)
print(x.shape)


# Split training and testing data sets
xtrain, xtest, ytrain, ytest=train_test_split(x, y, test_size=0.15) 


# Define and fit the model
model = Sequential()
model.add(Conv1D(32, 2, activation="relu", input_shape=(7, 1)))
model.add(Flatten())
model.add(Dense(64, activation="relu"))
model.add(Dense(1))
model.compile(loss="mse", optimizer="adam")
 
model.summary()

# Fit model
model.fit(xtrain, ytrain, batch_size=12,epochs=200, verbose=0)


# Predit and visualize
ypred = model.predict(xtest)


# Evaluate model
print(model.evaluate(xtrain, ytrain))
 
print("MSE: %.4f" % mean_squared_error(ytest, ypred))

x_ax = range(len(ypred))
plt.scatter(x_ax, ytest, s=5, color="blue", label="original")
plt.plot(x_ax, ypred, lw=0.8, color="red", label="predicted")
plt.legend()
plt.show()


