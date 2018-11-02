# -*- coding: utf-8 -*-
"""
Created on Thu Nov  1 18:42:16 2018

@author: amine bahlouli
"""

from __future__ import print_function, division
from builtins import range
from keras.models import print_function,division
from keras.layers import Dense,Activation, Dropout,BatchNormalization
from keras.models import Sequential
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import keras.losses


def get_normalized_data():
    df = pd.read_csv("train.csv")
    data = df.values.astype(np.float32)
    np.random.shuffle(data)
    x = data[:,1:]
    y = data[:,0]
    
    xTrain = x[:-1000,]
    yTrain = y[:-1000,]
    xTest = x[-1000:]
    yTest = y[-1000:]
    
    mu = xTrain.mean(axis=0)
    std = xTrain.std(axis=0)
    np.place(std,std==0,1)
    xTrain = (xTrain-mu)/std
    xTest = (xTest-mu)/std
    return xTrain,yTrain,xTest,yTest
def y2indicator(y):
    N = len(y)
    y = y.astype(np.int32)
    ind = np.zeros((N, 10))
    for i in range(N):
        ind[i, y[i]] = 1
    return ind

X,Y,xTest,yTest = get_normalized_data()
K = 10
Y = y2indicator(Y)
yTest = y2indicator(yTest)
model = Sequential()
N,D = X.shape
model.add(Dense(units=500, input_dim=D,init='uniform'))
model.add(BatchNormalization())
model.add(Activation('tanh'))
model.add(Dropout(0.5))

model.add(Dense(units=250,init='uniform'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.3))

model.add(Dense(10, init='uniform' ))
model.add(BatchNormalization())
model.add(Activation("softmax"))
model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
r = model.fit(X,Y,batch_size=128,epochs=5,verbose=1)
score = model.evaluate(xTest,yTest, verbose=0)
print("loss is ",score[0])
print("accuracy is: ",score[1])