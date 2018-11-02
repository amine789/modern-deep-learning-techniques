# -*- coding: utf-8 -*-
"""
Created on Thu Nov  1 16:17:28 2018

@author: amine bahlouli
"""
import numpy as np
import pandas as pd
from keras.layers import Dense, Dropout
from keras.models import Sequential
batch_size = 128
DROPOUT= True
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
Y = y2indicator(Y)
yTest = y2indicator(yTest)
model = Sequential()
N,D = X.shape
model.add(Dense(512, activation='relu', input_dim=D))
if DROPOUT:
    model.add(Dropout(0.5))
model.add(Dense(512, activation='relu'))
if DROPOUT:
    model.add(Dropout(0.5))
model.add(Dense(512, activation='relu'))
if DROPOUT:
    model.add(Dropout(0.5))
model.add(Dense(512, activation='relu'))
if DROPOUT:
    model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

model.summary()

model.compile(loss="categorical_crossentropy",
              optimizer="adam",
              metrics= ["accuracy"])
r = model.fit(X,Y,batch_size=128,epochs=5, verbose=1)
score = model.evaluate(xTest,yTest, verbose=0)
print("loss: ", score[0])
print("loss: ", score[1])



