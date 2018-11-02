# -*- coding: utf-8 -*-
"""
Created on Thu Nov  1 03:24:00 2018

@author: amine bahlouli
"""

from __future__ import print_function, division
from builtins import range
from keras.models import print_function,division
from keras.layers import Dense,Activation
from keras.models import Sequential
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import keras.losses
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

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
X,Y,xTrain,yTrain = get_normalized_data()
Y = y2indicator(Y)
yTrain = y2indicator(yTrain)
model = Sequential()
N,D = X.shape
model.add(Dense(units=500,input_dim=D))
model.add(Activation("relu"))
model.add(Dense(units=250))
model.add(Activation("relu"))
model.add(Dense(units=10))
model.add(Activation("softmax"))

def fit_model(m,xtrain, ytrain,xtest,ytest,epochs=100):
    checkpoint = ModelCheckpoint(monitor='val_acc', filepath='Documents', save_best_only=True)
    earlyStopping = EarlyStopping(monitor='val_acc',min_delta=0.01,patience=10,verbose=1,mode='max')
    reduce_lr = ReduceLROnPlateau(monitor="val_acc",patience=10,min_lr=10,mode="max")
    return m.fit(xtrain,ytrain,batch_size=1024,epochs=10,verbose=1
                 ,validation_data=[xtest,ytest],callbacks=[checkpoint, earlyStopping, reduce_lr])
    
model.compile(loss='categorical_crossentropy',
              optimizer = 'adam',
              metrics = ['accuracy'])

fit_model(model,X,Y,xTrain,yTrain)