# -*- coding: utf-8 -*-
"""
Created on Thu Oct 18 00:35:21 2018

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


def get_normalized_data():
    df = pd.read_csv("train.csv")
    data = df.values.astype(np.float32)
    np.random.shuffle(data)
    X = data[:,1:]
    Y = data[:,0]
    
   
    
    mu = X.mean(axis=0)
    std = X.std(axis=0)
    np.place(std,std==0,1)
    X = (X-mu)/std
    
    return X,Y

def y2indicator(y):
    N = len(y)
    y = y.astype(np.int32)
    ind = np.zeros((N, 10))
    for i in range(N):
        ind[i, y[i]] = 1
    return ind

X,Y = get_normalized_data()
K = 10
Y = y2indicator(Y)
model = Sequential()
N,D = X.shape
model.add(Dense(units=500, input_dim=D))
model.add(Activation("relu"))
model.add(Dense(units=300))
model.add(Activation("relu"))
model.add(Dense(units=K))
model.add(Activation("softmax"))
model.summary()
model.compile(loss='categorical_crossentropy',
              optimizer = 'adam',
              metrics = ['accuracy'])
r = model.fit(X,Y,validation_split=0.33,epochs=5, batch_size=32)
print("returned: ",r)
