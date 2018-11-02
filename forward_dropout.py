# -*- coding: utf-8 -*-
"""
Created on Fri Nov  2 16:36:58 2018

@author: amine bahlouli
"""
import numpy as np
X = np.random.rand(50,30)
W = np.random.rand(30,20)
keep = 0.5
D1 = X.dot(W)
A1 = np.tanh(D1)
D1 = A1<keep
A1_dropout = np.multiply(A1,D1)
W2 = np.random.rand(20,10)
D2 = A1_dropout.dot(W2)

expA = np.exp(D2)
expA = expA/(expA.sum(1,keepdims=True))

results = np.argmax(expA,1)
