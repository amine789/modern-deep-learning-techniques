# -*- coding: utf-8 -*-
"""
Created on Mon Oct  1 00:31:32 2018

@author: amine bahlouli
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA



def get_transformed_data():
    
    print("reading in transforming data")
    df = pd.read_csv("train.csv")
    data = df.as_matrix().astype(np.float32)
    np.random.shuffle(data)
    
    X= data[:, 1:]
    mu = X.mean(axis=0)
    X = X - mu
    pca = PCA()
    Z = pca.fit_transform(X)
    Y = data[:, 0]
    return Z, Y, pca, mu
def y2indicator(y):
    N = len(y)
    K = len(set(y))
    ind=np.zeros((N,K))
    for i in range(N):
        ind[i, y[i]]=1
    return ind
def cost(p_y,t):
    tot =t*np.log(p_y)
    return -tot.sum()

def forward(X,w,b):
    a = X.dot(w) +b
    expA = np.exp(a)
    y = expA/expA.sum(axis=1, keepdims=True)
    return y

def error_rate(p_y, t):
    prediction = predict(p_y)
    return np.mean(prediction!=t)
def gradW(t,y,X):
    return X.T.dot(y-t)
def gradb(t,y):
    return (t-y).sum(axis=0)
def predict(p_y):
    return np.argmax(p_y, axis=1)
def main():
    x,y,_,_ = get_transformed_data()
    Xtrain = x[:-1000,]
    Ytrain = y[:-1000]
    Xtest = x[-1000:,]
    Ytest = y[-1000:]
    
    

    N, D = Xtrain.shape
    w = np.random.randn(D,10)/28
    b = np.random.randn(10)
    Ytrain_ind = y2indicator(Ytrain)
    Ytest_ind = y2indicator(Ytest)
    

    # 1. full
    W = np.random.randn(D, 10) / 28
    b = np.zeros(10)
    LL = []
    lr = 0.0001
    reg = 0.01
    t0 = datetime.now()
    for i in range(50):
        p_y = forward(Xtrain, W, b)

        W += lr*(gradW(Ytrain_ind, p_y, Xtrain) - reg*W)
        b += lr*(gradb(Ytrain_ind, p_y) - reg*b)
        

        p_y_test = forward(Xtest, W, b)
        ll = cost(p_y_test, Ytest_ind)
        LL.append(ll)
        if i % 1 == 0:
            err = error_rate(p_y_test, Ytest)
            if i % 10 == 0:
                print("Cost at iteration %d: %.6f" % (i, ll))
                print("Error rate:", err)
    p_y = forward(Xtest, W, b)
    print("Final error rate:", error_rate(p_y, Ytest))
    print("Elapsted time for full GD:", datetime.now() - t0)


    # 2. stochastic
    W = np.random.randn(D, 10) / 28
    b = np.zeros(10)
    LL_stochastic = []
    lr = 0.0001
    reg = 0.01

    t0 = datetime.now()
    for i in range(1): # takes very long since we're computing cost for 41k samples
        tmpX, tmpY = shuffle(Xtrain, Ytrain_ind)
        for n in range(min(N, 500)): # shortcut so it won't take so long...
            x = tmpX[n,:].reshape(1,D)
            y = tmpY[n,:].reshape(1,10)
            p_y = forward(x, W, b)

            W += lr*(gradW(y, p_y, x) - reg*W)
            b += lr*(gradb(y, p_y) - reg*b)

            p_y_test = forward(Xtest, W, b)
            ll = cost(p_y_test, Ytest_ind)
            LL_stochastic.append(ll)

        if i % 1 == 0:
            err = error_rate(p_y_test, Ytest)
            if i % 10 == 0:
                print("Cost at iteration %d: %.6f" % (i, ll))
                print("Error rate:", err)
    p_y = forward(Xtest, W, b)
    print("Final error rate:", error_rate(p_y, Ytest))
    print("Elapsted time for SGD:", datetime.now() - t0)


    # 3. batch
    W = np.random.randn(D, 10) / 28
    b = np.zeros(10)
    LL_batch = []
    lr = 0.0001
    reg = 0.01
    batch_sz = 500
    n_batches = N // batch_sz

    t0 = datetime.now()
    for i in range(50):
        tmpX, tmpY = shuffle(Xtrain, Ytrain_ind)
        for j in range(n_batches):
            x = tmpX[j*batch_sz:(j*batch_sz + batch_sz),:]
            y = tmpY[j*batch_sz:(j*batch_sz + batch_sz),:]
            p_y = forward(x, W, b)

            W += lr*(gradW(y, p_y, x) - reg*W)
            b += lr*(gradb(y, p_y) - reg*b)

            p_y_test = forward(Xtest, W, b)
            ll = cost(p_y_test, Ytest_ind)
            LL_batch.append(ll)
        if i % 1 == 0:
            err = error_rate(p_y_test, Ytest)
            if i % 10 == 0:
                print("Cost at iteration %d: %.6f" % (i, ll))
                print("Error rate:", err)
    p_y = forward(Xtest, W, b)
    print("Final error rate:", error_rate(p_y, Ytest))
    print("Elapsted time for batch GD:", datetime.now() - t0)



    x1 = np.linspace(0, 1, len(LL))
    plt.plot(x1, LL, label="full")
    x2 = np.linspace(0, 1, len(LL_stochastic))
    plt.plot(x2, LL_stochastic, label="stochastic")
    x3 = np.linspace(0, 1, len(LL_batch))
    plt.plot(x3, LL_batch, label="batch")
    plt.legend()
    plt.show()
                
                
            
    
        
        