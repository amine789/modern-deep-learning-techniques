# -*- coding: utf-8 -*-
"""
Created on Sun Oct  7 02:15:47 2018

@author: amine bahlouli
"""
import numpy as np
import pandas as pd
def y2indicator(y):
    N = len(y)
    y = y.astype(np.int32)
    #K = len(set(y))
    print(y.shape)
    ind=np.zeros((N,10))
    for i in range(N):
        ind[i, y[i]]=1
    return ind

def cost(p_y,t):
    tot = t*np.log(p_y)
    return -tot.sum()

def predict(p_y):
    return np.argmax(p_y,axis=1)
def forward(X, W1, b1, W2, b2):
    # sigmoid
    # Z = 1 / (1 + np.exp(-( X.dot(W1) + b1 )))

    # relu
    Z = X.dot(W1) + b1
    Z[Z < 0] = 0

    A = Z.dot(W2) + b2
    expA = np.exp(A)
    Y = expA / expA.sum(axis=1, keepdims=True)
    return Y, Z

def derivative_w2(Z, T, Y):
    return Z.T.dot(Y - T)

def derivative_b2(T, Y):
    return (Y - T).sum(axis=0)

def derivative_w1(X, Z, T, Y, W2):
    # return X.T.dot( ( ( Y-T ).dot(W2.T) * ( Z*(1 - Z) ) ) ) # for sigmoid
    return X.T.dot( ( ( Y-T ).dot(W2.T) * (Z > 0) ) ) # for relu

def derivative_b1(Z, T, Y, W2):
    # return (( Y-T ).dot(W2.T) * ( Z*(1 - Z) )).sum(axis=0) # for sigmoid
    return (( Y-T ).dot(W2.T) * (Z > 0)).sum(axis=0) # for relu



def error_rate(p_y,t):
    prediction = predict(p_y)
    return np.mean(prediction!=t)

def get_normalized_data():
    
    print("reading in transforming data")
    df = pd.read_csv("train.csv")
    data = df.as_matrix().astype(np.float32)
    np.random.shuffle(data)
    
    X= data[:, 1:]
    Y = data[:, 0]
    Xtrain = X[:-1000,]
    Ytrain = Y[:-1000,]
    Xtest  = X[-1000:,]
    Ytest  = Y[-1000:,]

    # normalize the data
    mu = Xtrain.mean(axis=0)
    std = Xtrain.std(axis=0)
    np.place(std, std == 0, 1)
    Xtrain = (Xtrain - mu) / std
    Xtest = (Xtest - mu) / std
    
    return Xtrain, Ytrain,  Xtest, Ytest


def main():
    max_iter = 10
    print_period = 10

    xTrain, yTrain, xTest, yTest = get_normalized_data()
    reg = 0.01

    yTrain_ind = y2indicator(yTrain)
    yTest_ind = y2indicator(yTest)

    N, D = xTrain.shape
    batch_sz = 500
    batch_n = N // batch_sz
    M = 300
    K = 10
    w1_0 = np.random.randn(D, M) / np.sqrt(D)
    b1_0 = np.zeros(M)
    w2_0 = np.random.randn(M, K) / np.sqrt(M)
    b2_0 = np.zeros(K)
    
    w1 = w1_0.copy()
    b1 = b1_0.copy()
    w2 = w2_0.copy()
    b2 = b2_0.copy()
    
    # 1st momentum
    mw1=0
    mb1=0
    mb2=0
    mw2=0
    
    # 2nd momentum
    
    vw1=0
    vw2=0
    vb1=0
    vb2=0
    
    # hyperparams
    lr0 = 0.001
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
    
    # Adam
    loss_adam =  []
    err_adam = []
    t=1
    """
    for i in range(max_iter):
        for j in range(batch_n):
            xBatch = xTrain[j*batch_sz:(j*batch_sz + batch_sz),]
            yBatch = yTrain_ind[j*batch_sz:(j*batch_sz + batch_sz),]
            p_y, z = forward(xBatch, w1, b1, w2, b2)
            
            gw2 = derivative_w2(z,yBatch,p_y) + reg*w2
            gb2 = derivative_b2(yBatch,p_y) + reg*b2
            gw1 = derivative_w1(xBatch,z, yBatch,p_y,w2) + reg*w1
            gb1 = derivative_b1(z,yBatch,p_y,w2) + reg*b1
            
            #update momentum term
            mw1= beta1*mw1 + (1-beta1)*gw1
            mw2= beta1*mw2 + (1-beta1)*gw2
            mb2 = beta1*mb2 + (1-beta1)*gb2
            mb1 = beta1*mb1 + (1-beta1)*gb1
            
            #update RMprop term
            
            vw1 = beta2*vw1 + (1-beta2)*gw1*gw1
            vw2 = beta2*vw2 + (1-beta2)*gw2*gw2
            vb2 = beta2*vb2 + (1-beta2)*gb2*gb2
            vb1 = beta2*vb1 + (1-beta2)*gb1*gb1
            
            #bias correction
            correction1 = 1-beta1**t
            mw1_hat = mw1/correction1
            mw2_hat = mw2/correction1
            mb2_hat = mb2/correction1
            mb1_hat = mb1/correction1
            
            correction2 = 1-beta2**t
            vw1_hat = vw1/correction2
            vw2_hat = vw2/correction2
            vb2_hat = vb2/correction2
            vb1_hat = vb1/correction2
            t+=1
            
            w1 -= lr0*mw1_hat/(np.sqrt(vw1_hat+eps))
            w2-= lr0*mw2_hat/(np.sqrt(vw2_hat+eps))
            b1-= lr0*mb1_hat/(np.sqrt(vb1_hat+eps))
            b2 -= lr0*mb2_hat/(np.sqrt(vb2_hat+eps))
            
            if j % print_period == 0:
                p_y, _ = forward(xTest, w1, b1, w2, b2)
                l = cost(p_y, yTest_ind)
                loss_adam.append(l)
                print("Cost at iteration i=%d, j=%d: %.6f" % (i, j, l))

                e = error_rate(p_y, yTest)
                err_adam.append(e)
                print("Error rate:", e)

    p_y, _ = forward(xTest, w1, b1, w2, b2)
    print("Final error rate:", error_rate(p_y, yTest))
    """
    w1 = w1_0.copy()
    b1 = b1_0.copy()
    w2 = w2_0.copy()
    b2 = b2_0.copy()
    loss_rms = []
    err_rms = []

    # comparable hyperparameters for fair comparison
    lr0 = 0.001
    mu = 0.9
    decay_rate = 0.999
    eps = 1e-8

    # rmsprop cache
    w2_cach = 1
    b2_cach = 1
    w1_cach = 1
    b1_cach = 1

    # momentum
    dw1 = 0
    db1 = 0
    dw2 = 0
    db2 = 0
    for i in range(max_iter):
        for j in range(batch_n):
            xBatch = xTrain[j*batch_sz:(j*batch_sz + batch_sz),]
            yBatch = yTrain_ind[j*batch_sz:(j*batch_sz + batch_sz),]
            p_y, z = forward(xBatch, w1, b1, w2, b2)
            
            gw2 = derivative_w2(z,yBatch,p_y) + reg*w2
            w2_cach = w2_cach*decay_rate + (1-decay_rate)*gw2*gw2
            dw2 = mu*dw2 + lr0*(1-mu)*gw2/(np.sqrt(w2_cach+eps))
            w2 -= dw2
            
            gb2 = derivative_b2(yBatch,p_y) + reg*b2
            b2_cach = b2_cach*decay_rate + (1-decay_rate)*gb2*gb2
            db2 = mu*db2 + lr0*(1-mu)*gb2/(np.sqrt(b2_cach+eps))
            b2 -= db2
            
            gw1 = derivative_w1(xBatch,z, yBatch,p_y,w2) + reg*w1
            w1_cach = w1_cach*decay_rate + (1-decay_rate)*gw1*gw1
            dw1 = mu*dw1 + lr0*(1-mu)*gw1/(np.sqrt(w1_cach+eps))
            w1 -= dw1
            
            gb1 = derivative_b1(z,yBatch,p_y,w2) + reg*b1
            b1_cach = b1_cach*decay_rate + (1-decay_rate)*gb1*gb1
            db1 = db1*mu + lr0*(1-mu)* gb1/(np.sqrt(b1_cach+eps))
            b1 -= db1
            
            if j % print_period == 0:
                p_y, _ = forward(xTest, w1, b1, w2, b2)
                l = cost(p_y, yTest_ind)
                loss_rms.append(l)
                print("Cost at iteration i=%d, j=%d: %.6f" % (i, j, l))

                err = error_rate(p_y, yTest)
                err_rms.append(err)
                print("Error rate:", err)

    p_y, _ = forward(xTest, w1, b1, w2, b2)
    print("Final error rate:", error_rate(p_y, yTest))
    plt.plot(loss_adam, label='adam')
    plt.plot(loss_rms, label='rmsprop')
    plt.legend()
    plt.show()
            
            
            
    
            
            
            
            
    


