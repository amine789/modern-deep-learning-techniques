# -*- coding: utf-8 -*-
"""
Created on Thu Oct 11 00:09:40 2018

@author: amine bahlouli
"""
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def error_rate(p,t):
    return np.mean(p!=t)
def y2indicator(y):
    y = y.astype(np.int32)
    N = len(y)
    k=10
    ind = np.zeros((N,10))
    for i in range(N):
        ind[i, y[i]]=1
    return ind

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

def main():
    xTrain,yTrain,xTest,yTest = get_normalized_data()
    max_iter= 20
    print_period=10
    lr = 0.00004
    reg = 0.01
    yTrain_ind = y2indicator(yTrain)
    yTest_ind = y2indicator(yTest)
    N,D = xTrain.shape
    batch_n = N//500
    batch_sz=500
    M1=300
    M2=100
    k=10
    w1_init = np.random.randn(D,M1)/np.sqrt(D)
    b1_init = np.random.randn(M1)
    w2_init = np.random.randn(M1,M2)/np.sqrt(M1)
    b2_init = np.random.randn(M2)
    w3_init = np.random.randn(M2,k)/np.sqrt(k)
    b3_init = np.random.randn(k)
    X = tf.placeholder(tf.float32, shape=(None,D), name='X')
    T = tf.placeholder(tf.float32, shape=(None,k), name='T')
    w1 = tf.Variable(w1_init.astype(np.float32))
    b1 = tf.Variable(b1_init.astype(np.float32))
    w2 = tf.Variable(w2_init.astype(np.float32))
    b2 = tf.Variable(b2_init.astype(np.float32))
    w3 = tf.Variable(w3_init.astype(np.float32))
    b3 = tf.Variable(b3_init.astype(np.float32))
    
    z1 = tf.nn.relu(tf.matmul(X,w1)+b1)
    z2 = tf.nn.relu(tf.matmul(z1,w2)+b2)
    Yish = tf.matmul(z2,w3)+b3
    
    cost = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=Yish, labels=T))
    train_op = tf.train.RMSPropOptimizer(lr, decay=0.99, momentum=0.9).minimize(cost)
    predict_op = tf.argmax(Yish,1)
    LL = []
    init = tf.global_variables_initializer()
    with tf.Session() as session:
        session.run(init)
        for i in range(max_iter):
            for j in range(batch_n):
                xBatch = xTrain[j*batch_sz:(batch_sz+j*batch_sz),]
                yBatch = yTrain_ind[j*batch_sz:(batch_sz+j*batch_sz),]
                session.run(train_op, feed_dict={X:xBatch, T:yBatch})
                if j%print_period==0:
                    test_cost = session.run(cost, feed_dict={X:xTest, T:yTest_ind})
                    prediction = session.run(predict_op, feed_dict={X:xTest})
                    err = error_rate(prediction,yTest)
                    print("cost/err at i=%d j=%d: %.3f , %.3f"%(i,j,test_cost,err ))
                    LL.append(test_cost)
    plt.plot(LL)
    plt.show()
    
    