# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 17:36:35 2018

@author: amine bahlouli
"""

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
def init_weight(M1,M2):
    return np.random.randn(M1,M2) * np.sqrt(2.0/M1)

class HiddenLayerBatchNorm:
    def __init__(self,M1,M2,f):
        self.M1=M1
        self.M2=M2
        self.f=f
        w = init_weight(M1,M2)
        gamaa = np.ones(M2).astype(np.float32)
        beta = np.zeros(M2).astype(np.float32)
        self.gamma = tf.Variable(gamma)
        self.beta = tf.Variable(beta)
        self.running_mean = tf.Variable(np.zeros(M2).astype(np.float32), trainable=False)
        self.running_var = tf.Variable(np.zeros(M2).astype(np.float32), trainable=False)
    def forward(self,X,,is_trainning,decay=0.9):
        if is_training:
            batch_mean,batch_var = tf.nn.moments(activation, [0])
            update_running_mean = tf.assign(
                    self.running.mean,
                    self.running_mean*decay+(1-decay)*batch_decay)
            update_running_var = tf.assign(self.running_var,
                                           self.running_var*decay+batch_var*(1-decay))
            with tf.control_depencies([update_running_mean;ypdate_running_var]):
                out = tf.nn.batch_normalization(
                        activation,
                        batch_mean,
                        batch_var,
                        self.beta,
                        self.gamma,
                        1e-4)
        else:
            out = tf.nn.batch_normalization(
                    activation,
                    self.running_mean,
                    self.running_var,
                    self.beta,
                    self.gamma,
                    1e-4)
        return self.f(out)
class HiddenLayer(object):
    def __init__(self,M1,M2,f):
        self.M1=M2
        self.M2=M2
        self.f=f
    def forward(self,x):
        return self.f(tf.matmul(X,self.W)+ self.b)
class ANN:
    def __init__(self, hidden_layer_sizes):
        self.hidden_layer_sizes=hidden_layer_sizes
    def set_session(self, session):
        self.Session=session
    def fit(self,X,Y,xTest,yTest,activation=tf.nn.relu,learning_rate=1e-2,epochs=15, batch_sz=100, print_period=100, show_fig=True):
        X = X.astype(np.float32)
        Y = Y.astype(np.float32)
        N,D = X.shape
        self.layers = []
        M1=D
        for M2 in self.hidden_layer_sizes:
            h = HiddenlayerBatchNorm(M1,M2,activation)
            self.layers.append(h)
        k = len(set(Y))
        h = HiddenLayer(M1,,K,lambda x:x)
        self.layers.append(h)
        if batch_sz is None:
            batch_sz=N
        
        tfX = tf.placeholder(tf.float32,shape=(None,D), name="X")
        tfY = tf.placerholder(tf.int32, shape=(None,), name="Y")
        self.tfX=tfX
        logits = self.forward(tfX, is_training=True)
        cost = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                        logits=logits,
                        labels=tfY))
        train_op = tf.MomentumOptimizer(learning_rate, momentum=0.9,use_nesterov=True).minimize(cost)
        test_logits = self.forward(tfX, is_training=False)
        self.predict_op = tf.argmax(test_logits,1)
        
        self.session.run(self.global_initializer())
        n_batches = N//batch_sz
        costs = []
        for i in range(epochs):
            if n_batches>1:
                X,Y= shuffle(X,Y)
            for j in range(n_batches):
                xBatch = X[j*batch_sz:(j*batch_sz+batch_sz)]
                yBatch = Y[j*batch_sz:(j*batch_sz+batch_sz)]
                c,_,lgts = self.session.run([cost,train_op,logits], feed_dict={tfX:xBatch, tfY:yBatch})
                costs.append(c)
                if (j+1) %print_period==0:
                    acc= np.mean(yBatch==np.argmax(lgts,1))
                    print("epoch:", i, "batch:", j, "n_batches:", n_batches, "cost:", c, "acc: %.2f" % acc)
                print("Train acc ",self.score(X,Y),"Test score: ",self.score(xTest,yTest)))
    def forward(self,X,is_training):
        out=X
        for h in self.layers[:,-1]:
            
                
        
        
            