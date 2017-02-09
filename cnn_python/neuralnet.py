import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.utils import shuffle
from sklearn.datasets import fetch_mldata
import matplotlib.pyplot as plt

def add_noise(train_X,d):
    d/=100.
    shape=train_X.shape
    train_X_=train_X.copy()
    train_X_=train_X_.reshape(shape[0],-1)
    batch,length=train_X_.shape
    indices=np.random.permutation(np.arange(length))[:int(length*d)]
    train_X_[:,indices]=0
    tmp_X=np.zeros(train_X_.shape)
    tmp_X[:,indices]=np.random.rand(int(length*d))
    plt.imshow((train_X_+tmp_X)[1].reshape(28,28), cmap = 'gray')
    plt.show()
    return (train_X_+tmp_X).reshape(shape)

mnist = fetch_mldata('MNIST original')
mnist_X, mnist_y = shuffle(mnist.data, mnist.target.astype('int32'),
                           random_state=42)

mnist_X = mnist_X.reshape((-1,1,28,28)).astype(np.float32) / 255.0

mnist_y=np.eye(10)[mnist_y]
X_train, X_test, y_train, y_test = train_test_split(mnist_X, mnist_y,
                                                    test_size=0.2,
                                                    random_state=1)

N_train=len(X_train)
N_test=len(X_test)

class Sigmoid:
    def __call__(self, x):
        return 1/(1+np.exp(-x))
    
    def deriv(self, x):
        return self(x)*(1-self(x))

class ReLU:
    def __call__(self, x):
        return x*(x>0)
    
    def deriv(self, x):
        return 1*(x>0)

class Softmax:
    def __call__(self, x):
        return np.exp(x)/np.sum(np.exp(x),axis=1,keepdims=True)

class Linear:
    def __init__(self, in_dim, out_dim, activation):
        self.W = np.random.uniform(low=-0.08,high=0.08,size=[in_dim,out_dim]).astype(np.float32)
        self.b = np.zeros(out_dim,dtype="float32")
        self.delta = None
        self.activation = activation()

    def __call__(self, x):
        x=x.reshape((-1,self.W.shape[0]))
        self.x=x
        self.u = np.dot(x,self.W)+self.b
        self.z = self.activation(self.u)
        return self.z

    def backward(self,delta,W):
        self.delta=np.dot(delta,W.T)*self.activation.deriv(self.u)

    def update(self,lr):
        z=self.x.reshape((self.u.shape[0],-1))
        dW=np.dot(z.T,self.delta)/(self.u.shape[0])
        db=np.dot(np.ones(self.u.shape[0]),self.delta)/(self.u.shape[0])
        self.W-=lr*dW
        self.b-=lr*db
        return dW,db

class Conv:
    def __init__(self,in_fil,out_fil,kernel_size,activation,pad=0):
        self.W=np.random.uniform(low=-0.08,high=0.08,size=[out_fil,in_fil,kernel_size,kernel_size]).astype(np.float32)
        self.b=np.zeros(out_fil,dtype="float32")
        self.k=kernel_size
        self.activation=activation()
        self.padding=pad
        
    def __call__(self,X):
        if len(X.shape)==3:
            X=X.reshape((1,X.shape[0],X.shape[1],X.shape[2]))
        X_=np.zeros([X.shape[0],X.shape[1],X.shape[2]+2*self.padding,X.shape[3]+2*self.padding])
        X_[:,:,self.padding:X_.shape[2]-self.padding,self.padding:X_.shape[3]-self.padding]=X
        self.X=X_
        batch_num,k,length,_=X_.shape
        out_len=length-self.k//2*2
        m_=self.W.shape[0]
        out_image=np.empty([batch_num,m_,out_len,out_len])
        for b in range(batch_num):
            for m in range(m_):
                for j in range(out_len):
                    for i in range(out_len):
                        out_image[b,m,j,i]=np.sum(X_[b,:,j:j+self.k,i:i+self.k]*self.W[m,:,:,:])+self.b[m]
        self.u=out_image
        self.z=self.activation(out_image)
        return self.z
    
    def backward(self,delta,W):
        if len(W.shape)==2:
            self.delta=np.dot(delta,W.T).reshape(self.u.shape)*self.activation.deriv(self.u)
        else:
            m=self.k//2
            pad_delta=np.empty([delta.shape[0],delta.shape[1],delta.shape[2]+m*2,delta.shape[3]+m*2])
            pad_delta[:,:,m:pad_delta.shape[2]-m,m:pad_delta.shape[3]-m]=delta
            out_im=np.empty([delta.shape[0],self.W.shape[0],delta.shape[2],delta.shape[3]])
            for b in range(out_im.shape[0]):
                for m in range(out_im.shape[1]):
                    for j in range(out_im.shape[2]):
                        for i in range(out_im.shape[3]):
                            out_im[b,m,j,i]=np.sum(pad_delta[b,:,j+self.k:j:-1,i+self.k:i:-1]*W[:,m,:,:])
            self.delta=out_im*self.activation.deriv(self.u)

    def update(self,lr):
        delta=np.average(self.delta,axis=0)
        X=np.average(self.X,axis=0)
        dW=np.empty(self.W.shape)
        db=np.empty([self.W.shape[0]])
        for k in range(self.W.shape[0]):
            for c in range(self.W.shape[1]):
                for s in range(self.W.shape[2]):
                    for t in range(self.W.shape[3]):
                        dW[k,c,s,t]=np.sum(delta[k,:,:]*X[c,s:s+self.u.shape[2],t:t+self.u.shape[3]])
        for k in range(self.W.shape[0]):
            db[k]=np.sum(delta[k,:,:])
        self.W-=lr*dW
        self.b-=lr*db
        return dW,db

class model:
    def __init__(self,layers):
        self.layers=layers

    def set_lr(self,lr=0.01):
        self.lr=lr

    def train(self,X,t):
        self.y=X
        for layer in self.layers:
            self.y=layer(self.y)
        self.loss=-np.sum(np.log(self.y)*t)/len(X)

        delta=self.y-t
        self.layers[-1].delta=delta
        W=self.layers[-1].W
        for layer in self.layers[-2::-1]:
            layer.backward(delta,W)#deltaを計算するメソッドを書くこと
            delta=layer.delta
            W=layer.W

        
        #パラメータの更新
        for layer in self.layers:
            dW,db=layer.update(self.lr)#updateする関数を書くこと

        return self.loss

    def test(self,X,t):
        self.y=X
        for layer in self.layers:
            self.y=layer(self.y)
        self.loss=-np.sum(np.log(self.y)*t)/len(X)
        return self.loss

#model=model([Conv(1,5,3,ReLU,pad=1),
#             Linear(784*5,784*3,Sigmoid),
#             Linear(784*3,784,Sigmoid),
#             Linear(784,300,Sigmoid),
#             Linear(300,10,Softmax)])
n_epoch=40
batch_num=30
lr=0.1

train_acc=np.zeros((9,40))
test_acc=np.zeros((9,40))
model1=model([Linear(784,30,Sigmoid),
              #Linear(500,500,Sigmoid),
              #Linear(500,100,Sigmoid),
              Linear(30,10,Softmax)])


model1.set_lr(lr)
#print("%d noise added" % d)
for epoch in range(n_epoch):
    print("epoch %d " %epoch),
    sum_loss=0
    pred_y=[]
    perm=np.random.permutation(N_train)
    for i in range(0,N_train,batch_num):
        X=X_train[perm[i:i+batch_num]]
        t=y_train[perm[i:i+batch_num]]
        
        sum_loss+=model1.train(X,t)*len(X)
        pred_y.extend(np.argmax(model1.y,axis=1))
    loss=sum_loss/N_train
    accuracy=np.sum(np.eye(10)[pred_y] * y_train[perm]) / N_train
    print('Train loss %.3f, accuracy %.4f '%(loss, accuracy)),
    sum_loss=0
    pred_y=[]
    
    for i in range(0,N_test,batch_num):
        X=X_test[i:i+batch_num]
        t=y_test[i:i+batch_num]
        
        sum_loss+=model1.test(X,t)*len(X)
        pred_y.extend(np.argmax(model1.y,axis=1))
    loss=sum_loss/N_test
    accuracy=np.sum(np.eye(10)[pred_y] * y_test) / N_test
    print('Test loss %.3f, accuracy %.4f' %(loss, accuracy))

for i in range(30):
    plt.subplot(5,6,i+1)
    plt.imshow(model1.layers[0].W[:,i].reshape(28,28),cmap="gray")
plt.show()
