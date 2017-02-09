import numpy as np
import cv2
import re
import time
import os
import chainer
import glob
from chainer import cuda, Variable, FunctionSet, optimizers, serializers 
import chainer.functions  as F
import chainer.links as L
from PIL import Image



class VGGNet(chainer.Chain):
    """
    VGGNet
    - It takes (224, 224, 3) sized image as imput
    """
    
    def __init__(self):
        super(VGGNet, self).__init__(
            conv1_1=L.Convolution2D(3, 64, 3, stride=1, pad=1),
            conv1_2=L.Convolution2D(64, 64, 3, stride=1, pad=1),
            
            conv2_1=L.Convolution2D(64, 128, 3, stride=1, pad=1),
            conv2_2=L.Convolution2D(128, 128, 3, stride=1, pad=1),
            
            conv3_1=L.Convolution2D(128, 256, 3, stride=1, pad=1),
            conv3_2=L.Convolution2D(256, 256, 3, stride=1, pad=1),
            conv3_3=L.Convolution2D(256, 256, 3, stride=1, pad=1),
            
            conv4_1=L.Convolution2D(256, 512, 3, stride=1, pad=1),
            conv4_2=L.Convolution2D(512, 512, 3, stride=1, pad=1),
            conv4_3=L.Convolution2D(512, 512, 3, stride=1, pad=1),
            
            conv5_1=L.Convolution2D(512, 512, 3, stride=1, pad=1),
            conv5_2=L.Convolution2D(512, 512, 3, stride=1, pad=1),
            conv5_3=L.Convolution2D(512, 512, 3, stride=1, pad=1),
            
            fc6=L.Linear(7*7*512, 4096),
            fc7=L.Linear(4096, 4096),
            fc8=L.Linear(4096,9)
        )
        
    def __call__(self, x, t,train):
        h = F.relu(self.conv1_1(x))
        h = F.relu(self.conv1_2(h))
        h = F.max_pooling_2d(h, 2, stride=2)
        
        h = F.relu(self.conv2_1(h))
        h = F.relu(self.conv2_2(h))
        h = F.max_pooling_2d(h, 2, stride=2)
        
        h = F.relu(self.conv3_1(h))
        h = F.relu(self.conv3_2(h))
        h = F.relu(self.conv3_3(h))
        h = F.max_pooling_2d(h, 2, stride=2)
        
        h = F.relu(self.conv4_1(h))
        h = F.relu(self.conv4_2(h))
        h = F.relu(self.conv4_3(h))
        h = F.max_pooling_2d(h, 2, stride=2)
        
        h = F.relu(self.conv5_1(h))
        h = F.relu(self.conv5_2(h))
        h = F.relu(self.conv5_3(h))
        h = F.max_pooling_2d(h, 2, stride=2)
        
        h = F.dropout(F.relu(self.fc6(h)), train=train, ratio=0.5)
        h = F.dropout(F.relu(self.fc7(h)), train=train, ratio=0.5)
        
        h = self.fc8(h)
        
        self.loss = F.softmax_cross_entropy(h, t)
        self.acc = F.accuracy(h, t)
        self.pred = F.softmax(h)
        return self.loss,self.acc,self.pred



def test(images,model,mean):
    #model=VGGNet()
    #serializers.load_hdf5("VGGface_00010659227118.model",model)##input model path
    #mean=np.load("mean_face.npy")##input mean path
    X = np.transpose(np.array([cv2.resize(im,(224,224)) for im in images],dtype=np.float32),(0,3,1,2))/255.-mean
    t=np.zeros(len(X),dtype=np.int32)
    X = Variable(cuda.to_gpu(X))
    t = Variable(cuda.to_gpu(t))
    _,_,pre=model(X,t,train=False)
    pre = cuda.to_cpu(pre.data)
    classes= ["aragaki","hoshino","narita","other","fujii","mano","ohtani","yamaga","ishida"]
    prediction=np.argmax(pre,axis=1)
    probability=np.max(pre,axis=1)
    strings=[]
    for i,a in enumerate(prediction):
        print(classes[a]+str(probability[i]*100))
        strings.append(classes[a]+str(int(round(probability[i]*100))))
    b=np.sum(prediction==0)
    if b==0:
        return False, strings, prediction,probability
    else:
return True, strings, prediction,probability
