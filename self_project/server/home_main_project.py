#-*-coding=utf8-*-
import socket
import numpy as np
import cv2
import re
import time
import os
import chainer
import glob
import shutil
from chainer import cuda, Variable, FunctionSet, optimizers, serializers 
import chainer.functions  as F
import chainer.links as L
from PIL import Image

#socket通信により画像を10枚連続で受け取り、顔の名前を返す


def collect_faces():
    i=0
    if os.path.isdir("./tmp"):
        shutil.rmtree("./tmp")
    os.mkdir('./tmp')
    while(1):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("10.10.1.122", 6677+i))    # サーバプログラムが動くホスト(IP)とポートをソケットに設定
        s.listen(1)                     # 1つの接続要求を待つ
        soc, addr = s.accept()          # 要求が来るまでブロック
        print("Conneted by"+str(addr))  #サーバ側の合図
        recvlen=100
        buf=""
        while recvlen>0:
            print ("receiving")
            receivedStr=soc.recv(1024*8)
            recvlen=len(receivedStr)
            buf+=receivedStr
        soc.close()
        i+=1
        if i==10:
            break
        try:
            narray=np.fromstring(buf,dtype="uint8")
            image=cv2.imdecode(narray,1)
            cv2.imwrite("./tmp/sample"+str(i)+".jpg",image)
            print ("finish")
        except:
            pass


def test():

    class VGGNet(chainer.Chain):
        """
        VGGNet
        - It takes (96, 96, 3) sized image as imput
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
                
                fc6=L.Linear(3*3*512, 200),
                #fc7=L.Linear(512, 200),
                fc8=L.Linear(200,10)
                # the output dim '10' is set up to CIFAR-10. Please change this.
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
            #h = F.dropout(F.relu(self.fc7(h)), train=train, ratio=0.5)
            
            h = self.fc8(h)
                
            self.loss = F.softmax_cross_entropy(h, t)
            self.acc = F.accuracy(h, t)
            self.pred = F.softmax(h)
            return self.loss,self.acc,self.pred
    model=VGGNet()
    serializers.load_hdf5('VGG_000621006664932.model',model)########## input model path
    mean=np.load("./mean.npy")
    image_files=sorted(glob.glob("./tmp/*.jpg"))
    X=[]
    for image_file in image_files:
        X.append(np.transpose(np.array(Image.open(image_file).resize((96,96))),(2,0,1))/255.)
    t=np.zeros(len(X))
    X=np.array(X,dtype=np.float32)
    t=np.array(t,dtype=np.int32)
    X-=mean
    _,_,pre=model(X,t,train=False)
    classes=["ちゃんおにさん","かりのくん","かんくん","みきひろさん","のぐちくん","しんじょうくん","ゆうまくん","たけもとくん","やすあき","れな"]

    name=classes[np.argmax(np.sum(pre.data,axis=0))]
    return name

def send_name(name):
    ###誰と識別されたかをraspiに送る
    while(1):
        try:
            soc=socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            soc.connect(("10.10.3.77",6666))
            break
        except:
            pass
    soc.send(name)
    soc.close()

if __name__ == '__main__':
    flag=0
    name=""
    while(1):
        if flag==0:
            print("image waiting")
            collect_faces()
            flag=1
        if flag==1:
            name=test()
            flag=2
            print(name)
        if flag==2:
            print("sending name")
            send_name(name)
            flag=0
