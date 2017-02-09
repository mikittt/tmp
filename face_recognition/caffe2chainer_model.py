from chainer.links.caffe import CaffeFunction
import pickle

vgg=CaffeFunction("VGG_FACE.caffemodel")

pickle.dump(vgg,open("vgg.pkl","wb"),protocol=2)
