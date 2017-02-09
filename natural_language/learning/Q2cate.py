#tensorboard --logdir=/home/mech-user/tensorflow/rnn/log4
import tensorflow as tf
from gensim.models import word2vec
import random
from sklearn import cross_validation
import numpy as np
import json
import random

with open('create_data/mul_class/sentence_class_freq.json') as f:
    sentenceList=json.load(f)
with open('create_data/mul_class/class_list.json') as g:
    class_list=json.load(g)
with open('create_data/mul_class/sentence_pro.json') as h:
    sentence_prob=json.load(h)

model=word2vec.Word2Vec.load('create_data/word2vec/chie_navi_uni200.model')

#-------------------------------------------------------------
max_length=100
wordVec_length=200
num_class=len(class_list)
num_hidden1=200
num_hidden2=200
num_batch=10
filename="save4/Q2cate.ckpt"
#-------------------------------------------------------------
def create_dataset(num_of_batch):
    index=np.random.choice(len(sentenceList),num_of_batch,p=sentence_prob)
    data=[]
    te_=[class_list.index(sentenceList[i]["class_"]) for i in index]
    te=np.zeros([num_of_batch,num_class])
    te[np.arange(num_of_batch),te_]=1
    for i in index:
        da=np.zeros((max_length,wordVec_length))
        da_=np.array([model[w] for w in sentenceList[i]['q'] if w in model.vocab])
        da[:len(da_),:]=da_
        data.append(da)
    return index,np.array(data),te



def length(data):
    used = tf.sign(tf.reduce_max(tf.abs(data), reduction_indices=2))
    length = tf.reduce_sum(used, reduction_indices=1)
    length = tf.cast(length, tf.int32)
    return length

data=tf.placeholder(tf.float32,[None,max_length,wordVec_length])
t=tf.placeholder(tf.float32,[None,num_class])

with tf.variable_scope('encoder1') as encoder:
    out1, state1 = tf.nn.dynamic_rnn(
        tf.nn.rnn_cell.GRUCell(num_hidden1),
        data,
        dtype=tf.float32,
        sequence_length=length(data),)

with tf.variable_scope('encoder2') as encoder:
    out2, state2 = tf.nn.dynamic_rnn(
        tf.nn.rnn_cell.GRUCell(num_hidden2),
        out1,
        dtype=tf.float32,
        sequence_length=length(data),)


W=tf.Variable(tf.truncated_normal([num_hidden2,num_class],stddev=0.1))
b=tf.Variable(tf.truncated_normal([1,num_class],stddev=0.1))

prediction=tf.nn.softmax(tf.matmul(state2,W)+b)
loss=-tf.reduce_mean(tf.reduce_sum(t*tf.log(prediction),reduction_indices=[1]))

sess=tf.InteractiveSession()
sess.run(tf.initialize_all_variables())
training=tf.train.GradientDescentOptimizer(0.3).minimize(loss)
#with tf.name_scope("train") as scope:
#    loss_summary_train=tf.scalar_summary("train_acc",loss)
#with tf.name_scope("test") as scope:
#
#writer=tf.train.SummaryWriter("log4",sess.graph)
#mergeしてしまうと、すべての定義しているサマリーが追記されてしまう
s=tf.scalar_summary('loss',loss)
merged=tf.merge_all_summaries()
writer=tf.train.SummaryWriter('log4',sess.graph)
saver=tf.train.Saver()
saver.restore(sess,filename+"-16070")

for epoch in range(1000000):
    _,sentence,class_=create_dataset(num_batch)
    sess.run(training,feed_dict={data:sentence,t:class_})
    m_str=sess.run(merged,feed_dict={data:sentence,t:class_})
    if epoch%10==0:
        print(epoch+16070)
        index,sentence,class_=create_dataset(1)
        pre=sess.run(prediction,feed_dict={data:sentence,t:class_})
        pre2=np.argsort(np.array(pre[0]))[-3:]
        print(pre2)
        print("".join(sentenceList[index[0]]["q"]))
        print("ans: "+sentenceList[index[0]]["class_"])
        print("pre1: "+class_list[pre2[2]]),
        print("      ",pre[0][pre2[2]])
        print("pre2: "+class_list[pre2[1]]),
        print("      ",pre[0][pre2[1]])
        print("pre3: "+class_list[pre2[0]]),
        print("      ",pre[0][pre2[0]])
        saver.save(sess,filename,global_step=epoch+16070)
    writer.add_summary(m_str,epoch+16070)


