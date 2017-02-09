#dynamic_rnnを使用
#最後の一文字を予測
#tensorboard --logdir=/home/mech-user/tensorflow/rnn/logでグラフが見れる

#tf.nn.sampled_softmax_loss
import tensorflow as tf
from gensim.models import word2vec
import random
from sklearn import cross_validation
import numpy as np
import json
import random
random.seed(0)

#from tensorflow.nn.rnn_cell import GRUCell
#-----------------------------------------------------------------
#                 file load
#with open('create_data/sentence23_with_border400.json') as f:
#    sentenceList=json.load(f)
#sentenceList=[x for x in sentenceList if x]
#sentenceList=sentenceList[0:-1] 
#sen1=sentenceList[::2]
#sen2=sentenceList[1::2]
#sentenceList=[x[0:-1] for x in sentenceList] 
#with open('create_data/wordList23_with_border400.json') as g:
#    wordList=json.load(g)
#wordList.append('SOS')

with open('create_data/mynavi_chie/mynavi_chie_freq10_len100.json') as f:
    sentenceList=json.load(f)
#a=[]
#for part in sentenceList:
#    a.append(len(part))
#print(max(a))
for part in sentenceList:
    part.append('EOS')
sen1=sentenceList[::2]
sen2=sentenceList[1::2]
sentenceList=[]
with open('create_data/mynavi_chie/mynavi_chie_freq10_len100words.json') as g:
    wordList=json.load(g)
wordList.append('SOS')
wordList.append('EOS')

#-----------------------------------------------------------------

max_length = 100#文のmax長さ
max_length+=1#'EOS'をいれるため
frame_size = len(wordList)#単語ベクトルの次元
num_w=300
num_hidden1 = 200
num_hidden2 = 200
num_classes = len(wordList)
num_of_batch=10
#num_of_sample = len(talk)
filename='save/QtoA_2.ckpt'
#-----------------------------------------------------------------


def length(data):
    used = tf.sign(tf.reduce_max(tf.abs(data), reduction_indices=2))
    length = tf.reduce_sum(used, reduction_indices=1)
    length = tf.cast(length, tf.int32)
    return length



data1 = tf.placeholder(tf.float32, [None, max_length, frame_size])
data2 = tf.placeholder(tf.float32, [None, max_length, frame_size])
t = tf.placeholder(tf.float32, [None, max_length,num_classes])

W = tf.Variable(tf.truncated_normal([frame_size, num_w], stddev=0.1))
b = tf.Variable(tf.truncated_normal([num_w], stddev=0.1))
dada1=tf.sigmoid(tf.matmul(tf.reshape(data1,[-1,frame_size]),W)+b)
dada2=tf.sigmoid(tf.matmul(tf.reshape(data2,[-1,frame_size]),W)+b)
dada1=tf.reshape(dada1,[-1,max_length,num_w])
dada2=tf.reshape(dada2,[-1,max_length,num_w])

with tf.variable_scope('encoder1') as encoder:
    out1, stat1 = tf.nn.dynamic_rnn(
        tf.nn.rnn_cell.GRUCell(num_hidden1),
        dada1,
        dtype=tf.float32,
        sequence_length=length(data1),)


with tf.variable_scope('encoder2') as encoder:
    output1, state1 = tf.nn.dynamic_rnn(
        tf.nn.rnn_cell.GRUCell(num_hidden2),
        out1,
        dtype=tf.float32,
        sequence_length=length(data2),)

with tf.variable_scope('decoder1') as encoder:
    out2, stat2 = tf.nn.dynamic_rnn(
        tf.nn.rnn_cell.GRUCell(num_hidden2),
        dada2,
        initial_state=state1,
        dtype=tf.float32,
        sequence_length=length(data2),)


with tf.variable_scope('decoder2') as encoder:
    output2, state2 = tf.nn.dynamic_rnn(
        tf.nn.rnn_cell.GRUCell(num_hidden2),
        out2,
        initial_state=state1,
        dtype=tf.float32,
        sequence_length=length(data2),)


#print(output.get_shape())

def cost(output, target):
    #cross_entropy = tf.reduce_mean(-tf.reduce_sum(target * tf.log(output), reduction_indices=[1]))
    
    # Compute cross entropy for each frame.
    cross_entropy = target * tf.log(output)
    cross_entropy = -tf.reduce_sum(cross_entropy, reduction_indices=2)
    mask = tf.sign(tf.reduce_max(tf.abs(target), reduction_indices=2))
    cross_entropy *= mask
    # Average over actual sequence lengths.
    cross_entropy = tf.reduce_sum(cross_entropy, reduction_indices=1)
    cross_entropy /= tf.reduce_sum(mask, reduction_indices=1)
    #loss = tf.reduce_mean(tf.square(output - target))
    
    loss = cross_entropy
    
    return loss



def last_relevant(output, length):#最後のoutput
    batch_size = tf.shape(output)[0]
    max_length = int(output.get_shape()[1])
    out_size = int(output.get_shape()[2])
    index = tf.range(0, batch_size) * max_length + (length - 1)
    flat = tf.reshape(output, [-1, out_size])
    relevant = tf.gather(flat, index)
    return relevant


#last = last_relevant(output2,length(data))
output2_ = tf.reshape(output2,[-1,num_hidden2])
weight = tf.Variable(tf.truncated_normal([num_hidden2, num_classes], stddev=0.1))
bias = tf.Variable(tf.truncated_normal([num_classes], stddev=0.1))
prediction = tf.nn.softmax(tf.matmul(output2_, weight) + bias)
prediction = tf.reshape(prediction,[-1,max_length,num_classes])
#output_reshape=tf.reshape(output2,[-1,num_hidden2])
#output3=tf.reshape(tf.matmul(output_reshape,weight)+bias,[-1,max_length,frame_size])
loss = cost(prediction,t)

sess = tf.InteractiveSession()
sess.run(tf.initialize_all_variables())
training = tf.train.GradientDescentOptimizer(0.03).minimize(loss)
s=tf.scalar_summary('loss',tf.reduce_mean(loss))
merged=tf.merge_all_summaries()
writer=tf.train.SummaryWriter('log',sess.graph)
saver=tf.train.Saver()
saver.restore(sess,filename+'-81500')
"""
def makedata(sentence):
    data = np.zeros((max_length,frame_size))
    for i,word in enumerate(sentence):
        d = np.zeros(frame_size)
        d[wordList.index(word)]=1
        data[i]=d
    return data
"""
def create_dataSet(num,sen1,sen2):
    data1=np.zeros((max_length,frame_size))
    data2=np.zeros((max_length,frame_size))
    test=np.zeros((max_length,frame_size))
    da1=[]
    da2=[]
    te=[]
    k=0
    for i in range(num):
        index=random.randrange(len(sen2)-1)
        for i,word in enumerate(sen1[index]):
            d=np.zeros(frame_size)
            d[wordList.index(word)]=1
            data1[i]=d
        for i,word in enumerate(sen2[index][:-1]):
            d=np.zeros(frame_size)
            if i==0:
                d[wordList.index('SOS')]=1
                data2[0]=d
            elif i==len(sen2[index]):
                break
            d[wordList.index(word)]=1
            data2[i+1]=d
        for i,word in enumerate(sen2[index]):
            d=np.zeros(frame_size)
            d[wordList.index(word)]=1
            test[i]=d
        da1.append(data1)
        da2.append(data2)
        te.append(test)
    return np.array(da1),np.array(da2),np.array(te),
            
    da=[]
    te=[]
    for i in range(num):
        index=random.randrange(len(sen2)-1)
        da.append(QVec[index])
        te.append(AVec[index])
    return da,te
"""
QVec=[]
AVec=[]
print(1)
for sentence in sen1:
    dat= makedata(sentence)
    QVec.append(dat)
for sentence in sen2:
    dat=makedata(sentence)
    AVec.append(dat)
print(2)
QVec = np.array(QVec)
AVec = np.array(AVec)
"""
for epoch in range(100000):
    da1,da2,te=create_dataSet(num_of_batch,sen1,sen2)
    #print(da1.shape)
    #print(da2.shape)
    #print(te.shape)
    sess.run(training, {data1: da1,data2:da2,t:te})
    m_str=sess.run(merged,feed_dict={data1: da1,data2:da2,t:te})
    if epoch%10==0:
        print(epoch+81500)
        da1,da2,te=create_dataSet(1,sen1,sen2)
        pre = sess.run(prediction, feed_dict={data1: da1,data2:da2,t:te})
        pre = np.argmax(np.array(pre),axis = 2)
        #print(pre[0])
        #print(len(pre[0]))
        saver.save(sess,filename,global_step=epoch+81500)#epoch+?
        print(loss.get_shape)
        print('out',np.argmax(sess.run(output1, feed_dict={data1: da1,data2:da2,t:te}),axis=2))
        print('in',np.argmax(te,axis=2))
        print('pre',np.argmax(sess.run(prediction, feed_dict={data1: da1,data2:da2,t:te}),axis=2))
        print('pre',[wordList[p] for p in pre[0]])
        print('ans',[wordList[c] for c in np.argmax(te,axis=2)[0]])
        print('loss : ',sess.run(loss, feed_dict={data1: da1,data2:da2,t:te}))
    writer.add_summary(m_str,epoch+81500)
    """
    ts = 100/num_of_sample
    x_left,inputs,t_left,supervisors = cross_validation.train_test_split(X,t,test_size =ts,)
    train_dict = {
        input_ph:      inputs,
        supervisor_ph: supervisors,
        istate_ph:     np.zeros((len(supervisors), num_of_hidden_nodes )),
    }
    sess.run(training_op, feed_dict=train_dict)
    """
    """
    if (epoch ) % 100 == 0:
        summary_str, train_loss = sess.run([summary_op, loss_op], feed_dict=train_dict)
        print("train#%d, train loss: %e" % (epoch, train_loss))
        if (epoch ) % 500 == 0:
        calc_accuracy(output_op)
    """
#calc_accuracy(output_op, prints=True)
