import  re
import numpy as np
from tensorflow.contrib import learn
import tensorflow as tf

from sklearn.model_selection import KFold
output_dim=2
embedding_dim=100
epochs=10
batch_size=64
cvfold = 10
root = "rt-polaritydata/"
#prepare
def clean(string):
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

def get_data():
    x_pos = [clean(s.strip()) for s in open(root+"rt-polarity.pos","r",encoding="utf-8").readlines()]
    x_neg = [clean(s.strip()) for s in open(root + "rt-polarity.neg", "r",encoding="utf-8").readlines()]
    x = x_neg+x_pos
    y = [[0,1]]*len(x_neg)+[[1,0]]*len(x_pos)
    index = np.random.permutation(len(x))
    x = np.array(x)[index]
    y = np.array(y)[index]
    length = 40
    vocab = learn.preprocessing.VocabularyProcessor(length)
    x = np.array(list(vocab.fit_transform(x)))
    return x,y,length,len(vocab.vocabulary_)


def weight_variable(shape):
    return tf.Variable(tf.truncated_normal(shape=shape, stddev=0.1))

def b_variable(shape):
    return tf.Variable(tf.constant(0.1,shape=shape))

def batch_generate(x,y,batch_size,epochs):
    x_length = len(x)
    num_batch_epoch = (x_length-1)//batch_size+1
    for num in range(epochs):
        for i in range(num_batch_epoch):
            start = i*batch_size
            end = min((i+1)*batch_size,x_length)
            yield x[start:end],y[start:end]

def train():
    x,y,length,vocab_size=get_data()
    input = tf.placeholder(tf.int32,[None,length])
    Y = tf.placeholder(tf.int32,[None,output_dim])
    keep_prob = tf.placeholder(tf.float32)

    with tf.name_scope("embed"):
        W = weight_variable([vocab_size,embedding_dim])
        embedded = tf.expand_dims(tf.nn.embedding_lookup(W,input),3)
    #main block with 3 conv branch
    outputs = []
    for i,size in enumerate([3,4,5]):
        with tf.name_scope("conv_%d"%i):
            w1 = weight_variable([size,embedding_dim,1,100])
            b1 = b_variable([100])
            conv = tf.nn.relu(tf.nn.conv2d(embedded,w1,strides=[1,1,1,1],padding="VALID")+b1)
            pool = tf.nn.max_pool(conv,ksize=[1,length-size+1,1,1],strides=[1,1,1,1],padding="VALID")
            outputs.append(pool)

    out = tf.reshape(tf.concat(outputs,3),[-1,3*100])
    drop = tf.nn.dropout(out,keep_prob=keep_prob)

    with tf.name_scope("linear"):
        w2 = weight_variable([3*100,output_dim])
        b2 = b_variable([output_dim])
        final = tf.matmul(drop,w2)+b2
        y_softmax = tf.nn.softmax(final)
    with tf.name_scope("loss"):
        loss1 = tf.nn.softmax_cross_entropy_with_logits(labels=Y,logits=final)
        loss = tf.reduce_mean(loss1)

    with tf.name_scope("optimize"):
        global_step = tf.Variable(0, trainable=False)
        learning_rate = tf.train.exponential_decay(1e-2, global_step=global_step, decay_steps=10, decay_rate=0.95)
        train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step)


    kf = KFold(n_splits=10)

    for train_index, test_index in kf.split(x):
        x_train, x_dev, y_train, y_dev = x[train_index], x[test_index], y[train_index], y[test_index]
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            batch = batch_generate(x_train,y_train,batch_size,epochs)
            i=0
            for x_batch,y_batch  in batch:
                x_batch = np.array(x_batch)
                sess.run(train_step,feed_dict={input:x_batch,Y:y_batch,keep_prob:0.5})
                sess.run(tf.assign(w2,3*w2/tf.norm(w2)))
                i+=1
                if(i==30):
                    loss_valid, result_valid = sess.run([loss, y_softmax],feed_dict={input: x_dev, Y: y_dev, keep_prob: 1.0})
                    accuracy_valid = np.sum(np.argmax(result_valid, axis=1) == np.argmax(y_dev, axis=1)) / len(y_dev)
                    print("loss:%.4f " % loss_valid, "accuracy:%.4f" % accuracy_valid)

train()
