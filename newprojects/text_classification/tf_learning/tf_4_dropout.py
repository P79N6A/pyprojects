import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data",one_hot=True)

batch_size = 100
n_batch = mnist.train.num_examples // batch_size

x = tf.placeholder(tf.float32,[None,784])
y = tf.placeholder(tf.float32,[None,10])

w1 = tf.Variable(tf.truncated_normal([784,10],stddev=0.1))
b1 = tf.Variable(tf.zeros([10])+0.1)
l1 = tf.nn.tanh(tf.matmul(x,w1) + b1)
l1_dropout = tf.nn.dropout(l1,0.2) # dropout 层的使用，在激活函数之后定义

w2 = tf.Variable(tf.truncated_normal([10,10],stddev=0.1))
b2 = tf.Variable(tf.zeros([10])+0.1)
l2 = tf.nn.tanh(tf.matmul(l1_dropout,w2)+b2)
l2_dropout = tf.nn.dropout(l2,0.1)



