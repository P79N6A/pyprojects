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
prediction = tf.nn.softmax(tf.matmul(l1_dropout,w2)+b2)

# 要记得加tf.reduce_mean
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y,logits=prediction))

train_step = tf.train.MomentumOptimizer(learning_rate=0.01,momentum=0.9).minimize(loss)

init = tf.global_variables_initializer()

correct = tf.equal(tf.argmax(prediction),tf.argmax(y))

acc = tf.reduce_mean(tf.cast(correct,tf.float32))


with tf.Session() as sess:
    sess.run(init)
    for i in range(20):
        for batch in range(n_batch):
            batch_x,batch_y = mnist.train.next_batch(batch_size)
            sess.run(train_step,feed_dict={x:batch_x,y:batch_y})

    test_x,test_labels = mnist.test.images, mnist.test.labels
    sess.run(acc,feed_dict={x:test_x,y:test_labels})

