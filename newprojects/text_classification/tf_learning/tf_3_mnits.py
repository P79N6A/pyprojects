import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data",one_hot=True)

batch_size = 100
n_batch = mnist.train.num_examples // batch_size

x = tf.placeholder(tf.float32,[None,784])
y = tf.placeholder(tf.float32,[None,10])

W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

# 定义输出
prediction = tf.nn.softmax(tf.matmul(x,W) + b)

# 定义损失函数
# loss = tf.reduce_mean(tf.square(y-prediction))

# 定义交叉熵损失函数
loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=prediction))

# 定义训练步骤
train_step = tf.train.GradientDescentOptimizer(0.2).minimize(loss=loss)

# 初始化
init = tf.global_variables_initializer()

# 结果存放在一个布尔型列表中
# 返回最大值在list中所在的位置
correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(prediction,1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(21):
        for batch in range(n_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            sess.run(train_step,feed_dict={x:batch_xs,y:batch_ys})

        acc = sess.run(accuracy,feed_dict={x:mnist.test.images,y:mnist.test.labels})
        print("Iter " + str(epoch) + ",Testing Accuracy " + str(acc))
