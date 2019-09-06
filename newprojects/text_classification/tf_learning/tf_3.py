import tensorflow as tf
import numpy as np
import matplotlib as plt

x_data = np.linspace(-0.5,0.5,200)[:,np.newaxis]
noise = np.random.normal(0,0.02,x_data.shape)
y_data = np.square(x_data) + noise

# 定义两个placeholder
x = tf.placeholder(tf.float32,[None,1])
y = tf.placeholder(tf.float32,[None,1])

# 定义神经网络的中间层
weight_l1 = tf.Variable(tf.random_normal([1,10]))
biases_l1 = tf.Variable(tf.zeros([1,10]))
wx_plus_b_l1 = tf.matmul(x,weight_l1) + biases_l1
l1 = tf.nn.tanh(wx_plus_b_l1)
print("l1 shape",l1.shape)
# 定义输出层
weight_l2 = tf.Variable(tf.random_normal([10,1]))
biases_l2 = tf.Variable(tf.zeros(1,1))
wx_plus_b_l2 = tf.matmul(l1,weight_l2) + biases_l2
prediction = tf.nn.tanh(wx_plus_b_l2)
print("y shape:",y.shape)
print("predicstion shape:",prediction.shape)

# 定义二次代价函数
loss = tf.reduce_mean(tf.square(y - prediction))
# 使用梯度下降法进行训练
train_step = tf.train.GradientDescentOptimizer(0.2).minimize(loss=loss)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    # 变量初始化
    sess.run(init)
    for _ in range(2000):
        sess.run(train_step,feed_dict={x:x_data,y:y_data})

    # 获取预测值
    prediction_value = sess.run(prediction,feed_dict={x:x_data})

    # plt.figure()
    # plt.scatter(x_data,y_data)
    # plt.plot(x_data,prediction_value,'r-',lw=5)
    # plt.show()
