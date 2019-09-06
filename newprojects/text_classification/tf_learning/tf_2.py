import tensorflow as tf
import numpy as np
# 创建一个常量
m1 = tf.constant([[3,3]])
m2 = tf.constant([[2],[2]])
# 创建一个矩阵常量
product = tf.matmul(m1,m2)
print(product)

# 定义一个会话，启动默认图
ss = tf.Session()
# 调用session的run方法执行矩阵乘法
result = ss.run(product)
print(result)
ss.close() # 需要手动关闭

with tf.Session() as sess:
    result = sess.run(product)
    print(result)


# -----------变量--------------
x = tf.Variable([1,2])
a = tf.constant([3,3])
sub = tf.subtract(x,a)
add = tf.add(x,a)

# 变量需要初始化，否则会出现 "Attempting to use uninitialized value Variable"
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    print(sess.run(sub))
    print(sess.run(add))


input1 = tf.constant(3.0)
input2 = tf.constant(2.0)
input3 = tf.constant(5.0)

add = tf.add(input1,input2)
mul = tf.multiply(input1,add)

# fetch 使用：一次性run多个operation
with tf.Session() as sess:
    result = sess.run([mul,add])
    print(result)

input4 = tf.placeholder(tf.float32)
input5 = tf.placeholder(tf.float32)
output = tf.multiply(input4,input5)
# feed 用法，使用占位符，以及在run的时候用feed_dict进行载入
with tf.Session() as sess:
    print(sess.run(output,feed_dict = {input4:[8.0],input5:[2.]}))


x_data = np.random.rand(100)
y_data = x_data * 0.1 + 0.2

# 构造一个线性模型
b = tf.Variable(0.)
k = tf.Variable(0.)
y = k*x_data + b

# 定义一个二次代价函数
loss = tf.reduce_mean(tf.square(y_data - y)) # reduce是求根号，reduce_mean是平方后求平均值
# 定义一个优化器
optimizer = tf.train.GradientDescentOptimizer(0.2)
# 最小化代价函数
train = optimizer.minimize(loss=loss)
# 初始化变量
init = tf.global_variables_initializer() # 注意init的使用位置

with tf.Session() as sess:
    sess.run(init)
    for step in range(201):
        sess.run(train)
        if step%20 == 0:
            print(step,sess.run([k,b]))
