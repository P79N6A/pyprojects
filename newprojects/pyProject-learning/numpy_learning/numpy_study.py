import numpy as np

x = np.array([1,2,3])
y = np.array([[1,2,3,4]])
z = np.array([[1,2,3],[4,5,6]])

print(x)
print(y)
print(z)

# print(x.shape)
# print(y.shape)
# print(z.shape)

print(np.max(z, axis=1))
print(np.max(z, axis=1, keepdims=True))

import numpy as np
x = np.arange(100)
print(x)
x_double = [i*i for i in x]
print(x_double)


# numpy:
# ndarray: 多维数组
#   属性一：shape 大小
#   属性二：dtype 类型
#   属性三：ndim 维度

import numpy as np
list_a = [1,1,2,5]
a_array = np.array(list_a)
print(a_array)
print(a_array.shape)
print(a_array.dtype)

import numpy as np
array_b = [[1,2,3],[4,5,5]]
b_array = np.array(array_b)
print(b_array)
print(b_array.shape)
print(b_array.ndim)

zeros = np.zeros(10)
zero_array = np.zeros((2,3))
emptys = np.empty((1,2,3)) # 可能会返回未初始化的垃圾值

# 向量化：
#   任何在两个等尺寸数组之间的算术操作都应用逐元素操作的方式，
#   带有标量运算的算术操作会把计算参数传递给每给每一个元素

# 广播：不同尺寸数组之间的操作会用到广播的特性
#   沿行方向扩展，复制行：（4，3） + （1，3） ===>  （4，3） + （4，3）：括号中的值表示的是维度
#   沿列方向扩展，复制列：（4，3） + （4，1） ===>  （4，3） + （4，3）：括号中的值表示的是维度
#   沿更高维度扩展

# numpy的切片只是原来数组的视图，不会复制数组，所以对于切片的修改会同步到原始数组上
# 如果需要原始数组的复制而不是视图的话需要显示的调用 arr[3:9].copy()

# 多维数据切片
import numpy as np
arr = np.array([[1,2,3],[4,5,6]])
print(arr[:2,1:])
print(arr[2,1:]) # 指定行 ; 指定列方法类似

# 索引 ：索引操作是对原始数组的复制，而不是视图
#   布尔索引 ==> 通过逻辑判断对数组进行类似切片的操作
#   神奇索引 ==> 不知道是啥玩意 ~












