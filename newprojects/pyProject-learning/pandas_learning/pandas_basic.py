# 两种数据结构
#    Series: 存储一维数据 ==> 列表，字典
#    DataFrame: 存储二维甚至高维数据
import pandas as pd
import numpy as np
# -------------------Series -----------------
obj = pd.Series(["abc", "happy", "number", 4])
print(obj)
print("--------------")
obj_index = pd.Series(["i","am","very","happy"],index=[1,2,5,-1])
print(obj_index)
print(obj_index[-1])
print("+++++++++++++++")
# 使用布尔逻辑操作进行索引 或使用数学函数进行操作(操作都是针对数值的，不是针对索引的)
obj_index = pd.Series([1,2,5,-1],index=["a","b","c","d"])
print(obj_index[obj_index > 0])
print("===============")
print(obj_index * 2)
print(np.exp(obj_index))
# 可以利用字典直接构建series，key作为index
# 不知道有什么鬼用的name属性
obj_index.name = 'happy'
print(obj_index.name)

# -------------------DataFrame ---------------
# 特性：
#   表示矩阵的数据表，包含已经排好序的列集合，每一行可以是不同的值类型（数值、字符串、布尔值等）
#   既有行索引，也有列索引
#   可以通过等长度列表活numpy,字典来构造DataFrame(字段key作为column key，index需要额外添加或使用默认的序号索引)
# 属性：
#   head()
import pandas as pd
data = {"state":["ohio","happy","me"],"year":["1989","1098","1678"]}
frameData = pd.DataFrame(data)
print(frameData)

print(frameData.head(1))
print(pd.DataFrame(data,columns=["year"]))  # 指定列
print(frameData["state"])

#   .loc[''] 索引列
#   del data['']删除列
#   data[''] 创建列，如果列不存在
print(frameData.loc[0])
frameData["new_clo"] = frameData["state"] == "ohio"
print(frameData)
print("索引==================>")
print(frameData[:2])
print("索引==================>")
del frameData["new_clo"]
print(frameData)

# 从DataFrame中选取的列是数据的视图，不是拷贝；如果需要拷贝需要显示调用.copy()方法

# ---------------------索引、过滤-----------------
# Series ：index关键字索引、index序号索引、切片【pandas切片包含尾部数据】索引、逻辑操作索引
# DataFrame ：
#    索引列：关键字，关键字构成的list，
#    索引行：切片
#    行列一起索引：
#       loc: 关键字做一年行或列，行index，列columns对应的关键字；该方式还可以运用于切片
#       iloc: 根据整数标签进行索引；该方式还可以运用于切片





