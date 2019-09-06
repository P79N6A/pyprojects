# pandas读取文件：csv & json
# 读取csv：
import pandas as pd
data = pd.read_csv("path",chunksize= 1000) # 分块读入,每一块1000行
data = pd.read_csv("path",)
# 写入csv
data.to_csv("path")

# 读取json
# python：
#   读取 ==> python_object = json.loads(json) ;
#   写入 ==> json =  json.dumps(python_object);
data = pd.read_json("path_or_pro")