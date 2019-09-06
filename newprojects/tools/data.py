# from urllib import request
#
# res = request.urlopen("http://tce.byted.org/api/v3/clusters/7248/instances/")
# print("------------------------")
# print(res.read().decode('utf-8'))

import numpy as np
x = np.array([[1,2,3],[4,5,6],[7,8,9]])
y = np.array([[1,2,3],[4,5,6]])
print(np.dot(x,y))