# -*-coding:utf-8 -*-

import random

user_dict = {}
user_dict[0] = '陈弯'
user_dict[1] = '吴虹洁'
user_dict[2] = '邹远平'
user_dict[3] = '高玉军'
user_dict[4] = '时会升'
user_dict[5] = '赵斯琦'
user_dict[6] = '廖尧'
user_dict[7] = '甘铭乐'

index = [0, 1, 2, 3, 4, 5, 6, 7]

random.shuffle(index)

for i in index:
    print(user_dict[i])