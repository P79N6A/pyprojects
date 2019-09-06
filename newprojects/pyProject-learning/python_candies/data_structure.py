import numpy as np

data = {i:np.random.randn() for i in range(7)}
data = {i:np.random.randn() for i in range(7)}

data = [i for i in range(7)]

data = [np.random.randn() for i in range(7)]


tup = 4,5,6
nested_tup = (1,2,3),(4,5)
print(nested_tup)
((1, 2, 3), (4, 5))
print(nested_tup.count(3))
print(nested_tup.count((4,5)))

a_list = [2,3,4,None]

for i,value in enumerate(a_list):
    print(i)
    print(value)
    print("---------")

print(sorted(a_list))
# [None, 2, 3, 4]
seq1 = ["i","am","a","happy","dog"]
seq2 = ["you","are","a","cat"]
print(zip(seq1,seq2))
# [('i', 'you'), ('am', 'are'), ('a', 'a'), ('happy', 'cat')]
print(list(zip(seq1,seq2)))
# [('i', 'you'), ('am', 'are'), ('a', 'a'), ('happy', 'cat')]

for i,(a,b) in enumerate(zip(seq1,seq2)):
    print(i)
    print((a,b))
    print("-------------")

zipped = zip(seq1,seq2)
a,b = zip(*zipped)
print(a)
('i', 'am', 'a', 'happy')
print(b)
('you', 'are', 'a', 'cat')
print(reversed(a_list))
# <listreverseiterator object at 0x105b57810>
print(list(reversed(a_list)))
# [None, 4, 3, 2]

dict_1 = {}
dict_1["a"] = [2,3,4,None]
dict_1[4] = ("happy","dog")
print(dict_1)
# {'a': [2, 3, 4, None], 4: ('happy', 'dog')}
print(3 in dict_1)
# False
print("a" in dict_1)
# True
del dict_1["a"]
print(dict_1)
# {4: ('happy', 'dog')}

key_list = ["key1","key2","key8"]
value_list = ["value1","value2","value3"]
mapping = dict(zip(key_list,value_list))
print(mapping)
# {'key2': 'value2', 'key1': 'value1', 'key8': 'value3'}
value_list.append("happy")
mapping = dict(zip(key_list,value_list))
print(mapping)
# {'key2': 'value2', 'key1': 'value1', 'key8': 'value3'}
value =mapping.get("key1","happy")
print(value)
# value1
value =mapping.get("key0","happy")
print(value)
# happy


hash(["abc"])
# Traceback (most recent call last):
#   File "<stdin>", line 1, in <module>
# TypeError: unhashable type: 'list'
hash("abc")
# 1453079729188098211
hash(("abc"))
# 1453079729188098211
print({2,3,4,4,2,2})
# set([2, 3, 4])
strings = ["abc","cde"]
print([value.upper() for value in strings if len(value)>=2])
# ['ABC', 'CDE']
strings = ["abc","cde","a","gk"]
print([value.upper() for value in strings if len(value)>=2])
# ['ABC', 'CDE', 'GK']