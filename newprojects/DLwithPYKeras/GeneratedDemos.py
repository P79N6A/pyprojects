# lstm generate txt
import keras
import numpy as np

path = keras.utils.get_file('nietzsche.txt', origin='https://s3.amazonaws.com/text-datasets/nietzsche.txt')
text = open(path).read().lower()
print('corpus length:',len(text))

maxlen = 60
step = 3
sentences = []
next_char = []

for i in range(0,len(text)-maxlen,step):
    sentences.append(text[i:i+maxlen])
    next_char.append(text[i+maxlen])

chars = sorted(list(set(text)))
