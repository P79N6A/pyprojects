from keras.models import Model
from keras.layers import Input,LSTM,Dense
import numpy as np

batch_size = 64
eposh = 100
latent_dim = 256
num_samples = 10000
data_path = 'datas/corpus.train.0.txt'

input_texts = []
target_texts = []
with open(data_path,'r') as f:
    lines = f.read().split("#END#")
for line in lines[: min(num_samples),len(lines)-1]:
    line = line.strip("\n")
    input_text, target_text = line.split("###")
    input_texts.append(input_text)
    target_text = "\t" + target_text + "\n"
    target_texts.append(target_text)

num_encoder_token = len()





