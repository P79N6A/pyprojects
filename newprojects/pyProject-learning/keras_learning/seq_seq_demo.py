import json
import os
import numpy as np

min_count = 30
maxlen = 400
batch_size = 64
epochs = 100
char_size = 128
num_samples = 10000
latent_dim = 256


def load_data():
    file_path = "./datas/corpus.train.0.txt"
    contents_data = []
    with open(file_path, 'r') as f:
        contents = f.read()
        datas = contents.split("#END#")
        for data in datas:
            data = data.strip("\n")
            content_data = {}
            if "###" in data:
                content = data.split("###")
                content_data["content"] = content[0]
                content_data["title"] = content[1]
                contents_data.append(content_data)
    return contents_data


input_texts = []
target_texts = []
input_words = set()
target_words = set()

contents_data = load_data()
target_words.add("\t")
target_words.add("\n")
for content in contents_data[:min(num_samples,len(contents_data)-1)]:
    input_texts.append(content['content'])
    target_texts.append("\t" + " " + content['title'] + " " + "\n")
    for word in content['content'].split(" "):
        if word not in input_words:
            input_words.add(word)
    for word in content['title'].split(" "):
        if word not in target_words:
            target_words.add(word)

input_words = sorted(list(input_words))
target_words = sorted(list(target_words))
num_encoder_token = len(input_words)
num_decoder_token = len(target_words)
max_encoder_seq_length = max([len(txt) for txt in input_texts])
max_decoder_seq_length = max([len(txt) for txt in target_texts])


input_token_index = dict([(word,i) for i,word in enumerate(input_words)])
target_token_index = dict([(word,i) for i,word in enumerate(target_words)])

encoder_input_data = np.zeros((len(input_texts),max_encoder_seq_length,num_encoder_token), dtype='float32')
decoder_input_data = np.zeros((len(input_texts),max_decoder_seq_length,num_decoder_token), dtype='float32')
decoder_target_data = np.zeros((len(input_texts),max_decoder_seq_length,num_decoder_token),dtype='float32')

for i,(input_texts,target_texts) in enumerate(zip(input_texts,target_texts)):
    for t,word in enumerate(input_texts.split(" ")):
        encoder_input_data[i,t,input_token_index[word]] = 1.
    for t,word in enumerate(target_texts.split(" ")):
        decoder_input_data[i,t,target_token_index[word]] = 1.
        if t > 0:
            decoder_target_data[i,t-1,target_token_index[word]] = 1.

from keras.layers import Input,LSTM,Dense
from keras.models import Model
encoder_inputs = Input(shape=(None,num_encoder_token))
encoder = LSTM(latent_dim,return_state=True)
encoder_outputs,state_h,state_c = encoder(encoder_inputs)
print(encoder_outputs.shape)
print(state_h.shape)
print(state_c.shape)
encoder_states = [state_h,state_c]
print(len(encoder_states))

decoder_inputs = Input(shape=(None,num_decoder_token))
decoder_lstm = LSTM(latent_dim,return_sequences=True,return_state=True)
decoder_outputs,_,_ = decoder_lstm(decoder_inputs,initial_state = encoder_states)
decoder_dense = Dense(num_decoder_token,activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

model = Model([encoder_inputs,decoder_inputs],decoder_outputs)
model.compile(optimizer='rmsprop',loss='categorical_crossentropy')
print(model.summary())
model.fit([encoder_input_data,decoder_input_data],decoder_target_data,
          batch_size=batch_size,epochs=epochs,validation_split=0.2)
model.save("s2s.h5")


# Next: inference mode (sampling).
# Here's the drill:
# 1) encode input and retrieve initial decoder state
# 2) run one step of decoder with this initial state
# and a "start of sequence" token as target.
# Output will be the next target token
# 3) Repeat with the current target token and current states

# Define sampling models
encoder_model = Model(encoder_inputs, encoder_states)

decoder_state_input_h = Input(shape=(latent_dim,))
decoder_state_input_c = Input(shape=(latent_dim,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
decoder_outputs, state_h, state_c = decoder_lstm(
    decoder_inputs, initial_state=decoder_states_inputs)
decoder_states = [state_h, state_c]
decoder_outputs = decoder_dense(decoder_outputs)
decoder_model = Model(
    [decoder_inputs] + decoder_states_inputs,
    [decoder_outputs] + decoder_states)

# Reverse-lookup token index to decode sequences back to
# something readable.
reverse_input_char_index = dict(
    (i, char) for char, i in input_token_index.items())
reverse_target_char_index = dict(
    (i, char) for char, i in target_token_index.items())


def decode_sequence(input_seq):
    # Encode the input as state vectors.
    states_value = encoder_model.predict(input_seq)

    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1, num_decoder_token))
    # Populate the first character of target sequence with the start character.
    target_seq[0, 0, target_token_index['\t']] = 1.

    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict(
            [target_seq] + states_value)

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = reverse_target_char_index[sampled_token_index]
        decoded_sentence += sampled_char

        # Exit condition: either hit max length
        # or find stop character.
        if (sampled_char == '\n' or
           len(decoded_sentence) > max_decoder_seq_length):
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1, 1, num_decoder_token))
        target_seq[0, 0, sampled_token_index] = 1.

        # Update states
        states_value = [h, c]

    return decoded_sentence


for seq_index in range(100):
    # Take one sequence (part of the training set)
    # for trying out decoding.
    input_seq = encoder_input_data[seq_index: seq_index + 1]
    decoded_sentence = decode_sequence(input_seq)
    print('-')
    print('Input sentence:', input_texts[seq_index])
    print('Decoded sentence:', decoded_sentence)