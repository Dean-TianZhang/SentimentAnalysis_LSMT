from __future__ import print_function
import multiprocessing
import numpy as np
from numpy import distutils
# np.random.seed(1337)  # for reproducibility

from keras.preprocessing import sequence
import _pickle as pkl
from keras import optimizers , utils
from gensim.models.word2vec import Word2Vec
from gensim.corpora.dictionary import Dictionary
from keras.losses import categorical_crossentropy
import gensim
from keras.models import Model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, LSTM, Input, merge,Conv1D,MaxPooling1D , Activation
from nltk.corpus import stopwords
import nltk
import glob
import os
import re
import pandas as pd
from numpy import distutils
from collections import OrderedDict
from nltk.corpus import stopwords

# complimentary
# def genseq(text):
#     words=nltk.word_tokenize(text)
#     wes=np.array([0.0]*300)
#     for word in words:
#         if word in word_model:
#             wes = wes + np.array(word_model[word])
#         else:
#             wes = wes + np.array(word_model['UNKNOWN'])
#             # continue
#     wes=wes / float(len(words))
#     return wes
#
# def gendata(f,texts):
#     text_preprocess = np.array([f(text) for text in texts])
#     return text_preprocess

vocab_dim = 300
n_iterations = 10  # ideally more, since this improves the quality of the word vecs
n_exposures = 30
window_size = 7
input_length = 100
max_features = 5000
maxlen = 300  # cut texts after this number of words (among top max_features most  common words)
batch_size = 32
num_classes = None
cpu_count = multiprocessing.cpu_count()

# word_model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin',binary=True)

testPath = './new_review100000.txt'
trainPath = './review100000.txt'

test = []
train = []

tt = open(testPath, 'r')
tn = open(trainPath, 'r')
test = tt.readlines()
train = tn.readlines()

tokenizer=Tokenizer(nb_words=max_features)
tokenizer.fit_on_texts(train)
sequence=tokenizer.texts_to_sequences(train)
word_index=tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

X_train = pad_sequences(sequence, maxlen=maxlen)

# X_test = gendata(genseq, test)
# X_train = gendata(genseq, train)

y_test = [[1]] * 20000 + [[1]] * 20000 + [[2]] * 20000 + [[3]] * 20000 + [[4]] * 20000
y_train = [[1]] * 20000 + [[1]] * 20000 + [[2]] * 20000 + [[3]] * 20000 + [[4]] * 20000

# print("Pad sequences (samples x time)")
# X_train = sequence.pad_sequences(X_train, maxlen=maxlen)
# X_test = sequence.pad_sequences(X_test, maxlen=maxlen)
# print('X_train shape:', X_train.shape)
# print('X_test shape:', X_test.shape)
y_train = np.array(y_train)
y_test = np.array(y_test)


# convert class vectors to binary class matrices
y_train = utils.to_categorical(y_train, num_classes)
y_test = utils.to_categorical(y_test, num_classes)
print('y_train shape:', y_train.shape)
print('y_test shape:', y_test.shape)

embeddings_index = {}
f = open('./glove.6B.300d.txt',encoding='utf8')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()
print('Found %s word vectors.' % len(embeddings_index))

embedding_matrix = np.zeros((len(word_index) + 1, 300))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector

# embedding_layer = Embedding(input_dim=weight.shape[0], output_dim=weight.shape[1], weights=[weight])

model = Sequential()
model.add(Embedding(len(word_index) + 1,
                    300,
                    weights=[embedding_matrix],
                    #input_length=5000,
                    trainable=False))
# model.add(embedding_layer)
model.add(Conv1D(filters=100, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.2))
model.add(LSTM(100))
# model.add(LSTM(word_model.syn0.shape[1]))
model.add(Dense(5, activation='softmax'))
model.summary()

sgd = optimizers.SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=False)

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

print('Train...')
model.fit(X_train, y_train,
          batch_size=batch_size,
          nb_epoch=20#validation_data=[X_test, y_test]
          )

# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")
