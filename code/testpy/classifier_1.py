from __future__ import print_function
import numpy as np
from numpy import distutils

from keras.preprocessing import sequence
from keras.losses import categorical_crossentropy
from keras.models import model_from_json
import glob
import os
import re
from collections import OrderedDict
from nltk.corpus import stopwords
import _pickle as pkl
from keras import optimizers , utils
from gensim.models.word2vec import Word2Vec
from gensim.corpora.dictionary import Dictionary
from keras.models import Model
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, LSTM, Input, merge,Conv1D,MaxPooling1D , Activation
from nltk.corpus import stopwords
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

max_features = 5000
maxlen = 300

testPath = '../new_review100000.txt'

    #test = []
tt = open(testPath, 'r')
test = tt.readlines()


tokenizer=Tokenizer(nb_words=max_features)
tokenizer.fit_on_texts(str(test))
sequence=tokenizer.texts_to_sequences(test)
word_index=tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

X_test = pad_sequences(sequence, maxlen=maxlen)




    # print('Loading the W2V model...')
    # w2c_model = Word2Vec.load("Word2Vec_300.model")
    # #model = gensim.models.Word2Vec.load_word2vec_format('./GoogleNews-vectors-negative300.bin', binary=True)

    # print('Transform the Data...')
    # index_dict, word_vectors,test = create_dictionaries(test = test, model = w2c_model)

    # X_test = test.values()
    # X_test = sequence.pad_sequences(X_test, maxlen=300)

    #X_test = w2c_model.wv(X_test)

    # load json and create model
json_file = open('model_2.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
    # load weights into new model
model.load_weights("model_2.h5")
print("Loaded model from disk")

classes = model.predict_classes(X_test)
print(classes)

y_test = [1] * 20000 + [2] * 20000 + [3] * 20000 + [4] * 20000 + [5] * 20000

    # classes = model.predict_classes(X_test)
classestoArray = np.array(classes)
y_testtoArray = np.array(y_test)
acc = np.mean( classestoArray == y_testtoArray )
print ('Test accuracy:', acc)
    # acc = np.distutils.accuracy(classes, y_test)
    # print ('Test accuracy:', acc)

    # evaluate the model
    # scores = model.evaluate(X_test, y_test, verbose=0)
    # print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    # cvscores = []
    # cvscores.append(scores[1] * 100)
    # print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))

    #return classes
    #
    # txt_file = open('test_result.txt', 'w')
    # txt_file.write("Label: %s" % classes)
    # txt_file.close()


