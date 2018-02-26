from __future__ import print_function
import multiprocessing
import numpy as np
from numpy import distutils
np.random.seed(1337)  # for reproducibility

from keras.preprocessing import sequence
from keras import optimizers , utils
from gensim.models.word2vec import Word2Vec
from gensim.corpora.dictionary import Dictionary
from keras.losses import categorical_crossentropy
from keras.models import Model
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, LSTM, Input, merge,Conv1D,MaxPooling1D , Activation
import imdbReview
import imdb


max_features = 5000
maxlen = 100  # cut texts after this number of words (among top max_features most common words)
batch_size = 32
num_classes = 5


print('Loading data...')
(X_train, y_train), (X_test, y_test) = imdbReview.load_data() #nb_words=max_features
print(len(X_train), 'train sequences')
print(len(X_test), 'test sequences')


print("Pad sequences (samples x time)")
X_train = sequence.pad_sequences(X_train, maxlen=maxlen)
X_test = sequence.pad_sequences(X_test, maxlen=maxlen)
print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)
y_train = np.array(y_train)
y_test = np.array(y_test)

# convert class vectors to binary class matrices
y_train = utils.to_categorical(y_train, num_classes)
y_test = utils.to_categorical(y_test, num_classes)
print('y_train shape:', y_train.shape)
print('y_test shape:', y_test.shape)


# this is the placeholder tensor for the input sequences
# sequence = Input(shape=(maxlen,), dtype='float32')
# this embedding layer will transform the sequences of integers
# into vectors of size 128
# embedded = Embedding(max_features,100, input_length=maxlen,mask_zero=True)(sequence)

# # apply forwards LSTM
# forwards = LSTM(5)(embedded)
# # apply backwards LSTM
# backwards = LSTM(5, go_backwards=True)(embedded)
#
# # concatenate the outputs of the 2 LSTMs
# merged = merge([forwards, backwards], mode='concat', concat_axis=-1)
# after_dp = Dropout(0.5)(merged)
# output = Dense(1, activation='sigmoid')(after_dp)
#
# model = Model(input=sequence, output=output)
model = Sequential()
model.add(Embedding(output_dim = maxlen,
                    input_dim = len(X_train) + 1,
                    mask_zero = True,
                    input_length = 100))
# model.add(Conv1D(filters=100, kernel_size=3, padding='same', activation='relu'))
# model.add(MaxPooling1D(pool_size=2))
# model.add(Dense(5, activation='sigmoid'))
model.add(LSTM(100))
model.add(Dropout(0.3))
model.add(Dense(5, activation='softmax'))
model.summary()
# model.add(Dropout(0.1))


sgd = optimizers.SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=False)

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])


# try using different optimizers and different optimizer configs
# model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])

print('Train...')
model.fit(X_train, y_train,
          batch_size=batch_size,
          nb_epoch=3,
          #validation_data=[X_test, y_test]
          )

# classes = model.predict_classes(X_train[1])
# print(classes)


# model.save_weights('first_try.h5')

# classes = model.predict_classes(X_test)
# acc = np.distutils.accuracy(classes, y_test)
# print ('Test accuracy:', acc)

# validation_data=[X_test, y_test]

# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")

# # later...
#
# load json and create model
# json_file = open('model.json', 'r')
# loaded_model_json = json_file.read()
# json_file.close()
# loaded_model = model_from_json(loaded_model_json)
# # load weights into new model
# loaded_model.load_weights("model.h5")
# print("Loaded model from disk")
