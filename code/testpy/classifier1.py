from __future__ import print_function
import numpy as np
from numpy import distutils
from keras.models import model_from_json
from keras.preprocessing import sequence

from gensim.models.word2vec import Word2Vec
from gensim.corpora.dictionary import Dictionary


# data_locations = {'./ratingReview/star_1.txt': 'TEST_1star',
#                   './ratingReview/star_2.txt': 'TEST_2star',
#                   './ratingReview/star_3.txt': 'TEST_3star',
#                   './ratingReview/star_4.txt': 'TEST_4star',
#                   './ratingReview/star_5.txt': 'TEST_5star',
#                   './new_ratingReview/star_1.txt': 'TRAIN_1star',
#                   './new_ratingReview/star_2.txt': 'TRAIN_2star',
#                   './new_ratingReview/star_3.txt': 'TRAIN_3star',
#                   './new_ratingReview/star_4.txt': 'TRAIN_4star',
#                   './new_ratingReview/star_5.txt': 'TRAIN_5star'}
#
# def import_tag(datasets = None):
#     if datasets is not None:
#         train = {}
#         test = {}
#         for k, v in datasets.items():
#             with open(k) as fpath:
#                 data = fpath.readlines()
#             for val, each_line in enumerate(data):
#                 if v.endswith("1star") and v.startswith("TRAIN"):
#                     train[val] = each_line
#                 elif v.endswith("2star") and v.startswith("TRAIN"):
#                     train[val + 20000] = each_line
#                 elif v.endswith("3star") and v.startswith("TRAIN"):
#                     train[val + 40000] = each_line
#                 elif v.endswith("4star") and v.startswith("TRAIN"):
#                     train[val + 60000] = each_line
#                 elif v.endswith("5star") and v.startswith("TRAIN"):
#                     train[val + 80000] = each_line
#                 elif v.endswith("1star") and v.startswith("TEST"):
#                     test[val] = each_line
#                 elif v.endswith("2star") and v.startswith("TEST"):
#                     test[val + 20000] = each_line
#                 elif v.endswith("3star") and v.startswith("TEST"):
#                     test[val + 40000] = each_line
#                 elif v.endswith("4star") and v.startswith("TEST"):
#                     test[val + 60000] = each_line
#                 else:
#                     test[val + 80000] = each_line
#         return train, test
#     else:
#         print('Data not found...')
#
#
# def tokenizer(text):
#     ''' Simple Parser converting each document to lower-case, then
#         removing the breaks for new lines and finally splitting on the
#         whitespace
#     '''
#     text = [document.lower().replace('\n', '').split() for document in text]
#     return text
#
#
# vocab_dim = 100
# n_iterations = 10  # ideally more, since this improves the quality of the word vecs
# n_exposures = 30
# window_size = 7
# input_length = 100
# max_features = 5000
# maxlen = 100  # cut texts after this number of words (among top max_features most common words)
# batch_size = 32
# num_classes = None
#
# cpu_count = multiprocessing.cpu_count()
#
# def create_dictionaries(train = None,
#                         test = None,
#                         model = None):
#     ''' Function does are number of Jobs:
#         1- Creates a word to index mapping
#         2- Creates a word to vector mapping
#         3- Transforms the Training and Testing Dictionaries
#     '''
#     if (train is not None) and (model is not None) and (test is not None):
#         gensim_dict = Dictionary()
#         gensim_dict.doc2bow(model.wv.vocab.keys(),
#                             allow_update=True)
#         w2indx = {v: k+1 for k, v in gensim_dict.items()}
#         w2vec = {word: model[word] for word in w2indx.keys()}
#
#         def parse_dataset(data ):
#             ''' Words become integers
#             '''
#             for key in data.keys():
#                 txt = data[key].lower().replace('\n', '').split()
#                 new_txt = []
#                 for word in txt:
#                     try:
#                         new_txt.append(w2indx[word])
#                     except:
#                         new_txt.append(0)
#                 data[key] = new_txt
#             return data
#         train = parse_dataset(train )
#         test = parse_dataset(test )
#         return w2indx, w2vec, train, test
#     else:
#         print('No data provided...')
#
#
# print('Loading Data...')
# train, test = import_tag(datasets = data_locations)
# combined = list(train.values()) + list(test.values())
#
# print('Tokenising...')
# combined = tokenizer(combined)
#
# print('Training a Word2vec model...')
# model = Word2Vec(size = vocab_dim,
#                  min_count = n_exposures,
#                  window = window_size,
#                  workers = cpu_count,
#                  iter = n_iterations)
# model.build_vocab(combined)
# model.train(combined,total_examples=model.corpus_count, epochs=model.iter)
#
# # model_json1 = model.to_json()
# # with open("model_w2c.json", "w") as json_file:
# #     json_file.write(model_json1)
# # # serialize weights to HDF5
# # model.save_weights("model_w2c.h5")
# # print("Saved w2c model to disk")
#
# print('Transform the Data...')
# index_dict, word_vectors, train, test = create_dictionaries(train = train,
#                                                             test = test,
#                                                             model = model)
#
# print('Setting up Arrays for Keras Embedding Layer...')
# n_symbols = len(index_dict) + 1  # adding 1 to account for 0th index
# embedding_weights = np.zeros((n_symbols, vocab_dim))
# for word, index in index_dict.items():
#     embedding_weights[index, :] = word_vectors[word]
#
# print('Creating Datesets...')
# X_train = train.values()
# # y_train = [1 if value > 12500 else 0 for value in train.keys()]
# # path = './rating.txt'
# # with open(path) as fpath:
# #   label = fpath.readlines()
# # label = list(map(int, label))
# # df= pd.Series(list('12345'))
# # df2 = pd.get_dummies(df)
# # list1 = [list(df2.iloc[0])] * 20000 + [list(df2.iloc[1])] * 20000 + [list(df2.iloc[2])] * 20000 + [list(df2.iloc[3])] * 20000 + [list(df2.iloc[4])] * 20000
# # y_train = list1
# y_train = [[0]] * 20000 + [[1]] * 20000 + [[2]] * 20000 + [[3]] * 20000 + [[4]] * 20000
# X_test = test.values()
# # y_test = [1 if value > 12500 else 0 for value in test.keys()]
# y_test = [[0]] * 20000 + [[1]] * 20000 + [[2]] * 20000 + [[3]] * 20000 + [[4]] * 20000
#
#
# print("Pad sequences (samples x time)")
# X_train = sequence.pad_sequences(X_train, maxlen=maxlen)
# X_test = sequence.pad_sequences(X_test, maxlen=maxlen)
# print('X_train shape:', X_train.shape)
# print('X_test shape:', X_test.shape)
# y_train = np.array(y_train)
# y_test = np.array(y_test)
#
#
# # convert class vectors to binary class matrices
# y_train = utils.to_categorical(y_train, num_classes)
# y_test = utils.to_categorical(y_test, num_classes)
# print('y_train shape:', y_train.shape)
# print('y_test shape:', y_test.shape)


def tokenizer(text):
    text = [document.lower().replace('\n', '').split() for document in text]
    return text

def create_dictionaries(test = None, model = None):
    ''' Function does are number of Jobs:
        1- Creates a word to index mapping
        2- Creates a word to vector mapping
        3- Transforms the Training and Testing Dictionaries
    '''
    if (model is not None) and (test is not None):
        gensim_dict = Dictionary()
        gensim_dict.doc2bow(model.wv.vocab.keys(), allow_update=True)
        w2indx = {v: k+1 for k, v in gensim_dict.items()}
        w2vec = {word: model[word] for word in w2indx.keys()}

        def parse_dataset(data):
            ''' Words become integers
            '''
            for key in data.keys():
                txt = data[key].lower().replace('\n', '').split()
                new_txt = []
                for word in txt:
                    try:
                        new_txt.append(w2indx[word])
                    except:
                        new_txt.append(0)
                data[key] = new_txt
            return data
        
        test = parse_dataset(test)
        return w2indx, w2vec, test
    else:
        print('No data provided...')

# tdpath = './test_case/test_case.txt'
# test = {}
# with open(tdpath) as fpath:
#     data = fpath.readlines()
# for val, each_line in enumerate(data):
#     test[val] = each_line
#     print(each_line);

# X_test = list(test.values())
# X_test = tokenizer(X_test)

def run(test):
    print('Loading the W2V model...')
    w2c_model = Word2Vec.load("Word2Vec.model")
    #model = gensim.models.Word2Vec.load_word2vec_format('./GoogleNews-vectors-negative300.bin', binary=True)

    print('Transform the Data...')
    index_dict, word_vectors,test = create_dictionaries(test = test, model = w2c_model)

    X_test = test.values()
    X_test = sequence.pad_sequences(X_test, maxlen=300)

    #X_test = w2c_model.wv(X_test)

    # load json and create model
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    # load weights into new model
    model.load_weights("model.h5")
    print("Loaded model from disk")

    classes = model.predict_classes(X_test)
    print(classes)
    return  classes

    y_test = [0] * 20000

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


    #
    # txt_file = open('test_result.txt', 'w')
    # txt_file.write("Label: %s" % classes)
    # txt_file.close()


