# for smaller datasets please use the simpler model sarcasm_detection_model_CNN_LSTM_DNN_simpler.py

import os
import sys

sys.path.append('../')

import collections
import time
import numpy
import pandas as pd
numpy.random.seed(1337)
from sklearn import metrics
from keras.models import Model
from keras.layers import Input
from keras.models import Sequential, model_from_json
from keras.layers.core import Dropout, Dense, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from keras.utils import np_utils
from collections import defaultdict
import CNN_LSTM_DNN.data_processing.data_handler as dh
from sklearn.model_selection import train_test_split


class sarcasm_model():
    _train_file = None
    _test_file = None
    _tweet_file = None
    _output_file = None
    _model_file_path = None
    _word_file_path = None
    _split_word_file_path = None
    _emoji_file_path = None
    _vocab_file_path = None
    _input_weight_file_path = None
    _vocab = None
    _line_maxlen = None

    def __init__(self):
        self._line_maxlen = 30

    def _build_network(self, vocab_size, maxlen, emb_weights=[], embedding_dimension=256, hidden_units=256):
        print('Build model...')

        text_input = Input(name='text', shape=(maxlen,))

        if (len(emb_weights) == 0):
            emb = Embedding(vocab_size, embedding_dimension, input_length=maxlen,
                            embeddings_initializer='glorot_normal',
                            trainable=True)(text_input)
        else:
            emb = Embedding(vocab_size, emb_weights.shape[1], input_length=maxlen, weights=[emb_weights],
                            trainable=False)(text_input)

        cnn1 = Convolution1D(int(hidden_units / 4), 3, kernel_initializer='he_normal', activation='sigmoid',
                             padding='valid', input_shape=(1, maxlen))(emb)

        cnn2 = Convolution1D(int(hidden_units / 2), 3, kernel_initializer='he_normal', activation='sigmoid',
                             padding='valid', input_shape=(1, maxlen - 1))(cnn1)

        lstm1 = LSTM(hidden_units, kernel_initializer='he_normal', activation='sigmoid',
                     dropout=0, return_sequences=True)(cnn2)

        lstm2 = LSTM(hidden_units, kernel_initializer='he_normal', activation='sigmoid',
                     dropout=0)(lstm1)

        dnn_1 = Dense(hidden_units, kernel_initializer="he_normal", activation='sigmoid')(lstm2)
        dnn_2 = Dense(2, activation='softmax')(dnn_1)

        model = Model(inputs=[text_input], outputs=dnn_2)

        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        print('No of parameter:', model.count_params())

        print(model.summary())
        return model


class train_model(sarcasm_model):
    train = None
    validation = None
    print("Loading resource...")

    def __init__(self, train_file, validation_file, word_file_path, split_word_path, emoji_file_path, model_file,
                 vocab_file,
                 output_file,
                 word2vec_path=None):
        sarcasm_model.__init__(self)

        self._train_file = train_file
        self._validation_file = validation_file
        self._word_file_path = word_file_path
        self._split_word_file_path = split_word_path
        self._emoji_file_path = emoji_file_path
        self._model_file = model_file
        self._vocab_file_path = vocab_file
        self._output_file = output_file

        self.load_train_validation_data()
        
        self.trained_model = None
        self._X = None
        
        print(self._line_maxlen)

        # build vocabulary
        # truncates words with min freq=1
        self._vocab = dh.build_vocab(self.train, min_freq=1)
        if ('unk' not in self._vocab):
            self._vocab['unk'] = len(self._vocab.keys()) + 1

        print(len(self._vocab.keys()) + 1)
        print('unk::', self._vocab['unk'])

        dh.write_vocab(self._vocab_file_path, self._vocab)

        # prepares input
        X, Y, D, C, A = dh.vectorize_word_dimension(self.train, self._vocab)
        X = dh.pad_sequence_1d(X, maxlen=self._line_maxlen)
        self._X = X
        _, tX, __, tY = train_test_split(X, Y, test_size = 0.1, random_state = 0)
        
        # prepares input
        #tX, tY, tD, tC, tA = dh.vectorize_word_dimension(self.validation, self._vocab)
        #tX = dh.pad_sequence_1d(tX, maxlen=self._line_maxlen)

        # embedding dimension
        dimension_size = 300

        W = dh.get_word2vec_weight(self._vocab, n=dimension_size,
                                   path=word2vec_path)

        # solving class imbalance
        ratio = self.calculate_label_ratio(Y)
        ratio = [max(ratio.values()) / value for key, value in ratio.items()]
        print('class ratio::', ratio)

        Y, tY = [np_utils.to_categorical(x) for x in (Y, tY)]

        print('train_X', X.shape)
        print('train_Y', Y.shape)
        #print('validation_X', tX.shape)
        #print('validation_Y', tY.shape)

        model = self._build_network(len(self._vocab.keys()) + 1, self._line_maxlen, hidden_units=256, emb_weights=W)

        open(self._model_file + 'model.json', 'w').write(model.to_json())
        save_best = ModelCheckpoint(model_file + 'model.json.hdf5', save_best_only=True)
        save_all = ModelCheckpoint(self._model_file + 'weights.{epoch:02d}.hdf5',
                                   save_best_only=False)
        early_stopping = EarlyStopping(monitor='val_loss', patience=20, verbose=1)

        # training
        model.fit(X, Y, batch_size=64, epochs=1, validation_data=(tX, tY), shuffle=True,
                  callbacks=[save_best, save_all, early_stopping], class_weight=ratio, verbose=2)
        
        self.trained_model = model
        
        

    def load_train_validation_data(self):
        self.train = dh.loaddata(self._train_file, self._word_file_path, self._split_word_file_path,
                                 self._emoji_file_path, normalize_text=True,
                                 split_hashtag=True,
                                 ignore_profiles=False)
        print('Training data loading finished...')

        self.validation = dh.loaddata(self._train_file, self._word_file_path, self._split_word_file_path,
                                      self._emoji_file_path,
                                      normalize_text=True,
                                      split_hashtag=True,
                                      ignore_profiles=False)
        print('Validation data loading finished...')
        '''
        if (self._test_file != None):
            self.test = dh.loaddata(self._test_file, self._word_file_path, normalize_text=True,
                                    split_hashtag=True,
                                    ignore_profiles=True)
        '''
    def get_maxlen(self):
        return max(map(len, (x for _, x in self.train + self.validation)))

    def write_vocab(self):
        with open(self._vocab_file_path, 'w') as fw:
            for key, value in self._vocab.iteritems():
                fw.write(str(key) + '\t' + str(value) + '\n')

    def calculate_label_ratio(self, labels):
        return collections.Counter(labels)

if __name__ == "__main__":
    basepath = os.getcwd()[:os.getcwd().rfind('\\')]
    train_file = basepath + '/data/train.jsonl'
    #validation_file = basepath + '/resource/dev/Dev_v1.txt'
    test_file = basepath + '/data/test.jsonl'
    word_file_path = basepath + '/resource/word_list_freq.txt'
    split_word_path = basepath + '/resource/word_split.txt'
    emoji_file_path = basepath + '/resource/emoji_unicode_names_final.txt'

    output_file = basepath + '/resource/text_model/TestResults.txt'
    model_file = basepath + '/resource/text_model/weights/'
    vocab_file_path = basepath + '/resource/text_model/vocab_list.txt'

    word2vec_path = model_file + 'GoogleNews-vectors-negative300.bin'

    # uncomment for training
    tr = train_model(train_file=train_file, validation_file=train_file, word_file_path=word_file_path,
                     split_word_path=split_word_path, emoji_file_path=emoji_file_path, model_file=model_file,
                     vocab_file=vocab_file_path, output_file=output_file, word2vec_path=word2vec_path)
    
    model = tr.trained_model

    vocab = tr._vocab
    line_maxlen = tr._line_maxlen
 
#%%    
    test = dh.loaddata(test_file, word_file_path, split_word_path,
                                 emoji_file_path, normalize_text=True,
                                 split_hashtag=True,
                                 ignore_profiles=False)
 
    X_test, _, D_test, C_test, A_test = dh.vectorize_word_dimension(test, vocab)
    X_test = dh.pad_sequence_1d(X_test, maxlen=line_maxlen)
    

    y_predict_proba = model.predict(X_test, batch_size = 1, verbose = 1)
    y_predict = numpy.argmax(y_predict_proba, axis = 1)
    #y_predict = LR.predict(X_test)

    test_raw = pd.read_json(test_file, orient = 'columns', lines = True)
    test_ID = test_raw[['id']]
    test_ID['prediction'] = y_predict
    test_ID['prediction'].replace(1, "SARCASM", inplace=True)
    test_ID['prediction'].replace(0, 'NOT_SARCASM', inplace = True)
    #test_ID.to_csv(path_or_buf = basepath + '\\answer.txt', index = False, header = False)
