#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 23 18:15:45 2017

@author: naanu
"""
#testing
from keras.layers.core import Activation, Dense, Dropout, SpatialDropout1D
from keras.layers import Merge
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras.layers.advanced_activations import PReLU
from keras.layers.normalization import BatchNormalization
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
import collections
import matplotlib.pyplot as plt
from keras.utils import plot_model
plt.switch_backend('agg')
import nltk
#nltk.download('punkt')
#nltk.download('stopwords')
from nltk.tokenize import word_tokenize
import numpy as np
import sys, Utility
from PreProcessData import cleanData

from keras.layers import Convolution1D, Flatten
from keras.callbacks import TensorBoard

from numpy import array
import time

#DATA_DIR = "/home/naanu/Music/dataForTest/imdb.txt"
#DATA_DIR = "/home/naanu/Music/dataForTest/umich-sentiment-train.txt"

DATA_DIR = "../dataForTest/newcheck.log"

MAX_FEATURES = 0#12953 #15487 #114825 #2000
MAX_SENTENCE_LENGTH = 0#558 #1259 #2818 #40

EMBEDDING_SIZE = 128
HIDDEN_LAYER_SIZE = 64
BATCH_SIZE = 32
NUM_EPOCHS = 4
vocab_size = 0
word2index = {}
index2word = {}
# Read training data and generate vocabulary
maxlen = 0
word_freqs = collections.Counter()
num_recs = 0
history = ""
model1 = ""
xxtest = ""
yytest = ""
y_val = ""
X_val = ""
toknizer = ""
X = np.empty((5, ), dtype=list)
TestList  = np.empty((5, ), dtype=list)
d = []

def main(): #idiomatic way: allows me to write code in the order I like
    readData()
    
count = 0    
def readData():
    
    global maxlen, num_recs,MAX_FEATURES, MAX_SENTENCE_LENGTH, count
    print("\nPart one:Cleaning Data")
    x = Utility.UtilityClass()       
    try:
        with open(DATA_DIR, 'r') as textFile:
            for line in textFile:
                line = line+". "
                count = count + 1
                label, sentence = line.strip().split("\t",1)
                sentence = sentence.replace('\t'," ")
                sentence = cleanData(x, \
                                    sentence,True,True,True,True,False, False)
                #decode the UTF-8-encoded String into a unicode string.
                words = word_tokenize(sentence)
                if len(words) > maxlen:
                    maxlen = len(words)
                for word in words:
                    word_freqs[word] += 1
                num_recs += 1
    #        ftrain.close()
            
            # Get some information about our corpus
            print("\nResult for Counting")
            print ("\tMax length: "+str(maxlen)) # 42  2818
            print ("\tWord frequency: "+str(len(word_freqs))) # 2313 114825
            MAX_FEATURES = len(word_freqs)
            MAX_SENTENCE_LENGTH = maxlen
            print(MAX_SENTENCE_LENGTH)
            
            del x
            createDict()
    except IOError, (errno, strerror):
        print "I/O error(%s): %s" % (errno, strerror)
    except KeyError, e:
        print 'I got a KeyError - reason "%s"' % str(e)
    except ValueError,eb:
        print str(count)+"\n"
        print ' here I am:  "%s"' % str(eb)
    except:
        print "Unexpected error:", sys.exc_info()[0]
        raise

        ######======================================######
   

def createDict():
    # 1 is UNK, 0 is PAD
    # We take MAX_FEATURES-1 featurs to accound for PAD
#    global vocab_size, index2word, word2index
#    vocab_size = min(MAX_FEATURES, len(word_freqs)) + 2
#    word2index = {x[0]: i+2
#                 for i, x in enumerate(word_freqs.most_common(MAX_FEATURES))}
#    
#    # using the same above word2index array
#    word2index["PAD"] = 0
#    word2index["UNK"] = 1
#    index2word = {v:k for k, v in word2index.items()}
    print("\nEnd Part # 2") 
    trainTokenizer()
    sentToSequence()
   

        ###================================================####
        
def trainTokenizer():
    print("\nBefore Part # 3: Training Tokenizer")
    global toknizer,d
    x = Utility.UtilityClass()
    toknizer = Tokenizer(num_words=MAX_FEATURES)
    d = [None]*num_recs
    i = 0
    with open(DATA_DIR, 'r') as textFile:
        for line in textFile:
            line = line+". "
            label, sentence = line.strip().split("\t",1)
            sentence = sentence.replace('\t'," ")
            sentence = cleanData(x, \
                                    sentence,True,True,True,True,False, False)
                #decode the UTF-8-encoded String into a unicode string.
#            words = word_tokenize(sentence)
            sentence = sentence.encode("utf-8")
            d [i] = sentence
            i += 1 
#        print(d)
#        print("fitting text!")
        toknizer.fit_on_texts(d)
#        print(toknizer.word_index)

        ###================================================####
        
        
def sentToSequence():
    print("\nStarting Part # 3")
    global model1, xxtest, yytest, history, toknizer, vocab_size, X,TestList
    x = Utility.UtilityClass()       
    
    X = np.empty((num_recs, ), dtype=list)
    TestList = np.empty((num_recs, ), dtype=list)
    y = np.zeros((num_recs, ))
    i = 0
    with open(DATA_DIR, 'r') as textFile:
        for line in textFile:
            line = line+". "
            label, sentence = line.strip().split("\t",1)
            sentence = sentence.replace('\t'," ")
            sentence = cleanData(x, \
                                    sentence,True,True,True,True,False, False)
                #decode the UTF-8-encoded String into a unicode string.
#            words = word_tokenize(sentence)
#            seqs = []
#            for word in words:
#                if word in word2index:   #if word2index.has_key(word):
#                    seqs.append(word2index[word])
#                else:
#                    seqs.append(word2index["UNK"])
#            
#            X[i] = seqs
#            if i < 1:
#                print("\nSeq Value: ")
#                print(seqs)
#                print("\nX Value: ")
#                print(X)
            seq = toknizer.texts_to_sequences([sentence.encode("utf-8")])
#            TestList[i] = sequence.pad_sequences(seq, maxlen=MAX_SENTENCE_LENGTH)
#            X[i] = sequence.pad_sequences(seq, maxlen=MAX_SENTENCE_LENGTH)
            X[i] = seq[0]
#            if i < 1:
#                print("\n Test Seq Value: ")
#                print(seq)
#                print("\n Test list Value: ")
#                print(TestList)
        
            y[i] = int(label)
            i += 1
#    print("delete Object X")
    del x
#    print("working with sequence")
#    print(X.shape)
#    print("Test list shape with sequence")
#    print(TestList.shape)
    vocab_size = min(MAX_FEATURES, len(word_freqs))
#    X = array(X)
#    print(X)
    # Pad the sequences (left padded with zeros)
    X = sequence.pad_sequences(X, maxlen=MAX_SENTENCE_LENGTH)
#    print("After padding ")
#    print(X)
#    print("\nAfter as array List\n ")
#    print(np.asarray(TestList))

    # Split input into training and test
    Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, 
                                                     random_state=42)
    # Split Xtrain into training and validation
    Xtrain, X_val, ytrain, y_val = train_test_split(Xtrain, ytrain, test_size=0.2, random_state=1)
    
    print(Xtrain.shape, Xtest.shape, ytrain.shape, ytest.shape)
    print("\nStarting Part # 4 (Model building)")
    # Build model
#    time.sleep(45)
    ''' Original One
    model = Sequential()
#    model.add(Dropout(0.2)
    model.add(Embedding(vocab_size, EMBEDDING_SIZE, input_length=MAX_SENTENCE_LENGTH))
    model.add(SpatialDropout1D(Dropout(0.2)))
    model.add(LSTM(HIDDEN_LAYER_SIZE, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(1))
    model.add(Activation("sigmoid"))
    '''
    model1 = Sequential()
    model1.add(Embedding(vocab_size, EMBEDDING_SIZE, input_length=MAX_SENTENCE_LENGTH))
    model1.add(SpatialDropout1D(Dropout(0.2)))
    model1.add(LSTM(HIDDEN_LAYER_SIZE, dropout=0.2, recurrent_dropout=0.2))
    model1.add(BatchNormalization())
    
    model1.add(Dense(64))
    model1.add(PReLU())
    model1.add(Dropout(0.4))
    model1.add(BatchNormalization())

    model1.add(Dense(32))
    model1.add(PReLU())
    model1.add(Dropout(0.4))
    model1.add(BatchNormalization())

    model1.add(Dense(1))
    model1.add(Activation("sigmoid"))

    model1.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    history = model1.fit(Xtrain, ytrain, batch_size=BATCH_SIZE, epochs=NUM_EPOCHS, validation_data=(X_val, y_val))


# Convolutional model (3x conv, flatten, 2x dense)
#    model = Sequential()
#    model.add(Embedding(vocab_size, EMBEDDING_SIZE, input_length=MAX_SENTENCE_LENGTH))
#    model.add(Convolution1D(64, 3, border_mode='same'))
#    model.add(Convolution1D(32, 3, border_mode='same'))
#    model.add(Convolution1D(16, 3, border_mode='same'))
#    model.add(Flatten())
#    model.add(Dropout(0.2))
#    model.add(Dense(180,activation='sigmoid'))
#    model.add(Dropout(0.2))
#    model.add(Dense(1,activation='sigmoid'))
#    
#    # Log to tensorboard
#    tensorBoardCallback = TensorBoard(log_dir='./logs', write_graph=True)
#    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#    model.fit(Xtrain, ytrain, nb_epoch=4, callbacks=[tensorBoardCallback], batch_size=32, validation_data=(X_val, y_val))



    xxtest = Xtest
    yytest = ytest
    plotAndEvaluate()



def plotAndEvaluate():
    print("Starting Part # 5")
    global history, xxtest, yytest,X_val,y_val
    '''
    # plot loss and accuracy
    f = plt.figure()
    plt.subplot(211)
    plt.title("Accuracy")
    plt.plot(history.history["acc"], color="g", label="Train")
    plt.plot(history.history["val_acc"], color="b", label="Validation")
    plt.legend(loc="best")
     
    plt.subplot(212)
    plt.title("Loss")
    plt.plot(history.history["loss"], color="g", label="Train")
    plt.plot(history.history["val_loss"], color="b", label="Validation")
    plt.legend(loc="best")
     
    plt.tight_layout()
    #plt.show()
    f.savefig("result_1.pdf", bbox_inches='tight')
    '''
    # summarize history for accuracy
    f = plt.figure()
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    #plt.show()
    f.savefig("acc_1.pdf", bbox_inches='tight')

    # summarize history for loss
    f1 = plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    #plt.show()
    f1.savefig("loss_1.pdf", bbox_inches='tight')

    #evaluate
    score, acc = model1.evaluate(xxtest, yytest, batch_size=BATCH_SIZE)
    print("Test score: %.3f, accuracy: %.3f" % (score, acc))

    #saving model shape
    plot_model(model1, to_file='model.png', show_shapes=True)
    test()

def test():
    print("Starting Part # 6") 
    
    for i in range(10):
        idx = np.random.randint(len(xxtest))
        xtest = xxtest[idx].reshape(1,MAX_SENTENCE_LENGTH) #40 #1259
        ylabel = yytest[idx]
        ypred = model1.predict(xtest)[0][0]
        #sent = " ".join([index2word[x] for x in xtest[0].tolist() if x != 0])
        print("%.0f\t%d" % (ypred, ylabel)) #, sent

if __name__ == '__main__':
    main()


        ####===============================================####
