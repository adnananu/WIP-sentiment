#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  5 11:52:44 2017

@author: naanu
"""

import Utility
from PreProcessData import cleanData
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np

#A = np.empty((10, ), dtype=list)
#print(A)
tokenizer = Tokenizer()
texts = ["da vince code book awesome","The sun is shining in June!","September is grey.","Life is beautiful in August."]
textss = ["da vince code book awesome"]
tokenizer.fit_on_texts(textss)
#print(tokenizer.word_index)
test = "da vince code book awesome"
c = [test]
print(c)
A = tokenizer.texts_to_sequences([test])
#i = 0
#for text in texts:
#    A[i] = tokenizer.texts_to_sequences(text)
#    i+=1


print("before")
print(A)
A = pad_sequences(A, maxlen=10)
print("After")
print(A)



cc = ["da vince code book awesome","The sun is shining in June!"]
d = list()

for w in cc:
        d = [w]
print d











def trainTokenizer():
    DATA_DIR = "/home/naanu/Music/dataForTest/testData.txt"
    print("\nBefore Part # 3: Training Tokenizer")
    global toknizer
    x = Utility.UtilityClass()
    toknizer = Tokenizer(num_words=12953)
    d = []
    
    with open(DATA_DIR, 'r') as textFile:
        for line in textFile:
            label, sentence = line.strip().split("\t",1)
            sentence = sentence.replace('\t'," ")
            sentence = cleanData(x, \
                                    sentence,True,True,True,True,True,True)
                #decode the UTF-8-encoded String into a unicode string.
#            words = word_tokenize(sentence)
            sentence = sentence.encode("utf-8")
            print(sentence)
            d.extend(sentence)
            break
        print(d)
        print("fitting text!")
        toknizer.fit_on_texts(d)
        print(toknizer.word_index)
        
#trainTokenizer()