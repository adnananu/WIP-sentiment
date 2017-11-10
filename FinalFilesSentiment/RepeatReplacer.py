#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 23 23:31:24 2017


Dealing with repeating characters . Words involve
repeating characters that cause grammatical errors. For instance
consider a sentence, I like it lotttttt. Here, lotttttt refers to lot.
so replacing lotttttt to lot.

This function is not going to work with sentences, so only words!!

@author: naanu
"""

import re
from nltk.corpus import wordnet

class RepeatReplacing(object):
    def __init__(self):
        self.repeat_regexp = re.compile(r'(\w*)(\w)\2(\w*)')
        self.repl = r'\1\2\3'
#    @staticmethod    
    def replace(self, word):
        if wordnet.synsets(word):
            return word
        repl_word = self.repeat_regexp.sub(self.repl, word)
        if repl_word != word:
            return self.replace(repl_word)
        else:
            return repl_word

