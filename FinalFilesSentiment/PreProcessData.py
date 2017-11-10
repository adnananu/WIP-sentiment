#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 28 12:34:34 2017

This script defines function used to pre-process data before training
I have add most of the functionality to clean the data for this particular
problem.
@author: naanu
"""



from bs4 import BeautifulSoup
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from string import punctuation
import Utility
import re,sys


def cleanData(UtilObj,
              line,
              rmvContraction = False,
              rmvSmileys = False,
              rmvStopWords = False,
              rmvNumbers = False,
              correctSpelling=False,
              rmvReaptWords=False
              ):
    
    """
    clean the data and return sentences as String
    
    """
    try:
        processedData = processCleaning(  UtilObj,
                                          line.decode("utf-8"),
                                          rmvContraction,
                                          rmvSmileys,
                                          rmvStopWords,
                                          rmvNumbers,
                                          correctSpelling,
                                          rmvReaptWords
                                          )
    except (StandardError, RuntimeError) as err:
        print('Error occured while cleaning the data!\n'+str(err))
    except:
        print("Unexpected error:", sys.exc_info()[0]) # catch *all* exceptions
        raise
    
    return processedData




def processCleaning(  UtilObj,
                      rawData,
                      rmvContraction,
                      rmvSmileys,
                      rmvStopWords,
                      rmvNumbers,
                      correctSpelling,
                      rmvReaptWords
                      ):
    
    """
    This function performs all the actions required to clean the data.
    
    """
    
    # Using BeautifulSoup library to remove HTML tags from raw text
    # for example (<br /><br />)!!
    rawData = BeautifulSoup(rawData).text
    
    """ Ignoring expandAbbreviations() for the time being"""
    # This function expand the abbreviations in our text. for example
    # US -> United States
    #rawData = expandAbbreviations(rawData)
    
    # Replacing Contractions like doesn't can be replaced by does not
    if rmvContraction:
        rawData = UtilObj.expandContractions(rawData.lower())
    
    # Removing punctuations such as ":)" (smileys)
    if rmvSmileys:
        rawData = UtilObj.removeSmileys(rawData)

    # Removing stop words like . # is, it, shall...etc
    if rmvStopWords:
        StopWords = set(stopwords.words('english')+list(punctuation))
        newWords = [w for w in word_tokenize(rawData) if w not in StopWords]
        rawData = " ".join(newWords)
    
    # Removing digits as they dont give us much information in this case
    if rmvNumbers:
        rawData = re.sub("[^A-Za-z]+", ' ', rawData)
    
    
    # Dealing with repeating characters and wrong spellings. 
    if correctSpelling and rmvReaptWords:
        result = []
        words = rawData.split()
        for word in words:
            tempWord = UtilObj.replaceRepeat(word)
            result.append(UtilObj.replaceSpelling(tempWord))
        rawData = " ".join(result).lower()
        del result
       
    return rawData
    
"""
code below is temporary!

"""    
def callme():
    
    fReading = open("/home/naanu/Music/dataForTest/sampleOnly.txt", 'r')
    text =""
    for line in fReading:
        text = line
    x = Utility.UtilityClass()    
    newData = cleanData(x,text,True,True,True,True,True,True)
    print(newData)
    
#callme()