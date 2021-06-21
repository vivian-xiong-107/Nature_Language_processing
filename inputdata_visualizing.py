from collections import Counter

import gensim
import pandas as pd
import numpy as np

from pandas.io.parsers import read_csv


clinical_text_df = read_csv("/Users/xiongcaiwei/Downloads/mtsamples.csv")

# show the first five terms within input data

clinical_text_df.head(5)

# get the information from input data
clinical_text_df.info()

from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
import nltk

# Account the number of sentences and the number of words

def get_sentence_word_count(text_list):
    sent_count = 0
    word_count = 0
    vocab = {}
  
    for text in text_list:
        sentences=sent_tokenize(str(text).lower())
        sent_count = sent_count + len(sentences)
        for sentence in sentences:
            words=word_tokenize(sentence)
            for word in words:
                if(word in vocab.keys()):
                    vocab[word] = vocab[word] +1
                else:
                    vocab[word] =1 
    word_count = len(vocab.keys())
    return sent_count,word_count


clinical_text_df = clinical_text_df[clinical_text_df['transcription'].notna()]
sent_count,word_count= get_sentence_word_count(clinical_text_df['transcription'].tolist())
print("Number of sentences in transcriptions column: "+ str(sent_count))
print("Number of unique words in transcriptions column: "+str(word_count))

# Account the category and each category's sentences

data_categories  = clinical_text_df.groupby(clinical_text_df['medical_specialty'])
i = 1
print('===========Original Categories =======================')
for catName,dataCategory in data_categories:
    print('Cat:'+str(i)+' '+catName + ' : '+ str(len(dataCategory)) )
    i = i+1
print('==================================')


