import os
import csv
import time
import datetime
import random
import json
import string
import re

import warnings
from collections import Counter
from math import sqrt

import gensim
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score
warnings.filterwarnings("ignore")
from pandas.io.parsers import read_csv

from absl import app, flags, logging

import sh
import torch as th
import pytorch_lightning as pl
import nlp
import transformers

from sklearn.feature_extraction.text import TfidfVectorizer

raw_datasets = read_csv("/Users/xiongcaiwei/Downloads/mtsamples.csv")

raw_datasets.head(5)

raw_datasets.info()


from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained('nvidia/megatron-bert-cased-345m')

encoded_input = tokenizer("Hello, I'm a single sentence!")
print(encoded_input)



tokenizer.decode(encoded_input["input_ids"])

######## Test #########
batch_sentences = ["Hello I'm a single sentence",
                    "And another sentence",
                    "And the very very last one"]
batch_of_second_sentences = ["I'm a sentence that goes with the first sentence",
                              "And I should be encoded with the second sentence",
                              "And I go with the very last one"]
encoded_inputs = tokenizer(batch_sentences, batch_of_second_sentences)
print(encoded_inputs)

######## Test #########

for ids in encoded_inputs["input_ids"]:
     print(tokenizer.decode(ids))


batch_sentences = ["Hello I'm a single sentence",
                    "And another sentence",
                    "And the very very last one"]
encoded_inputs = tokenizer(batch_sentences)
print(encoded_inputs)


batch = tokenizer(batch_sentences, padding=True, truncation=True, return_tensors="tf")
print(batch)

for ids in encoded_inputs["input_ids"]:
     print(tokenizer.decode(ids))

batch = tokenizer(batch_sentences, batch_of_second_sentences, padding=True, truncation=True, return_tensors="tf")

#  apply these preprocessing steps to all the splits of our dataset at once

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)

# generate a small subset of the training and validation set, to enable faster training:

small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))
full_train_dataset = tokenized_datasets["train"]
full_eval_dataset = tokenized_datasets["test"]

