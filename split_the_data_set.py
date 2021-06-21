import numpy as np
from pandas.io.parsers import read_csv
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
import string
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow import keras

clinical_text_df = read_csv("/Users/xiongcaiwei/Downloads/mtsamples.csv")

no_punc_translator = str.maketrans(string.punctuation, ' '*len(string.punctuation))
clinical_text_df['transcription_lower']=clinical_text_df['transcription'].apply(lambda x: ' '.join([i for i in str(x).lower().translate(no_punc_translator).split(' ') if i.isalpha()]))

print(clinical_text_df.head(1))

vectorizer=CountVectorizer(analyzer='word')
feature_space=vectorizer.fit_transform(list(clinical_text_df['transcription_lower']))

count_vect_df = pd.DataFrame(feature_space.todense(), columns=vectorizer.get_feature_names())
new_df=pd.concat([clinical_text_df, count_vect_df], axis=1)

print(new_df.columns)


X=new_df.loc[:, 'aa':]


lb_make = LabelEncoder()
new_df["medical_specialty_code"] = lb_make.fit_transform(new_df["medical_specialty"])


Y=new_df['medical_specialty_code']


X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)


y_train=keras.utils.to_categorical(y_train, clinical_text_df['medical_specialty'].nunique())
y_test=keras.utils.to_categorical(y_test, clinical_text_df['medical_specialty'].nunique())


print(X_train.shape)

print(y_train.shape)