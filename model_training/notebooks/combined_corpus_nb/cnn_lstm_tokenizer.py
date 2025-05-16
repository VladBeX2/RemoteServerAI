import os
import re
import json
import numpy as np
import pandas as pd
import string
import pickle
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split

np.random.seed(42)
tf.random.set_seed(42)

def wordopt(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r"https?://\S+|www\.\S+", '', text)
    text = re.sub(r"<.*?>+", '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\n', '', text)
    text = re.sub(r'\w*\d\w*', '', text)  # eliminare cuvinte cu cifre
    text = re.sub(r'\s+', ' ', text).strip()  # eliminare spații multiple
    text = re.sub(r'[“”‘’]', '', text)
    tokens = text.split()
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return " ".join(tokens)

data = pd.read_csv("../../datasets/Combined_Corpus/All.csv")
print("Forma dataset-ului inițial:", data.shape)
data = data[data['word_count'] >= 30]
print("După filtrare:", data.shape)

data['Statement'] = data['Statement'].apply(wordopt)
texts = data['Statement'].values
labels = data['Label'].values

from sklearn.model_selection import train_test_split
X_train_texts, X_test_texts, y_train, y_test = train_test_split(
    texts, labels, test_size=0.2, random_state=42, stratify=labels
)

# 2. Tokenizare și padding
max_num_words = 20000
tokenizer = Tokenizer(num_words=max_num_words)
tokenizer.fit_on_texts(X_train_texts)

with open("saved_models/cnn_lstm_glove/tokenizer.pkl", "wb") as handle:
    pickle.dump(tokenizer, handle)