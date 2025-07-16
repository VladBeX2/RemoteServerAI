import os
import re
import json
import numpy as np
import pandas as pd
import string

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Conv1D, GlobalMaxPooling1D, MaxPooling1D
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.metrics import classification_report
import pickle

 
np.random.seed(42)
tf.random.set_seed(42)

def wordopt(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r"https?://\S+|www\.\S+", '', text)
    text = re.sub(r"<.*?>+", '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\n', '', text)
    text = re.sub(r'\w*\d\w*', '', text)   
    text = re.sub(r'\s+', ' ', text).strip()   
    text = re.sub(r'[“”‘’]', '', text)
    tokens = text.split()
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return " ".join(tokens)

def load_glove_embeddings(glove_path, embedding_dim=300):
    embeddings_index = {}
    with open(glove_path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
    return embeddings_index

def create_embedding_matrix(word_index, embeddings_index, embedding_dim=300, max_num_words=20000):
    num_words = min(max_num_words, len(word_index) + 1)
    embedding_matrix = np.zeros((num_words, embedding_dim))
    for word, i in word_index.items():
        if i >= max_num_words:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    return embedding_matrix

def create_lstm_model(max_sequence_length, num_words, embedding_dim, embedding_matrix):
    model = Sequential()
    model.add(Embedding(input_dim=num_words, output_dim=embedding_dim,
                        weights=[embedding_matrix], input_length=max_sequence_length,
                        trainable=False))
    model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def create_cnn_model(max_sequence_length, num_words, embedding_dim, embedding_matrix):
    model = Sequential()
    model.add(Embedding(input_dim=num_words, output_dim=embedding_dim,
                        weights=[embedding_matrix], input_length=max_sequence_length,
                        trainable=False))
    model.add(Conv1D(filters=128, kernel_size=5, activation='relu'))
    model.add(GlobalMaxPooling1D())
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def create_cnn_lstm_model(max_sequence_length, num_words, embedding_dim, embedding_matrix):
    model = Sequential()
    model.add(Embedding(input_dim=num_words, output_dim=embedding_dim,
                        weights=[embedding_matrix], input_length=max_sequence_length,
                        trainable=False))
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(LSTM(64, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def main():
     
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

     
    max_num_words = 20000
    tokenizer = Tokenizer(num_words=max_num_words)
    tokenizer.fit_on_texts(X_train_texts)

    with open("saved_models/cnn_lstm_glove/tokenizer.pkl", "wb") as handle:
        pickle.dump(tokenizer, handle)

    X_train_seq = tokenizer.texts_to_sequences(X_train_texts)
    X_test_seq = tokenizer.texts_to_sequences(X_test_texts)

    max_sequence_length = 100
    X_train_pad = pad_sequences(X_train_seq, maxlen=max_sequence_length)
    X_test_pad = pad_sequences(X_test_seq, maxlen=max_sequence_length)

     
    glove_path = "../../datasets/GloVe_embeddings/glove.6B.300d.txt"   
    embedding_dim = 300
    print("Se încarcă GloVe embeddings...")
    embeddings_index = load_glove_embeddings(glove_path, embedding_dim)
    print("Numărul de token-uri GloVe:", len(embeddings_index))

     
    word_index = tokenizer.word_index
    num_words = min(max_num_words, len(word_index) + 1)
    embedding_matrix = create_embedding_matrix(word_index, embeddings_index, embedding_dim, max_num_words)

     
    strategy = tf.distribute.MirroredStrategy()
    print("Numărul de GPU-uri folosite:", strategy.num_replicas_in_sync)

    results = []
    save_dir = "saved_models/cnn_lstm_glove2"
    os.makedirs(save_dir, exist_ok=True)

     
    for model_name in ["LSTM"]:
        with strategy.scope():
            if model_name == "LSTM":
                model = create_lstm_model(max_sequence_length, num_words, embedding_dim, embedding_matrix)
            elif model_name == "CNN":
                model = create_cnn_model(max_sequence_length, num_words, embedding_dim, embedding_matrix)
            elif model_name == "CNN_LSTM":
                model = create_cnn_lstm_model(max_sequence_length, num_words, embedding_dim, embedding_matrix)

        print(f"\nAntrenarea modelului {model_name} folosind MirroredStrategy...")
        early_stop = EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True)
        history = model.fit(X_train_pad, y_train,
                            validation_split=0.1,
                            epochs=10,
                            batch_size=128,
                            callbacks=[early_stop],
                            verbose=2)
        
        loss, accuracy = model.evaluate(X_test_pad, y_test, verbose=0)
        print(f"{model_name} - Test Accuracy: {accuracy:.4f}")

        y_pred_prob = model.predict(X_test_pad)
        y_pred = (y_pred_prob > 0.5).astype("int32")
        report = classification_report(y_test, y_pred, output_dict=True)

        model_path = os.path.join(save_dir, f"{model_name}_GloVe.h5")
        model.save(model_path, save_format='h5', include_optimizer=False)
        print(f"Modelul {model_name} salvat la {model_path}")

        results.append({
            "model": model_name,
            "accuracy": accuracy,
            "report": report,
            "model_path": model_path
        })

    results_summary = {"results": results}
    results_file = os.path.join(save_dir, "results_summary_deep.json")
    with open(results_file, "w") as f:
        json.dump(results_summary, f, indent=4)
    print(f"Rezumatul rezultatelor salvat în {results_file}")

if __name__ == "__main__":
    main()
