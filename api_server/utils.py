import numpy as np
import re
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

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

def wordopt_lite(text):
    text = re.sub(r'https?://\S+|www\.\S+', '[URL]', text)
    text = text.replace('\n', ' ')
    text = text.encode('ascii', 'ignore').decode('ascii')
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def load_glove_embeddings(glove_path):
    
    embeddings_index = {}
    with open(glove_path, "r", encoding="utf-8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
    return embeddings_index

def glove_transform(texts, embeddings_index, embedding_dim=300):
    
    X_vectors = []
    for text in texts:
        tokens = text.split()
        valid_vectors = []
        for token in tokens:
            if token in embeddings_index:
                valid_vectors.append(embeddings_index[token])
        if len(valid_vectors) == 0:
            X_vectors.append(np.zeros(embedding_dim, dtype='float32'))
        else:
            X_vectors.append(np.mean(valid_vectors, axis=0))
    return np.array(X_vectors, dtype='float32')