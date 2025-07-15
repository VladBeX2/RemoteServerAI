# %%
import os
import re
import json
import joblib
import string
import numpy as np
import pandas as pd

from joblib import Parallel, delayed
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score

# %%
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# %%
def wordopt(text):
    
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub(r"https?://\S+|www\.\S+", '', text)
    text = re.sub(r"<.*?>+", '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\n', '', text)
    text = re.sub(r'\w*\d\w*', '', text)  # eliminare cuvinte cu cifre
    text = re.sub(r'\s+', ' ', text).strip()  
    text = re.sub(r'[“”‘’]', '', text)  

    tokens = text.split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return " ".join(tokens)

# %%
def load_glove_embeddings(glove_path):
    
    embeddings_index = {}
    with open(glove_path, "r", encoding="utf-8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
    return embeddings_index

# %%
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

# %%
def train_and_evaluate(vec_name, X_train_vec, X_test_vec, clf_name, clf, y_train, y_test, save_dir):
   
    clf.fit(X_train_vec, y_train)
    y_pred = clf.predict(X_test_vec)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    filename = f"{clf_name}_{vec_name}.joblib".replace(" ", "_")
    model_path = os.path.join(save_dir, filename)
    joblib.dump(clf, model_path)
    print(f"Saved model: {model_path} | Accuracy: {acc:.4f}")

    return {
        "vectorizer": vec_name,
        "classifier": clf_name,
        "accuracy": acc,
        "report": report,
        "model_path": model_path
    }

# %%
data = pd.read_csv("../../datasets/Combined_Corpus/All.csv")
print("Forma initiala a dataset-ului:", data.shape)

# %%
data = data[data['word_count'] >= 30]
print("Dupa filtrare (word_count >= 30):", data.shape)

# %%
data['Statement'] = data['Statement'].apply(wordopt)

# %%
X = data['Statement'].values
y = data['Label'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# %%
glove_path = "../../datasets/GloVe_embeddings/glove.6B.300d.txt"  
embedding_dim = 300
print(f"Loading GloVe embeddings from {glove_path}...")
embeddings_index = load_glove_embeddings(glove_path)

# %%
print("Transforming texts into GloVe vectors...")
X_train_glove = glove_transform(X_train, embeddings_index, embedding_dim)
X_test_glove = glove_transform(X_test, embeddings_index, embedding_dim)


# %%
classifiers = {
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
    "NaiveBayes": GaussianNB(),
    "SVM": SVC(random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
}

# %%
save_dir = "saved_models"
os.makedirs(save_dir, exist_ok=True)

# %%
tasks = []
vec_name = "GloVe_300d"  
for clf_name, clf in classifiers.items():
    tasks.append((vec_name, X_train_glove, X_test_glove, clf_name, clf))

# %%
results = Parallel(n_jobs=4)(
        delayed(train_and_evaluate)(
            vec_name, X_train_vec, X_test_vec, clf_name, clf, y_train, y_test, save_dir
        )
        for (vec_name, X_train_vec, X_test_vec, clf_name, clf) in tasks
    )

# %%
results_summary = {"results": results}
results_file = os.path.join(save_dir, "GloVe_results_summary.json")
with open(results_file, "w") as f:
    json.dump(results_summary, f, indent=4)
print(f"Results summary saved to {results_file}")


