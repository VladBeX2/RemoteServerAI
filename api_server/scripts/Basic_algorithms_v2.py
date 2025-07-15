# %%
import os
import re
import json
import joblib
import pandas as pd
import numpy as np
from joblib import Parallel, delayed
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score
from sklearn.pipeline import make_pipeline
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import string
import nltk
from nltk.corpus import wordnet
import random
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.model_selection import learning_curve

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

try:
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
except:
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('punkt')
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

def preprocess_text(text, remove_stopwords=True, lemmatize=True):
    
    if not isinstance(text, str):
        return ""
    
    text = text.lower()
    
    text = re.sub(r"https?://\S+|www\.\S+", '[URL]', text)
    
    text = re.sub(r"<.*?>+", '[HTML]', text)
    
    text = re.sub(r'[#%&\(\)\*\+/:;<=>@\[\\\]^_`{|}~]', ' ', text)
    
    text = re.sub(r'\s+', ' ', text).strip()
    
    tokens = word_tokenize(text)
    
    if remove_stopwords:
        tokens = [word for word in tokens if word not in stop_words]
    
    if lemmatize:
        tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
    return " ".join(tokens)

def check_class_distribution(y):
    unique, counts = np.unique(y, return_counts=True)
    distribution = dict(zip(unique, counts))
    
    print("Distribuția claselor:")
    for label, count in distribution.items():
        print(f"Clasa {label}: {count} ({count/len(y)*100:.2f}%)")
    
    minor_class_percentage = min(counts) / len(y) * 100
    is_imbalanced = minor_class_percentage < 40
    print(f"Distribuție dezechilibrată: {is_imbalanced} (clasa minoritară: {minor_class_percentage:.2f}%)")
    return is_imbalanced

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    
    plt.figure(figsize=(10, 6))
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Dimensiunea setului de antrenare")
    plt.ylabel("Acuratețe")
    
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, scoring='accuracy')
    
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    
    plt.grid()
    
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Acuratețe pe antrenare")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Acuratețe pe validare")
    
    plt.legend(loc="best")
    
    diff = train_scores_mean[-1] - test_scores_mean[-1]
    plt.annotate(f'Diferență: {diff:.4f}', 
                 xy=(train_sizes[-1], test_scores_mean[-1]),
                 xytext=(train_sizes[-1] * 0.8, test_scores_mean[-1] - 0.1),
                 arrowprops=dict(facecolor='black', shrink=0.05))
    
    return plt

def train_evaluate_with_cv(vec_name, vectorizer, clf_name, clf, X, y, save_dir):
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    
    X_processed = [preprocess_text(text) for text in X]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )
    
    is_imbalanced = check_class_distribution(y_train)
    
    if vec_name == "Bag_of_Words_(1-2gram)":
        vectorizer = CountVectorizer(ngram_range=(1, 2), max_features=5000)
    else:  # TF-IDF
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=5000, 
                                    sublinear_tf=True, min_df=5)
    
    feature_selector = SelectKBest(chi2, k=3000)
    
    if is_imbalanced and clf_name not in ["SVM", "KNN"]:
        pipeline = ImbPipeline([
            ('vectorizer', vectorizer),
            ('feature_selection', feature_selector),
            ('smote', SMOTE(random_state=RANDOM_SEED)),
            ('classifier', clf)
        ])
    else:
        pipeline = make_pipeline(
            vectorizer,
            feature_selector,
            clf
        )
    
    cv_accuracies = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='accuracy')
    cv_f1 = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='f1_weighted')
    
    pipeline.fit(X_train, y_train)
    
    y_pred = pipeline.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    test_f1 = f1_score(y_test, y_pred, average='weighted')
    test_precision = precision_score(y_test, y_pred, average='weighted')
    test_recall = recall_score(y_test, y_pred, average='weighted')
    
    cv_test_diff = np.mean(cv_accuracies) - test_accuracy
    
    classification_rep = classification_report(y_test, y_pred, output_dict=True)
    
    learning_curve_title = f'Curba de învățare - {clf_name} cu {vec_name}'
    plt = plot_learning_curve(
        pipeline, learning_curve_title, X_train, y_train, cv=cv)
    learning_curve_path = os.path.join(save_dir, f"{clf_name}_{vec_name}_learning_curve.png")
    plt.savefig(learning_curve_path)
    plt.close()
    
    filename = f"{clf_name}_{vec_name}.joblib".replace(" ", "_")
    model_path = os.path.join(save_dir, filename)
    joblib.dump(pipeline, model_path)
    
    print(f"\n--- Model: {clf_name} cu {vec_name} ---")
    print(f"CV Acuratețe: {np.mean(cv_accuracies):.4f} (±{np.std(cv_accuracies):.4f})")
    print(f"CV F1-Score: {np.mean(cv_f1):.4f} (±{np.std(cv_f1):.4f})")
    print(f"Test Acuratețe: {test_accuracy:.4f}")
    print(f"Test F1-Score: {test_f1:.4f}")
    print(f"Diferența CV-Test: {cv_test_diff:.4f}")
    print(f"Salvat la: {model_path}")
    
    return {
        "vectorizer": vec_name,
        "classifier": clf_name,
        "cv_accuracy_mean": float(np.mean(cv_accuracies)),
        "cv_accuracy_std": float(np.std(cv_accuracies)),
        "cv_f1_mean": float(np.mean(cv_f1)),
        "cv_f1_std": float(np.std(cv_f1)),
        "test_accuracy": float(test_accuracy),
        "test_f1": float(test_f1),
        "test_precision": float(test_precision),
        "test_recall": float(test_recall),
        "cv_test_difference": float(cv_test_diff),
        "classification_report": classification_rep,
        "model_path": model_path,
        "learning_curve_path": learning_curve_path
    }

def plot_model_comparison(results, save_path):
    
    models = []
    cv_scores = []
    test_scores = []
    differences = []
    
    for result in results:
        model_name = f"{result['classifier']} + {result['vectorizer']}"
        models.append(model_name)
        cv_scores.append(result['cv_accuracy_mean'])
        test_scores.append(result['test_accuracy'])
        differences.append(result['cv_test_difference'])
    
    sorted_indices = np.argsort(cv_scores)[::-1]
    models = [models[i] for i in sorted_indices]
    cv_scores = [cv_scores[i] for i in sorted_indices]
    test_scores = [test_scores[i] for i in sorted_indices]
    differences = [differences[i] for i in sorted_indices]
    
    fig, ax1 = plt.subplots(figsize=(14, 8))
    
    x = np.arange(len(models))
    width = 0.35
    ax1.bar(x - width/2, cv_scores, width, label='CV Acuratețe', color='skyblue')
    ax1.bar(x + width/2, test_scores, width, label='Test Acuratețe', color='lightgreen')
    
    ax1.set_title('Comparație Acuratețe și Diferență CV-Test', fontsize=16)
    ax1.set_ylabel('Acuratețe', fontsize=14)
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, rotation=45, ha='right', fontsize=12)
    ax1.legend(loc='upper left', fontsize=12)
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    
    ax2 = ax1.twinx()
    ax2.plot(x, differences, 'ro-', linewidth=2, markersize=8, label='Diferență CV-Test')
    ax2.set_ylabel('Diferență CV-Test', color='r', fontsize=14)
    ax2.tick_params(axis='y', labelcolor='r')
    ax2.axhline(y=0.05, color='r', linestyle='--', alpha=0.7)
    ax2.text(x[-1], 0.05, '  Prag overfitting (0.05)', color='r', va='center')
    
    for i, v in enumerate(cv_scores):
        ax1.text(i - width/2, v + 0.01, f'{v:.4f}', ha='center', va='bottom', fontsize=10)
    
    for i, v in enumerate(test_scores):
        ax1.text(i + width/2, v + 0.01, f'{v:.4f}', ha='center', va='bottom', fontsize=10)
    
    for i, v in enumerate(differences):
        ax2.text(i, v + 0.01, f'{v:.4f}', ha='center', va='bottom', color='r', fontsize=10)
    
    ax2.legend(loc='upper right', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def analyze_dataset(data):
    
    print(f"Număr total de înregistrări: {data.shape[0]}")
    
    print(f"Coloane disponibile: {data.columns.tolist()}")
    
    class_dist = data['Label'].value_counts(normalize=True) * 100
    print("\nDistribuția claselor:")
    print(class_dist)
    
    data['text_length'] = data['Statement'].str.len()
    
    print(f"\nLungimea medie a textelor: {data['text_length'].mean():.2f} caractere")
    print(f"Lungimea minimă: {data['text_length'].min()} caractere")
    print(f"Lungimea maximă: {data['text_length'].max()} caractere")
    
    print("\nLungimea medie a textelor pe clase:")
    print(data.groupby('Label')['text_length'].mean())
    
    print("\nVerificare cuvinte comune în fiecare clasă...")
    
    def get_common_words(texts, n=20):
        all_words = ' '.join(texts).split()
        from collections import Counter
        return Counter(all_words).most_common(n)
    
    fake_samples = data[data['Label'] == 0]['Statement'].sample(min(1000, sum(data['Label'] == 0)))
    real_samples = data[data['Label'] == 1]['Statement'].sample(min(1000, sum(data['Label'] == 1)))
    
    processed_fake = [preprocess_text(text) for text in fake_samples]
    processed_real = [preprocess_text(text) for text in real_samples]
    
    print("\nCele mai comune cuvinte în știri false:")
    print(get_common_words(processed_fake))
    
    print("\nCele mai comune cuvinte în știri reale:")
    print(get_common_words(processed_real))
    
    from sklearn.feature_extraction.text import HashingVectorizer
    
    vectorizer = HashingVectorizer(n_features=1000)
    X_hashed = vectorizer.transform(data['Statement'])
    
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
    
    sample_size = min(1000, data.shape[0])
    sample_indices = np.random.choice(data.shape[0], sample_size, replace=False)
    X_sample = X_hashed[sample_indices]
    
    similarity_matrix = cosine_similarity(X_sample)
    
    np.fill_diagonal(similarity_matrix, 0)
    
    high_similarity = np.where(similarity_matrix > 0.9)
    
    print(f"\nNumărul de perechi de texte cu similaritate > 0.9: {len(high_similarity[0])}")
    
    if len(high_similarity[0]) > 0:
        print("\nExemple de texte foarte similare:")
        for i in range(min(5, len(high_similarity[0]))):
            idx1, idx2 = sample_indices[high_similarity[0][i]], sample_indices[high_similarity[1][i]]
            print(f"\nText 1 (Label {data.iloc[idx1]['Label']}):")
            print(data.iloc[idx1]['Statement'][:100] + "...")
            print(f"Text 2 (Label {data.iloc[idx2]['Label']}):")
            print(data.iloc[idx2]['Statement'][:100] + "...")
    
    return {
        'total_records': data.shape[0],
        'class_distribution': class_dist.to_dict(),
        'avg_text_length': data['text_length'].mean(),
        'similar_texts_count': len(high_similarity[0])
    }

def main():
    import nltk

    nltk.download("punkt_tab", quiet=True)
    save_dir = "saved_models/optimized"
    os.makedirs(save_dir, exist_ok=True)
    
    data = pd.read_csv("../model_training/datasets/Combined_Corpus/All.csv")
    print(f"Dimensiunea inițială a datelor: {data.shape}")
    
    data = data[data['word_count'] >= 30]
    print(f"Dimensiunea după filtrare: {data.shape}")
    
    dataset_analysis = analyze_dataset(data)
    
    vectorizers = {
        "Bag_of_Words_(1-2gram)": None,
        "TFIDF_(1-2gram)": None  
    }
    
    classifiers = {
        "RandomForest": RandomForestClassifier(
            n_estimators=100, 
            max_depth=20, 
            min_samples_split=10, 
            min_samples_leaf=5,  
            max_features='sqrt', 
            random_state=RANDOM_SEED,
            n_jobs=-1
        ),
        "LogisticRegression": LogisticRegression(
            C=1.0, 
            max_iter=1000,
            random_state=RANDOM_SEED
        ),
        "NaiveBayes": MultinomialNB(
            alpha=0.5 
        ),
        "SVM": SVC(
            C=1.0, 
            kernel='linear', 
            probability=True,
            random_state=RANDOM_SEED
        ),
        "KNN": KNeighborsClassifier(
            n_neighbors=5,
            weights='distance',  
            n_jobs=-1
        )
    }
    
    X = data['Statement'].values
    y = data['Label'].values
    
    tasks = []
    for vec_name in vectorizers.keys():
        for clf_name, clf in classifiers.items():
            tasks.append((vec_name, None, clf_name, clf))
    
    if len(tasks) <= 2:
        results = []
        for vec_name, vec, clf_name, clf in tasks:
            result = train_evaluate_with_cv(vec_name, vec, clf_name, clf, X, y, save_dir)
            results.append(result)
    else:
        results = Parallel(n_jobs=2)(
            delayed(train_evaluate_with_cv)(vec_name, vec, clf_name, clf, X, y, save_dir)
            for vec_name, vec, clf_name, clf in tasks
        )
    
    results_summary = {
        "dataset_analysis": dataset_analysis,
        "results": results
    }
    
    results_file = os.path.join(save_dir, "results_summary_optimized.json")
    with open(results_file, "w") as f:
        json.dump(results_summary, f, indent=4)
    
    print(f"\nRezultatele au fost salvate în {results_file}")
    
    plot_model_comparison(results, os.path.join(save_dir, "model_comparison.png"))
    
    min_diff_idx = np.argmin([r['cv_test_difference'] for r in results])
    best_model = results[min_diff_idx]
    
    print("\n==== Cel mai bun model (diferență minimă între CV și testare) ====")
    print(f"Model: {best_model['classifier']} cu {best_model['vectorizer']}")
    print(f"CV Acuratețe: {best_model['cv_accuracy_mean']:.4f}")
    print(f"Test Acuratețe: {best_model['test_accuracy']:.4f}")
    print(f"Diferența: {best_model['cv_test_difference']:.4f}")

if __name__ == "__main__":
    main()


