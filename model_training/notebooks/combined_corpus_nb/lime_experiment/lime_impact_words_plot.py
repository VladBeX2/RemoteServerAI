import pandas as pd
import ast
import matplotlib.pyplot as plt
from collections import defaultdict
from nltk.corpus import stopwords

# Set engleză pentru stop words
stop_words = set(stopwords.words("english"))

# Adăugăm și "url" la lista de ignorat
ignored_words = stop_words.union({"URL", "url"})

# 1. Încarcă fișierul CSV
df = pd.read_csv("lime_explanations.csv")

# 2. Dicționare pentru scoruri cumulate
positive_contributions = defaultdict(float)
negative_contributions = defaultdict(float)

# 3. Extrage și filtrează contribuțiile
for contrib_str in df["contributions"]:
    try:
        contrib_list = ast.literal_eval(contrib_str)
        for word, score in contrib_list:
            word_clean = word.lower()
            if word_clean in ignored_words:
                continue
            if score > 0:
                positive_contributions[word_clean] += score
            else:
                negative_contributions[word_clean] += abs(score)
    except Exception as e:
        print(f"Eroare la parsare: {contrib_str[:30]}... — {e}")

# 4. Selectează top 20 din fiecare
top_pos = sorted(positive_contributions.items(), key=lambda x: x[1], reverse=True)[:20]
top_neg = sorted(negative_contributions.items(), key=lambda x: x[1], reverse=True)[:20]

# 5. Plot pozitiv (REAL)
plt.figure(figsize=(10, 6))
words, scores = zip(*top_pos)
plt.barh(words[::-1], scores[::-1])
plt.title("Top 20 cuvinte cu impact pozitiv (Real)")
plt.xlabel("Contribuție totală în decizia modelului")
plt.tight_layout()
plt.savefig("lime_positive_words.png")
plt.close()

# 6. Plot negativ (FAKE)
plt.figure(figsize=(10, 6))
words, scores = zip(*top_neg)
plt.barh(words[::-1], scores[::-1], color="red")
plt.title("Top 20 cuvinte cu impact negativ (Fake)")
plt.xlabel("Contribuție totală în decizia modelului")
plt.tight_layout()
plt.savefig("lime_negative_words.png")
plt.close()

print("Graficele au fost salvate ca 'lime_positive_words.png' și 'lime_negative_words.png'.")
