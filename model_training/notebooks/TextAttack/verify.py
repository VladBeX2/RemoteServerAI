import pandas as pd

# Încarcă cele două fișiere
train_df = pd.read_csv("train_augmented.csv")
test_df  = pd.read_csv("test_augmented.csv")

# Extrage doar coloana 'text'
train_texts = set(train_df["text"].tolist())
test_texts  = set(test_df["text"].tolist())

# Calculează intersecția
common_texts = train_texts.intersection(test_texts)
n_common = len(common_texts)

# Raportează rezultatele
print(f"Număr de exemple duplicate în test_augmented vs. train_augmented: {n_common}")
if n_common > 0:
    print("\nPrimele 10 exemple comune:")
    for i, txt in enumerate(list(common_texts)[:10], 1):
        print(f"{i}. {txt}")

