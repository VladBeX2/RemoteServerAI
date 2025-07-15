import pandas as pd

 
train_df = pd.read_csv("train_augmented.csv")
test_df  = pd.read_csv("test_augmented.csv")

 
train_texts = set(train_df["text"].tolist())
test_texts  = set(test_df["text"].tolist())

 
common_texts = train_texts.intersection(test_texts)
n_common = len(common_texts)

 
print(f"Număr de exemple duplicate în test_augmented vs. train_augmented: {n_common}")
if n_common > 0:
    print("\nPrimele 10 exemple comune:")
    for i, txt in enumerate(list(common_texts)[:10], 1):
        print(f"{i}. {txt}")

