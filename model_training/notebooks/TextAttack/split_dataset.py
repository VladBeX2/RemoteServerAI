import re
import pandas as pd
from sklearn.model_selection import train_test_split

# ============================================
# Funcții de preprocesare și filtrare inițială
# ============================================
def wordopt(text: str) -> str:
    text = re.sub(r'https?://\S+|www\.\S+', '[URL]', text)
    text = text.replace('\n', ' ')
    text = text.encode('ascii', 'ignore').decode('ascii')
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ============================================
# Split dataset cu filtrare și curățare
# ============================================
def load_and_split(
    csv_path: str,
    text_col: str = 'text',
    label_col: str = 'label',
    lang_col: str = 'language',
    min_word_count: int = 30,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42
):
    """
    Încarcă CSV-ul, elimină textele sub min_word_count cuvinte,
    curăță textul cu wordopt, filtrează pe limba engleză,
    apoi împarte stratificat în train/validation/test.

    Return:
        train_df, val_df, test_df
    """
    # 1) Încarcă datele
    df = pd.read_csv(csv_path)

    # 2) Elimină texte prea scurte ("garbage")
    df['word_count'] = df[text_col].apply(lambda x: len(x.split()))
    df = df[df['word_count'] >= min_word_count]
    df.drop(columns=['word_count'], inplace=True)

    # 3) Filtrează pe limba engleză
    if lang_col in df.columns:
        df = df[df[lang_col] == 'en']

    # 4) Curățare text cu funcția wordopt
    df[text_col] = df[text_col].apply(wordopt)

    # 5) Primul split: train_val and test
    train_val, test_df = train_test_split(
        df,
        test_size=test_size,
        stratify=df[label_col],
        random_state=random_state
    )

    # 6) Split train_val în train și val
    relative_val = val_size / (1.0 - test_size)
    train_df, val_df = train_test_split(
        train_val,
        test_size=relative_val,
        stratify=train_val[label_col],
        random_state=random_state
    )

    # 7) Reset index
    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True)
    )

# Exemplu de utilizare:
# train_df, val_df, test_df = load_and_split(
#     csv_path='../../datasets/Combined_Corpus/All_cleaned.csv',
#     text_col='text',
#     label_col='label',
#     lang_col='language',
#     min_word_count=30,
#     test_size=0.15,
#     val_size=0.15,
#     random_state=42
# )
