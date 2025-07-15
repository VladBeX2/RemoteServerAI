import re
import pandas as pd
from sklearn.model_selection import train_test_split

 
 
 
def wordopt(text: str) -> str:
    text = re.sub(r'https?://\S+|www\.\S+', '[URL]', text)
    text = text.replace('\n', ' ')
    text = text.encode('ascii', 'ignore').decode('ascii')
    text = re.sub(r'\s+', ' ', text).strip()
    return text

 
 
 
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
   
    df = pd.read_csv(csv_path)

     
    df['word_count'] = df[text_col].apply(lambda x: len(x.split()))
    df = df[df['word_count'] >= min_word_count]
    df.drop(columns=['word_count'], inplace=True)

     
    if lang_col in df.columns:
        df = df[df[lang_col] == 'en']

     
    df[text_col] = df[text_col].apply(wordopt)

     
    train_val, test_df = train_test_split(
        df,
        test_size=test_size,
        stratify=df[label_col],
        random_state=random_state
    )

     
    relative_val = val_size / (1.0 - test_size)
    train_df, val_df = train_test_split(
        train_val,
        test_size=relative_val,
        stratify=train_val[label_col],
        random_state=random_state
    )

     
    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True)
    )

 
 
 
 
 
 
 
 
 
 
 
