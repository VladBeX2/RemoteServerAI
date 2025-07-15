import pandas as pd

 
 
 

def sample_for_attacks(
    df: pd.DataFrame,
    text_col: str = 'text',
    label_col: str = 'label',
    lang_col: str = 'language',
    min_words: int = 100,
    max_words: int = 500,
    num_per_attack: int = 3000,
    random_state: int = 42
) -> dict:
   
    df['word_count'] = df[text_col].str.split().apply(len)
    mask = (
        (df[lang_col] == 'en') &
        (df['word_count'] >= min_words) &
        (df['word_count'] <= max_words)
    )
    eligible = df[mask].copy()

    total_needed = num_per_attack * 3
    if len(eligible) < total_needed:
        raise ValueError(
            f"Nu sunt suficiente articole eligibile: nevoie de {total_needed}, disponibile {len(eligible)}"
        )

     
    eligible = eligible.sample(frac=1, random_state=random_state).reset_index(drop=True)

     
    samples = {}
    splits = {
        'textfooler': (0, num_per_attack),
        'pwws':         (num_per_attack, 2*num_per_attack),
        'textbugger':   (2*num_per_attack, 3*num_per_attack),
    }
    for attack_name, (start, end) in splits.items():
        chunk = eligible.iloc[start:end]
        samples[attack_name] = list(zip(chunk[text_col].tolist(), chunk[label_col].tolist()))

    return samples

 
 
 
 
 
 
 
