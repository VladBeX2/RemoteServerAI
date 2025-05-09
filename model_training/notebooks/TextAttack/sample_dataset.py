import pandas as pd

# ===============================================
# Funcție pentru eșantionare disjunctă pe atacuri
# ===============================================

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
    """
    Filtrează articolele în limba engleză care au între min și max cuvinte,
    apoi eșantionează disjunct câte `num_per_attack` exemple pentru fiecare dintre
    cele 3 atacuri: TextFooler, PWWS, TextBugger.

    Returnează un dict cu cheile 'textfooler', 'pwws', 'textbugger'
    și valorile liste de tuple (text, label).

    Ridică eroare dacă nu sunt suficiente exemple eligibile.
    """
    # 1) Filtrare limba + lungime
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

    # 2) Shuffle reproducibil
    eligible = eligible.sample(frac=1, random_state=random_state).reset_index(drop=True)

    # 3) Partitionare disjunctă
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

# Exemplu de utilizare:
# from sample_dataset import sample_for_attacks
# train_df, val_df, test_df = load_and_split(...)
# attack_samples = sample_for_attacks(train_df)
# textfooler_samples = attack_samples['textfooler']
# pwws_samples         = attack_samples['pwws']
# textbugger_samples   = attack_samples['textbugger']
