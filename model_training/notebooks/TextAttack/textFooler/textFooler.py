import os
import sys
import torch
import torch.multiprocessing as mp
import nltk
import re
import pandas as pd

# Adaugă directorul părinte pentru importuri locale
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from split_dataset import load_and_split
from sample_dataset import sample_for_attacks

from transformers import AutoModelForSequenceClassification, AutoTokenizer
import textattack
from textattack import Attacker, AttackArgs
from textattack.attack_recipes import TextFoolerJin2019
from textattack.models.wrappers import HuggingFaceModelWrapper
from textattack.datasets import Dataset

# ================================
# Configurații generale
# ================================
MODEL_PATH     = "../../combined_corpus_nb/saved_models/roberta_v3/roberta_v3_torch_model"
TOKENIZER_PATH = "../../combined_corpus_nb/saved_models/roberta_v3/roberta_v3_tokenizer"
DATA_PATH      = "../../../datasets/Combined_Corpus/All_cleaned.csv"
LABEL_COL      = "label"
TEXT_COL       = "text"
LANG_COL       = "language"
MIN_WORDS      = 100
MAX_WORDS      = 500
NUM_PER_ATTACK = 800
RANDOM_STATE   = 42

# ================================
# Funcție de sampling pentru TextFooler
# ================================
# def sample_textfooler(df: pd.DataFrame):
#     """
#     Filtrează df pentru limba engleză și număr de cuvinte între MIN_WORDS și MAX_WORDS,
#     apoi eșantionează reproducibil NUM_PER_ATTACK exemple.
#     """
#     df['word_count'] = df[TEXT_COL].str.split().apply(len)
#     mask = (
#         (df[LANG_COL] == 'en') &
#         (df['word_count'] >= MIN_WORDS) &
#         (df['word_count'] <= MAX_WORDS)
#     )
#     elig = df[mask].sample(n=NUM_PER_ATTACK, random_state=RANDOM_STATE)
#     return list(zip(elig[TEXT_COL].tolist(), elig[LABEL_COL].tolist()))

# ================================
# Worker pentru TextFooler
# ================================

def worker(rank: int, world_size: int, samples_chunks):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(rank)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    nltk.download('averaged_perceptron_tagger_eng', quiet=True)

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH).to(device)
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
    wrapper = HuggingFaceModelWrapper(model, tokenizer)

    attack = TextFoolerJin2019.build(wrapper)
    attack.transformation.max_candidates = 50  # default

    attack_args = AttackArgs(
        num_examples=len(samples_chunks[rank]),
        log_to_csv=f'test_textfooler_results_gpu_{rank}.csv',
        disable_stdout=False,
        checkpoint_interval=100,
        checkpoint_dir=f'test_textfooler_chkpts_gpu_{rank}'
    )

    attacker = Attacker(attack, Dataset(samples_chunks[rank]), attack_args)
    attacker.attack_dataset()

# ================================
# Script principal
# ================================
if __name__ == '__main__':
    # 1) Split reproducibil dataset
    train_df, val_df, test_df = load_and_split(
        csv_path=DATA_PATH,
        text_col=TEXT_COL,
        label_col=LABEL_COL,
        lang_col=LANG_COL,
        min_word_count=30,
        test_size=0.15,
        val_size=0.15,
        random_state=RANDOM_STATE
    )

    # 2) Sample pentru TextFooler
    attack_samples = sample_for_attacks(
        test_df,
        text_col=TEXT_COL,
        label_col=LABEL_COL,
        lang_col=LANG_COL,
        min_words=MIN_WORDS,
        max_words=MAX_WORDS,
        num_per_attack=NUM_PER_ATTACK,
        random_state=RANDOM_STATE
    )
    samples = attack_samples['textfooler']

    # 3) Împărțire pe GPU
    ngpu = min(3, torch.cuda.device_count())
    if ngpu == 0:
        raise RuntimeError('Nu am găsit GPU-uri!')
    chunk_size = len(samples) // ngpu
    chunks = [samples[i*chunk_size:(i+1)*chunk_size] for i in range(ngpu)]
    if len(samples) % ngpu:
        chunks[-1] += samples[ngpu*chunk_size:]

    # 4) Rulează atacurile
    mp.spawn(
        fn=worker,
        args=(ngpu, chunks),
        nprocs=ngpu,
        join=True
    )

    print('=== TextFooler a atacat 3000 exemple în paralel pe GPU-uri ===')
