import os
import sys
import torch
import torch.multiprocessing as mp
import nltk
import re

# Adaugă directorul părinte pentru importuri locale
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from transformers import AutoModelForSequenceClassification, AutoTokenizer
import textattack
from textattack import Attacker, AttackArgs
from textattack.attack_recipes import TextBuggerLi2018
from textattack.models.wrappers import HuggingFaceModelWrapper
from textattack.datasets import Dataset

# Funcții de split și sampling
from split_dataset import load_and_split
from sample_dataset import sample_for_attacks

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
# Worker pentru TextBugger
# ================================

def worker(rank: int, world_size: int, samples_chunks):
    # izolează GPU-ul
    os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # NLTK pentru PoS tagging (opțional)
    nltk.download("averaged_perceptron_tagger_eng", quiet=True)

    # Încarcă model și tokenizer
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH).to(device)
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
    wrapper = HuggingFaceModelWrapper(model, tokenizer)

    # Construiește atacul TextBugger
    attack = TextBuggerLi2018.build(wrapper)

    # Setează AttackArgs
    attack_args = AttackArgs(
        num_examples=len(samples_chunks[rank]),
        log_to_csv=f"test_textbugger_results_gpu_{rank}.csv",
        disable_stdout=False,
        checkpoint_interval=100,
        checkpoint_dir=f"test_textbugger_chkpts_gpu_{rank}"
    )

    # Rulează atacul
    attacker = Attacker(attack, Dataset(samples_chunks[rank]), attack_args)
    attacker.attack_dataset()

# ================================
# Script principal
# ================================

if __name__ == "__main__":
    # 1) Split reproducibil
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

    # 2) Eșantionare disjunctă pentru TextBugger
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
    textbugger_samples = attack_samples['textbugger']

    # 3) Setare multiprocessing
    ngpu = min(3, torch.cuda.device_count())
    if ngpu == 0:
        raise RuntimeError("Nu am găsit GPU-uri pentru atac!")

    # Împarte în chunk-uri
    chunk_size = len(textbugger_samples) // ngpu
    chunks = [
        textbugger_samples[i*chunk_size:(i+1)*chunk_size]
        for i in range(ngpu)
    ]
    if len(textbugger_samples) % ngpu:
        chunks[-1] += textbugger_samples[ngpu*chunk_size:]

    # 4) Rulează în paralel
    mp.spawn(
        fn=worker,
        args=(ngpu, chunks),
        nprocs=ngpu,
        join=True
    )

    print("=== TextBugger: 3000 articole atacate în paralel pe GPU-uri ===")
