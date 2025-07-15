import os
import sys
import torch
import torch.multiprocessing as mp
import nltk
import re

 
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from transformers import AutoModelForSequenceClassification, AutoTokenizer
import textattack
from textattack import Attacker, AttackArgs
from textattack.attack_recipes import PWWSRen2019
from textattack.models.wrappers import HuggingFaceModelWrapper
from textattack.datasets import Dataset

 
from split_dataset import load_and_split
from sample_dataset import sample_for_attacks

 
 
 
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

 
 
 

def worker(rank: int, world_size: int, samples_chunks):
     
    os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

     
    nltk.download("averaged_perceptron_tagger_eng", quiet=True)

     
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH).to(device)
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
    wrapper = HuggingFaceModelWrapper(model, tokenizer)

     
    attack = PWWSRen2019.build(wrapper)

     
    attack_args = AttackArgs(
        num_examples=len(samples_chunks[rank]),
        log_to_csv=f"test_pwws_results_gpu_{rank}.csv",
        disable_stdout=False,
        checkpoint_interval=100,
        checkpoint_dir=f"test_pwws_chkpts_gpu_{rank}"
    )

     
    attacker = Attacker(attack, Dataset(samples_chunks[rank]), attack_args)
    attacker.attack_dataset()

 
 
 

if __name__ == "__main__":
     
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
    pwws_samples = attack_samples['pwws']

     
    ngpu = min(3, torch.cuda.device_count())
    if ngpu == 0:
        raise RuntimeError("Nu am găsit GPU-uri pentru atac!")

     
    chunk_size = len(pwws_samples) // ngpu
    chunks = [
        pwws_samples[i*chunk_size:(i+1)*chunk_size]
        for i in range(ngpu)
    ]
    if len(pwws_samples) % ngpu:
        chunks[-1] += pwws_samples[ngpu*chunk_size:]

     
    mp.spawn(
        fn=worker,
        args=(ngpu, chunks),
        nprocs=ngpu,
        join=True
    )

    print("=== PWWS: 3000 articole atacate în paralel pe GPU-uri ===")
