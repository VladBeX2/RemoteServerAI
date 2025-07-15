import os
import re
import pandas as pd
import torch
import torch.multiprocessing as mp

 
 
 
MODEL_PATH     = "../../combined_corpus_nb/saved_models/roberta_v3/roberta_v3_torch_model"
TOKENIZER_PATH = "../../combined_corpus_nb/saved_models/roberta_v3/roberta_v3_tokenizer"
DATA_PATH      = "../../../datasets/Combined_Corpus/All_cleaned.csv"

NUM_DEMO   = 30
MIN_WORDS  = 100
MAX_WORDS  = 500

 
def preprocess(text: str) -> str:
    text = re.sub(r'https?://\S+|www\.\S+', '[URL]', text)
    text = text.replace('\n', ' ')
    return re.sub(r'\s+', ' ', text).strip()

 
def load_demo_samples():
    df = pd.read_csv(DATA_PATH)
    df["text"] = df["text"].apply(preprocess)
    df["word_count"] = df["text"].str.split().apply(len)
    mask = (
        (df["language"] == "en") &
        (df["word_count"] > MIN_WORDS) &
        (df["word_count"] < MAX_WORDS)
    )
    sub = df[mask].iloc[:NUM_DEMO]
    return list(zip(sub["text"].tolist(), sub["label"].tolist()))

 
def worker(rank: int, world_size: int, all_chunks):
     
    os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)

     
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    import textattack
    from textattack import Attacker, AttackArgs
    from textattack.attack_recipes import BERTAttackLi2020
    from textattack.models.wrappers import HuggingFaceModelWrapper
    from textattack.datasets import Dataset
    import nltk
    nltk.download("averaged_perceptron_tagger_eng")

     
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

     
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH).to(device)
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
    wrapper = HuggingFaceModelWrapper(model, tokenizer)

     
    attack = BERTAttackLi2020.build(wrapper)

     
    attack_args = AttackArgs(
        num_examples=len(all_chunks[rank]),
        log_to_csv=f"bertattack_results_gpu_{rank}.csv",
        disable_stdout=False,
        checkpoint_interval=10,
        checkpoint_dir=f"bertattack_chkpts_gpu_{rank}"
    )

     
    attacker = Attacker(attack, Dataset(all_chunks[rank]), attack_args)
    attacker.attack_dataset()

if __name__ == "__main__":
     
    all_samples = load_demo_samples()
    ngpu = min(3, torch.cuda.device_count())
    if ngpu == 0:
        raise RuntimeError("Nu am găsit GPU-uri!")

     
    chunk_size = len(all_samples) // ngpu
    chunks = [all_samples[i*chunk_size:(i+1)*chunk_size] for i in range(ngpu)]
    if len(all_samples) % ngpu:
        chunks[-1] += all_samples[ngpu*chunk_size:]

     
    mp.spawn(
        fn=worker,
        args=(ngpu, chunks),
        nprocs=ngpu,
        join=True
    )

