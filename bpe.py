import os
from datasets import load_dataset
import sentencepiece as spm

def train_dataset():
    ds = load_dataset("wmt/wmt19", "cs-en")

    os.makedirs("data", exist_ok=True)
    corpus_path = "data/corpus.txt"

    with open(corpus_path, "w", encoding="utf-8") as f:
        for folder in ['train', 'validation']:
            for text in ds[folder]:
                f.write(text['translation']['cs'] + "\n")
                f.write(text['translation']['en'] + "\n")

    spm.SentencePieceTrainer.train(
        input=corpus_path,
        model_prefix = 'data/spm_cs_en',
        vocab_size = 37000,
        model_type = 'bpe',
        pad_id = 0,
        unk_id=1,
        bos_id=2,
        eos_id=3
    )

if __name__ == "__main__":
    train_dataset()






