import os
from datasets import load_dataset, Dataset, DatasetDict, load_from_disk
from tqdm import tqdm
from transformers import AutoTokenizer

DATA_DIR = "data/de_en_bpe"        
MAX_LEN = 64                     
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
# PAD: [PAD] 0
# BOS: None None
# EOS: None None
# UNK: [UNK] 100
# CLS: [CLS] 101
# SEP: [SEP] 102

def tokenize_dataset(examples):
    src = examples['text']
    tgt = examples['labels']

    src = tokenizer(src, padding='max_length', max_length=MAX_LEN, truncation=True)
    tgt = tokenizer(tgt, padding='max_length', max_length=MAX_LEN, truncation=True)


    return {
        'src_input_ids': src['input_ids'],
        "tgt_input_ids": tgt['input_ids'],
        'src_attention_mask': src['attention_mask'],
        'tgt_attention_mask': tgt['attention_mask'],
        # "tgt_label_ids": tgt_label_ids,
    }

def main():
    if os.path.isdir(DATA_DIR):
        print(f"Found existing preprocessed dataset at {DATA_DIR}")
        ds = load_from_disk(DATA_DIR)
        print(ds)
        return

    raw_ds = load_dataset("wmt19", "de-en")

    train_raw = raw_ds["train"].shuffle(seed=42).select(range(1000000))
    val_raw = raw_ds["validation"]

    train_dataset = train_raw.map(
        lambda x: {'text': x['translation']['de'], 'labels': x['translation']['en']},
        remove_columns=['translation']
    )
    val_dataset = val_raw.map(
        lambda x: {'text': x['translation']['de'], 'labels': x['translation']['en']},
        remove_columns=['translation']
    )

    train_ds = train_dataset.map(tokenize_dataset, batched=True, desc="Tokenizing train")
    val_ds = val_dataset.map(tokenize_dataset, batched=True, desc="Tokenizing valid")
    
    train_ds.set_format(type='torch', columns=['src_input_ids', 'tgt_input_ids', 'src_attention_mask', 'tgt_attention_mask'])
    val_ds.set_format(type='torch', columns=['src_input_ids', 'tgt_input_ids', 'src_attention_mask', 'tgt_attention_mask'])

    val_len = len(val_ds)
    mid = val_len // 2

    dev_ds = val_ds.select(range(0, mid))
    test_ds = val_ds.select(range(mid, val_len))

    dataset = DatasetDict({
        "train": train_ds,
        "dev": dev_ds,
        "test": test_ds,
    })

    os.makedirs(DATA_DIR, exist_ok=True)
    dataset.save_to_disk(DATA_DIR)

    print("Preprocessing complete")
    print(dataset)

if __name__ == "__main__":
    main()

