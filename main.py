import os
import math
import argparse
from typing import List, Dict, Any, Tuple, Optional

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from datasets import load_from_disk
from tqdm import tqdm
import wandb

import sacrebleu
from transformers import AutoTokenizer

from Transformer_From_Scratch.model import Transformer  

# Mask
def make_src_mask(src_ids: torch.Tensor, pad_id: int) -> torch.Tensor:
    mask = (src_ids != pad_id).unsqueeze(1).unsqueeze(2)
    return mask

def make_causal_mask(T: int, device: torch.device) -> torch.Tensor:
    # (1,1,T,T)
    return torch.triu(torch.ones(T, T, device=device, dtype=torch.bool), diagonal=1)[None, None, :, :]

def make_tgt_mask(tgt_ids: torch.Tensor, pad_id: int) -> torch.Tensor:
    # (B,Tt) -> (B,1,Tt,Tt) True=mask
    mask = (tgt_ids != pad_id).unsqueeze(1).unsqueeze(2)
    seq_len = tgt_ids.size(1)
    look_ahead_mask = torch.tril(
        torch.ones(seq_len, seq_len, device=tgt_ids.device, dtype=torch.bool)
    ).unsqueeze(0).unsqueeze(0)

    mask = mask & look_ahead_mask
    return mask

class NoamLR:
    def __init__(self, optimizer, d_model: int, warmup_steps: int):
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup = warmup_steps
        self.step_num = 0

    def step(self):
        self.step_num += 1
        lr = (self.d_model ** -0.5) * min(
            self.step_num ** -0.5,
            self.step_num * (self.warmup ** -1.5),
        )
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr
        self.optimizer.step()
        return lr

@torch.no_grad()
def greedy_decode(
    model: nn.Module,
    src_ids: torch.Tensor,          # (B, Ts)
    pad_id: int,
    bos_id: int,
    eos_id: int,
    max_len: int,
    device: torch.device,
):
    model.eval()

    B, Ts = src_ids.size()
    # (B,1)
    generated = torch.full((B, 1), bos_id, dtype=torch.long, device=device)

    src_mask = make_src_mask(src_ids, pad_id)

    for _ in range(max_len - 1):
        tgt_mask = make_tgt_mask(generated, pad_id)

        logits = model(src_ids, generated, src_mask=src_mask, tgt_mask=tgt_mask)  # (B, T, V)

        next_tok = (logits[:, -1] / 1.3).argmax(dim=-1, keepdim=True)  # (B,1)

        generated = torch.cat([generated, next_tok], dim=1)  # (B, T+1)

    B, T = generated.size()
    for b in range(B):
        for i in range(T):
            if generated[b, i].item() == eos_id:
                if i + 1 < T:
                    generated[b, i + 1:] = pad_id
                break

    return generated.tolist()

@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader,
    loss_fn: nn.Module,    
    tokenizer,
    pad_id: int,
    bos_id: int,
    eos_id: int,
    max_len: int,
    device: torch.device,
    *,
    bleu_on: bool = True,   
) -> Tuple[float, float]:
   
    model.eval()

    ce_sum = nn.CrossEntropyLoss(ignore_index=pad_id, reduction="sum")
    total_loss_sum = 0.0
    total_tokens = 0

    hyps: List[str] = []
    refs: List[str] = []

    for batch in tqdm(loader, desc="eval", leave=False):
        src_ids = batch["src_input_ids"].to(device)  # (B,S)
        tgt_ids = batch["tgt_input_ids"].to(device)  # (B,T)

        tgt_in = tgt_ids[:, :-1]    # (B,T-1)
        tgt_y  = tgt_ids[:, 1:]     # (B,T-1)

        src_mask = make_src_mask(src_ids, pad_id)     # (B,1,1,S)
        tgt_mask = make_tgt_mask(tgt_in, pad_id)      # (B,1,T-1,T-1)

        logits = model(src_ids, tgt_in, src_mask=src_mask, tgt_mask=tgt_mask)  # (B,T-1,V)
        V = logits.size(-1)

        loss_sum = ce_sum(logits.reshape(-1, V), tgt_y.reshape(-1))
        nonpad = (tgt_y != pad_id).sum().item()

        total_loss_sum += float(loss_sum.item())
        total_tokens += int(nonpad)

        pred_ids_list = None
        if bleu_on:
            pred_ids_list = greedy_decode(
                model=model,
                src_ids=src_ids,
                pad_id=pad_id,
                bos_id=bos_id,
                eos_id=eos_id,
                max_len=max_len,
                device=device,
            )

        B = src_ids.size(0)
        pred_texts = tokenizer.batch_decode(pred_ids_list, skip_special_tokens=True)
        ref_texts = tokenizer.batch_decode(tgt_ids, skip_special_tokens=True)

        hyps.extend([t.strip() for t in pred_texts])
        refs.extend([t.strip() for t in ref_texts])
        

    # ---- final metrics ----
    avg_nll = total_loss_sum / max(1, total_tokens)
    ppl = math.exp(avg_nll)

    if bleu_on and len(hyps) > 0:
        bleu = sacrebleu.corpus_bleu(hyps, [refs]).score
    else:
        bleu = 0.0
    
    # wandb.log({
    #     "BLEU Score" : bleu
    # })

    return ppl, bleu

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/de_en_bpe")
    parser.add_argument("--save_dir", type=str, default="ckpt")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    # run = wandb.init(
    # entity="dhfkzlfj-hanyang-university",
    # project = "tranformer"
    # )

    device = torch.device(args.device)
    print("CUDA available:", torch.cuda.is_available())

    d_model = 512
    d_ff = 2048
    num_heads = 8
    num_layers = 6
    dropout = 0.1
    max_len = 64

    batch_size = 32
    epochs = 5
    eval_every_steps = 2000
    warmup_steps = 4000

    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

    pad_id = 0
    bos_id = 101
    eos_id = 102
    vocab_size = len(tokenizer)

    ds = load_from_disk(args.data_dir) 
    train_ds = ds["train"]
    dev_ds = ds["dev"]
    test_ds = ds["test"]

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
    )
    dev_loader = DataLoader(
        dev_ds,
        batch_size=batch_size,
        shuffle=False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False
    )

    model = Transformer(
        src_vocab_size=vocab_size,
        tgt_vocab_size=vocab_size,
        max_len=max_len,
        num_layer=num_layers,
        d_model=d_model,
        num_head=num_heads,
        d_ff=d_ff,
        dropout=dropout,
    ).to(device)

    loss_fn = nn.CrossEntropyLoss(ignore_index=pad_id, label_smoothing=0.1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0, betas=(0.9, 0.98), eps=1e-9)
    scheduler = NoamLR(optimizer, d_model=d_model, warmup_steps=warmup_steps)

    os.makedirs(args.save_dir, exist_ok=True)

    best_ppl = float("inf")
    step = 0

    model.train()
    for epoch in range(1, epochs + 1):
        for batch in tqdm(train_loader, desc=f"train epoch {epoch}"):
            step += 1
            src_ids = batch["src_input_ids"].to(device)
            tgt_ids = batch["tgt_input_ids"].to(device)

            src_mask = make_src_mask(src_ids, pad_id)
            tgt_mask = make_tgt_mask(tgt_ids[:, :-1], pad_id)

            logits = model(src_ids, tgt_ids[:, :-1], src_mask=src_mask, tgt_mask=tgt_mask)  # (B,T,V)
            V = logits.size(-1)
            loss = loss_fn(logits.view(-1, V), tgt_ids[:, 1:].contiguous().view(-1))

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            lr = scheduler.step()

            if step % 200 == 0:
                print(f"[epoch {epoch} step {step}] lr={lr:.6f} loss={loss.item():.4f} ppl~={math.exp(loss.item()):.2f}")
                wandb.log({
                    "loss" : loss.item(),
                    "Perplexity" : math.exp(loss.item()),
                })

            if step % eval_every_steps == 0:
                ppl, bleu = evaluate(
                    model, dev_loader, loss_fn, tokenizer,
                    pad_id=pad_id, bos_id=bos_id, eos_id=eos_id,
                    max_len=max_len, device=device,
                )
                print(f"DEV: ppl={ppl:.2f} BLEU={bleu:.2f}")

                torch.save(
                    {"model": model.state_dict(), "opt": optimizer.state_dict(), "step": step},
                    os.path.join(args.save_dir, "last.pt"),
                )
                if ppl < best_ppl:
                    best_ppl = ppl
                    torch.save(
                        {"model": model.state_dict(), "opt": optimizer.state_dict(), "step": step},
                        os.path.join(args.save_dir, "best.pt"),
                    )
                model.train()
            
            

    print("Training done.")
    # wandb.finish()
    # test
    state_dict = torch.load("ckpt/best.pt", map_location=device)
    model.load_state_dict(state_dict["model"])
    model.eval()

    ppl, bleu = evaluate(
        model, test_loader, loss_fn, tokenizer,
        pad_id=pad_id, bos_id=bos_id, eos_id=eos_id,
        max_len=max_len, device=device,
    )

    print(f"\nTEST (best.pt): ppl={ppl:.2f} BLEU={bleu:.2f}")

    # wandb.finish()

if __name__ == "__main__":
    main()
