import os
import math
import argparse
from typing import Tuple, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from datasets import load_from_disk
from transformers import AutoTokenizer


from Transformer_From_Scratch.model import Transformer

def make_src_mask(src_ids: torch.Tensor, pad_id: int) -> torch.Tensor:
    mask = (src_ids != pad_id).unsqueeze(1).unsqueeze(2)
    # return (src_ids == pad_id)[:, None, None, :]
    return mask

def make_causal_mask(T: int, device: torch.device) -> torch.Tensor:
    # (1,1,T,T) True=mask (future positions)
    return torch.triu(torch.ones(T, T, device=device, dtype=torch.bool), diagonal=1)[None, None, :, :]

def make_tgt_mask(tgt_ids: torch.Tensor, pad_id: int) -> torch.Tensor:
    # (B,Tt) -> (B,1,Tt,Tt) True=mask
    # pad_mask = (tgt_ids == pad_id)[:, None, None, :]  # (B,1,1,Tt)
    mask = (tgt_ids != pad_id).unsqueeze(1).unsqueeze(2)
    seq_len = tgt_ids.size(1)
    # causal = make_causal_mask(tgt_ids.size(1), tgt_ids.device)  # (1,1,Tt,Tt)
    look_ahead_mask = torch.tril(
        torch.ones(seq_len, seq_len, device=tgt_ids.device, dtype=torch.bool)
    ).unsqueeze(0).unsqueeze(0)

    mask = mask & look_ahead_mask
    # return pad_mask | causal
    return mask

@torch.no_grad()
def beam_search_decode(
    model: nn.Module,
    src_ids: torch.Tensor,          # (B, Ts)
    pad_id: int,
    bos_id: int,
    eos_id: int,
    max_len: int,
    device: torch.device,
    beam_size: int = 4,
    length_penalty: float = 1.0,    # 1.0이면 거의 없음, 0.6~1.2 실험
    early_stop: bool = True,
):
    model.eval()
    B, Ts = src_ids.size()

    # source mask once
    src_mask = make_src_mask(src_ids, pad_id)

    results: List[List[int]] = []

    for b in range(B):
        src_b = src_ids[b:b+1]          # (1, Ts)
        src_mask_b = src_mask[b:b+1]    # (1,1,1,Ts)

        beams = [(torch.tensor([[bos_id]], device=device, dtype=torch.long), 0.0, False)]

        for _ in range(max_len - 1):
            all_candidates = []

            if early_stop and all(f for _, _, f in beams):
                break

            for seq, score, finished in beams:
                if finished:
              
                    all_candidates.append((seq, score, True))
                    continue

                tgt_mask = make_tgt_mask(seq, pad_id)  # (1,1,t,t)
                logits = model(src_b, seq, src_mask=src_mask_b, tgt_mask=tgt_mask)  # (1,t,V)
                log_probs = torch.log_softmax(logits[:, -1, :], dim=-1)  # (1,V)

                topk_logp, topk_ids = torch.topk(log_probs, k=beam_size, dim=-1)    # (1,k)

                for k in range(beam_size):
                    tok_id = topk_ids[0, k].item()
                    tok_logp = topk_logp[0, k].item()

                    new_seq = torch.cat(
                        [seq, torch.tensor([[tok_id]], device=device, dtype=torch.long)],
                        dim=1
                    )
                    new_score = score + tok_logp
                    new_finished = (tok_id == eos_id)
                    all_candidates.append((new_seq, new_score, new_finished))

    
            def norm_score(s: float, length: int) -> float:
                # length includes BOS; avoid div-by-zero
                lp = (length ** length_penalty) if length_penalty != 0 else 1.0
                return s / lp

            all_candidates.sort(key=lambda x: norm_score(x[1], x[0].size(1)), reverse=True)
            beams = all_candidates[:beam_size]

    
        finished_beams = [x for x in beams if x[2]]
        best = finished_beams[0] if len(finished_beams) > 0 else beams[0]
        best_seq = best[0].squeeze(0).tolist()  

     
        if len(best_seq) < max_len:
            best_seq = best_seq + [pad_id] * (max_len - len(best_seq))
        else:
            best_seq = best_seq[:max_len]

        for i in range(len(best_seq)):
            if best_seq[i] == eos_id:
                for j in range(i + 1, len(best_seq)):
                    best_seq[j] = pad_id
                break

        results.append(best_seq)

    return results

@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    tokenizer,
    pad_id: int,
    bos_id: int,
    eos_id: int,
    max_len: int,
    device: torch.device,
    *,
    bleu_on: bool = True,
    debug_print: int = 3,
) -> Tuple[float, float]:
   
    model.eval()

    ce_sum = nn.CrossEntropyLoss(ignore_index=pad_id, reduction="sum")

    total_loss_sum = 0.0
    total_tokens = 0

    hyps: List[str] = []
    refs: List[str] = []

    for i, batch in enumerate(tqdm(loader, desc="eval")):
        src_ids = batch["src_input_ids"].to(device)          # (B,Ts)
        tgt_ids = batch["tgt_input_ids"].to(device)          # (B,Tt)

        src_mask = make_src_mask(src_ids, pad_id)
        tgt_in = tgt_ids[:, :-1]
        tgt_y = tgt_ids[:, 1:].contiguous()

        tgt_mask = make_tgt_mask(tgt_in, pad_id)

        logits = model(src_ids, tgt_in, src_mask=src_mask, tgt_mask=tgt_mask)  # (B,T,V)
        V = logits.size(-1)

        loss_sum = ce_sum(logits.view(-1, V), tgt_y.view(-1))
        nonpad = (tgt_y != pad_id).sum().item()

        total_loss_sum += loss_sum.item()
        total_tokens += nonpad

        if bleu_on:
            # pred_ids = greedy_decode(
            #     model=model,
            #     src_ids=src_ids,
            #     pad_id=pad_id,
            #     bos_id=bos_id,
            #     eos_id=eos_id,
            #     max_len=max_len,
            #     device=device,
            # )  # (B, <=max_len)
            pred_ids = beam_search_decode(
                model=model,
                src_ids=src_ids,
                pad_id=pad_id,
                bos_id=bos_id,
                eos_id=eos_id,
                max_len=max_len,
                device=device,
                beam_size=4,           
                length_penalty=1.0,    
                early_stop=True,
            )

            pred_texts = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
            ref_texts = tokenizer.batch_decode(tgt_ids, skip_special_tokens=True)

            hyps.extend([t.strip() for t in pred_texts])
            refs.extend([t.strip() for t in ref_texts])

            if i < debug_print:
                for k in range(min(2, len(pred_texts))):
                    print("\n--- sample ---")
                    print("HYP:", pred_texts[k])
                    print("REF:", ref_texts[k])

    if total_tokens == 0:
        ppl = float("inf")
    else:
        avg_ce = total_loss_sum / total_tokens
        ppl = float(math.exp(avg_ce))

    bleu = 0.0
    if bleu_on:
        try:
            import sacrebleu
            bleu = float(sacrebleu.corpus_bleu(hyps, [refs]).score)
        except Exception:
            bleu = 0.0

    return ppl, bleu


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/de_en_bpe")
    parser.add_argument("--ckpt_path", type=str, default="ckpt/best.pt")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--d_ff", type=int, default=2048)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--num_layers", type=int, default=6)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max_len", type=int, default=64)

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--no_bleu", action="store_true")
    args = parser.parse_args()

    device = torch.device(args.device)
    print("CUDA available:", torch.cuda.is_available())
    print("Using device:", device)

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    pad_id = 0
    bos_id = 101
    eos_id = 102
    vocab_size = len(tokenizer)

    ds = load_from_disk(args.data_dir)
    if "test" not in ds:
        raise KeyError(f"No 'test' split in {args.data_dir}. Available: {list(ds.keys())}")
    test_ds = ds["test"]

    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    model = Transformer(
        src_vocab_size=vocab_size,
        tgt_vocab_size=vocab_size,
        max_len=args.max_len,
        num_layer=args.num_layers,
        d_model=args.d_model,
        num_head=args.num_heads,
        d_ff=args.d_ff,
        dropout=args.dropout,
    ).to(device)

    if not os.path.exists(args.ckpt_path):
        raise FileNotFoundError(f"ckpt not found: {args.ckpt_path}")

    ckpt = torch.load(args.ckpt_path, map_location=device)
    if "model" not in ckpt:
        raise KeyError(f"Checkpoint missing 'model' key: {args.ckpt_path}")

    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    loss_fn = nn.CrossEntropyLoss(ignore_index=pad_id)

    ppl, bleu = evaluate(
        model=model,
        loader=test_loader,
        loss_fn=loss_fn,
        tokenizer=tokenizer,
        pad_id=pad_id,
        bos_id=bos_id,
        eos_id=eos_id,
        max_len=args.max_len,
        device=device,
        bleu_on=(not args.no_bleu),
    )

    if args.no_bleu:
        print(f"\nTEST ({os.path.basename(args.ckpt_path)}): ppl={ppl:.2f}")
    else:
        print(f"\nTEST ({os.path.basename(args.ckpt_path)}): ppl={ppl:.2f} BLEU={bleu:.2f}")


if __name__ == "__main__":
    main()
