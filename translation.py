import argparse
import math
import torch
import torch.nn as nn

from Transformer_From_Scratch.model import Transformer  
from transformers import AutoTokenizer


def make_src_mask(src_ids: torch.Tensor, pad_id: int) -> torch.Tensor:
    mask = (src_ids != pad_id).unsqueeze(1).unsqueeze(2)
    return mask

def make_causal_mask(T: int, device: torch.device) -> torch.Tensor:
    return torch.triu(torch.ones(T, T, device=device, dtype=torch.bool), diagonal=1)[None, None, :, :]

def make_tgt_mask(tgt_ids: torch.Tensor, pad_id: int) -> torch.Tensor:
    mask = (tgt_ids != pad_id).unsqueeze(1).unsqueeze(2)
    seq_len = tgt_ids.size(1)
    look_ahead_mask = torch.tril(
        torch.ones(seq_len, seq_len, device=tgt_ids.device, dtype=torch.bool)
    ).unsqueeze(0).unsqueeze(0)

    mask = mask & look_ahead_mask
    return mask

@torch.no_grad()
def greedy_decode(
    model: nn.Module,
    src_ids: torch.Tensor,        
    pad_id: int,
    bos_id: int,
    eos_id: int,
    max_len: int,
    device: torch.device,
):
    model.eval()

    B, Ts = src_ids.size()

    generated = torch.full((B, 1), bos_id, dtype=torch.long, device=device)

    src_mask = make_src_mask(src_ids, pad_id)

    for _ in range(max_len - 1):
        tgt_mask = make_tgt_mask(generated, pad_id)

        logits = model(
            src_ids,
            generated,
            src_mask=src_mask,
            tgt_mask=tgt_mask,
        )  

        next_tok = (logits[:, -1] / 1.3).argmax(dim=-1, keepdim=True)  

        generated = torch.cat([generated, next_tok], dim=1) 

    B, T = generated.size()
    for b in range(B):
        for i in range(T):
            if generated[b, i].item() == eos_id:
                if i + 1 < T:
                    generated[b, i + 1:] = pad_id
                break

    return generated.tolist()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default="ckpt/best.pt")
 
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--d_ff", type=int, default=2048)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--num_layers", type=int, default=6)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max_len", type=int, default=128)

    args = parser.parse_args()
    device = torch.device(args.device)
    d_model = 512
    d_ff = 2048
    num_heads = 8
    num_layers = 6
    dropout = 0.1
    max_len = 64
    
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    pad_id = 0
    bos_id = 101
    eos_id = 102
    vocab_size = len(tokenizer)
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
    ckpt = torch.load(args.ckpt, map_location=device)
    state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    print("✅ Model loaded. Type a source sentence. (empty input to quit)")
    while True:
        src_text = input("\nSRC> ").strip()
        if not src_text:
            break

        enc = tokenizer(
            src_text,
            add_special_tokens=True,
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        )
        src_ids = enc["input_ids"].to(device)  # (1,S)

        pred = greedy_decode(
            model=model,
            src_ids=src_ids,
            pad_id=pad_id,
            bos_id=bos_id,
            eos_id=eos_id,
            max_len=max_len,
            device=device,
        ) 

        pred_ids = pred[0]

        if pred_ids and pred_ids[0] == bos_id:
            pred_ids = pred_ids[1:]

        if eos_id in pred_ids:
            pred_ids = pred_ids[: pred_ids.index(eos_id)]

        pred_ids = [t for t in pred_ids if t != pad_id]

        out_text = tokenizer.decode(pred_ids, skip_special_tokens=True) if pred_ids else ""
        print(f"TGT> {out_text}")


if __name__ == "__main__":
    main()