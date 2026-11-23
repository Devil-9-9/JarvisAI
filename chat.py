import argparse, torch
from pathlib import Path
from personal_ai.model import PersonalTransformer

def sample(logits, temperature=0.8, top_k=40):
    logits = logits / max(temperature, 1e-5)
    if top_k:
        values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits[logits < values[:, [-1]]] = -float("inf")
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)

@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="runs/best.pt")
    ap.add_argument("--prompt", default=None)
    ap.add_argument("--max-new-tokens", type=int, default=500)
    ap.add_argument("--temperature", type=float, default=.8)
    ap.add_argument("--top-k", type=int, default=40)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg, vocab = ckpt["config"], ckpt["vocab"]
    stoi = {c:i for i,c in enumerate(vocab)}
    prompt = args.prompt if args.prompt is not None else input("You: ")
    ids = [stoi[c] for c in prompt if c in stoi]
    if not ids: ids = [0]
    model = PersonalTransformer(len(vocab), cfg["block_size"], cfg["n_embd"], cfg["n_head"],
                                cfg["n_layer"], cfg["dropout"]).to(device)
    model.load_state_dict(ckpt["model"]); model.eval()
    x = torch.tensor([ids[-cfg["block_size"]:]], dtype=torch.long, device=device)
    for _ in range(args.max_new_tokens):
        logits, _ = model(x[:, -cfg["block_size"]:])
        nxt = sample(logits[:, -1, :], args.temperature, args.top_k)
        x = torch.cat([x, nxt], dim=1)
    print("AI:", "".join(vocab[i] for i in x[0].tolist()))

if __name__ == "__main__": main()
