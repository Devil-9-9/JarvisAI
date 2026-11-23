import argparse, json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from personal_ai.model import PersonalTransformer
from personal_ai.data import TokenDataset

@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="runs/best.pt")
    ap.add_argument("--data", default="data/processed")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--batches", type=int, default=0, help="0 = evaluate every test window")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg, vocab = ckpt["config"], ckpt["vocab"]
    stoi = {c:i for i,c in enumerate(vocab)}
    manifest = json.loads((Path(args.data)/"manifest.json").read_text())
    docs = []
    for name in manifest["test"]:
        docs += json.loads((Path(args.data)/name).read_text())
    ids = [stoi[c] for c in "\n\n".join(docs) if c in stoi]
    ds = TokenDataset(ids, cfg["block_size"])
    if not len(ds): raise SystemExit("Test split too small for this model context length.")
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, drop_last=False)
    model = PersonalTransformer(len(vocab), cfg["block_size"], cfg["n_embd"], cfg["n_head"],
                                cfg["n_layer"], cfg["dropout"]).to(device)
    model.load_state_dict(ckpt["model"]); model.eval()
    losses, count = 0.0, 0
    for i, (x,y) in enumerate(loader):
        if args.batches and i >= args.batches: break
        _, loss = model(x.to(device), y.to(device))
        n = x.size(0); losses += loss.item()*n; count += n
    test_loss = losses/count
    result = {"test_loss": test_loss, "test_perplexity": float(torch.exp(torch.tensor(test_loss))),
              "test_windows": count, "checkpoint_step": ckpt["step"],
              "validation_loss_at_selection": ckpt["validation_loss"]}
    Path("runs").mkdir(exist_ok=True)
    Path("runs/test_metrics.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))

if __name__ == "__main__": main()
