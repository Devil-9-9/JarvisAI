import argparse, json, math, random, time, sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from personal_ai.data import CharTokenizer, TokenDataset, load_splits
from personal_ai.model import PersonalTransformer

ROOT = Path(__file__).resolve().parent

def set_seed(seed):
    random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

@torch.no_grad()
def estimate_loss(model, loaders, device, batches):
    model.eval()
    out = {}
    for name, loader in loaders.items():
        vals = []
        it = iter(loader)
        for _ in range(batches):
            try: x, y = next(it)
            except StopIteration:
                it = iter(loader)
                try: x, y = next(it)
                except StopIteration: break
            x, y = x.to(device), y.to(device)
            _, loss = model(x, y)
            vals.append(loss.item())
        out[name] = sum(vals) / max(1, len(vals))
    model.train()
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/small.json")
    ap.add_argument("--data", default="data/processed")
    ap.add_argument("--out", default="runs")
    args = ap.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    set_seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    data_dir, out_dir = Path(args.data), Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = CharTokenizer.load(data_dir / "tokenizer.json")
    splits = load_splits(data_dir)
    encoded = {k: tokenizer.encode("\n\n".join(v)) for k, v in splits.items()}
    datasets = {k: TokenDataset(v, cfg["block_size"]) for k, v in encoded.items()}
    for k, ds in datasets.items():
        if len(ds) < 1: raise SystemExit(f"{k} split is too small for block_size={cfg['block_size']}")
    loaders = {k: DataLoader(ds, batch_size=cfg["batch_size"], shuffle=(k=="train"),
                             num_workers=cfg.get("num_workers", 0), drop_last=True)
               for k, ds in datasets.items()}
    model = PersonalTransformer(len(tokenizer.chars), cfg["block_size"], cfg["n_embd"],
                                cfg["n_head"], cfg["n_layer"], cfg["dropout"]).to(device)
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad: continue
        (no_decay if p.ndim < 2 or "ln" in name else decay).append(p)
    optimizer = torch.optim.AdamW(
        [{"params": decay, "weight_decay": cfg["weight_decay"]},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=cfg["learning_rate"], betas=(0.9, 0.95)
    )
    use_amp = device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    train_iter = iter(loaders["train"])
    best_val, stale = float("inf"), 0
    history_path = out_dir / "history.jsonl"
    history_path.write_text("")
    start = time.time()
    for step in range(1, cfg["max_steps"] + 1):
        if step <= cfg["warmup_steps"]:
            lr = cfg["learning_rate"] * step / max(1, cfg["warmup_steps"])
        else:
            progress = min(1.0, (step-cfg["warmup_steps"]) /
                           max(1, cfg["max_steps"]-cfg["warmup_steps"]))
            lr = cfg["min_learning_rate"] + .5*(cfg["learning_rate"]-cfg["min_learning_rate"])*(1+math.cos(math.pi*progress))
        for group in optimizer.param_groups: group["lr"] = lr
        try: x, y = next(train_iter)
        except StopIteration:
            train_iter = iter(loaders["train"]); x, y = next(train_iter)
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device, dtype=torch.float16, enabled=use_amp):
            _, loss = model(x, y)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
        scaler.step(optimizer); scaler.update()
        if step % cfg["eval_interval"] == 0 or step == 1:
            metrics = estimate_loss(model, {"train": loaders["train"], "validation": loaders["validation"]},
                                    device, cfg["eval_batches"])
            row = {"step": step, "train_loss": metrics["train"], "validation_loss": metrics["validation"], "lr": lr}
            with history_path.open("a") as f: f.write(json.dumps(row) + "\n")
            print(f"step {step:6d} | train loss {row['train_loss']:.4f} | "
                  f"validation loss {row['validation_loss']:.4f} | lr {lr:.2e}", flush=True)
            payload = {"model": model.state_dict(), "config": cfg,
                       "vocab": tokenizer.chars, "step": step, "validation_loss": row["validation_loss"]}
            torch.save(payload, out_dir / "last.pt")
            if row["validation_loss"] < best_val - cfg["min_delta"]:
                best_val, stale = row["validation_loss"], 0
                torch.save(payload, out_dir / "best.pt")
            else:
                stale += 1
                if stale >= cfg["patience"]:
                    print(f"Early stopping: validation loss did not improve for {stale} evaluations.")
                    break
    print(f"Training finished in {(time.time()-start)/60:.1f} minutes. Best validation loss: {best_val:.4f}")
    print("Best checkpoint:", out_dir / "best.pt")

if __name__ == "__main__":
    main()
