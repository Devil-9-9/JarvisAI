"""Clean text, split by whole source documents, and build tokenizer from train only."""
from pathlib import Path
import argparse, json, random, re, sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from personal_ai.data import CharTokenizer

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"

def clean_text(text):
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Remove common Project Gutenberg header/footer while retaining the story body.
    start = re.search(r"\*\*\*\s*START OF (?:THE|THIS) PROJECT GUTENBERG", text, re.I)
    end = re.search(r"\*\*\*\s*END OF (?:THE|THIS) PROJECT GUTENBERG", text, re.I)
    if start and end and end.start() > start.end():
        text = text[start.end():end.start()]
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--train-ratio", type=float, default=.80)
    ap.add_argument("--val-ratio", type=float, default=.10)
    args = ap.parse_args()
    files = sorted(RAW.glob("*.txt"))
    if len(files) < 3:
        raise SystemExit("Need at least 3 .txt source documents. Run download_corpus.py or add your own.")
    docs = []
    for f in files:
        txt = clean_text(f.read_text(encoding="utf-8", errors="ignore"))
        if len(txt) >= 2000:
            docs.append((f.name, txt))
    if len(docs) < 3:
        raise SystemExit("At least 3 usable documents (2,000+ chars each) are required.")
    random.Random(args.seed).shuffle(docs)
    n = len(docs)
    n_train = max(1, int(n * args.train_ratio))
    n_val = max(1, int(n * args.val_ratio))
    if n_train + n_val >= n:
        n_train, n_val = n-2, 1
    grouped = {
        "train": docs[:n_train],
        "validation": docs[n_train:n_train+n_val],
        "test": docs[n_train+n_val:]
    }
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for split, items in grouped.items():
        encoded_name = f"{split}.json"
        (OUT / encoded_name).write_text(
            json.dumps([txt for _, txt in items], ensure_ascii=False), encoding="utf-8"
        )
        manifest[split] = [encoded_name]
    manifest["sources"] = {split: [name for name, _ in items] for split, items in grouped.items()}
    manifest["split_method"] = "source-document-level; no source document crosses splits"
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    tokenizer = CharTokenizer.from_training_texts([txt for _, txt in grouped["train"]])
    tokenizer.save(OUT / "tokenizer.json")
    print("Documents by split:", {k: len(v) for k, v in grouped.items()})
    print("Characters in tokenizer (train only):", len(tokenizer.chars))
    print("Manifest:", OUT / "manifest.json")

if __name__ == "__main__":
    main()
