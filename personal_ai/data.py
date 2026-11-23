import json, random
from pathlib import Path
import torch
from torch.utils.data import Dataset

class CharTokenizer:
    """A transparent character tokenizer learned only from training documents."""
    def __init__(self, chars):
        self.chars = list(chars)
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}
    @classmethod
    def from_training_texts(cls, texts):
        return cls(sorted(set("".join(texts))))
    def encode(self, text):
        # Unknown characters are skipped rather than mapped to a fake answer token.
        return [self.stoi[c] for c in text if c in self.stoi]
    def decode(self, ids):
        return "".join(self.itos[int(i)] for i in ids if int(i) in self.itos)
    def save(self, path):
        Path(path).write_text(json.dumps({"chars": self.chars}, ensure_ascii=False), encoding="utf-8")
    @classmethod
    def load(cls, path):
        return cls(json.loads(Path(path).read_text(encoding="utf-8"))["chars"])

class TokenDataset(Dataset):
    def __init__(self, token_ids, block_size):
        self.data = torch.tensor(token_ids, dtype=torch.long)
        self.block_size = block_size
    def __len__(self):
        return max(0, len(self.data) - self.block_size - 1)
    def __getitem__(self, index):
        x = self.data[index:index+self.block_size]
        y = self.data[index+1:index+self.block_size+1]
        return x, y

def load_splits(processed_dir):
    p = Path(processed_dir)
    manifest = json.loads((p / "manifest.json").read_text(encoding="utf-8"))
    splits = {}
    for name in ("train", "validation", "test"):
        chunks = []
        for fname in manifest[name]:
            chunks.extend(json.loads((p / fname).read_text(encoding="utf-8")))
        splits[name] = chunks
    return splits
