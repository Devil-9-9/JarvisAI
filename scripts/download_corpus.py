"""Download public-domain plain-text books. This is a dataset download, not an AI API."""
from pathlib import Path
from urllib.request import Request, urlopen
import time

OUT = Path(__file__).resolve().parents[1] / "data" / "raw"
OUT.mkdir(parents=True, exist_ok=True)
BOOKS = {
    "sherlock_holmes.txt": "https://www.gutenberg.org/cache/epub/1661/pg1661.txt",
    "alice_in_wonderland.txt": "https://www.gutenberg.org/cache/epub/11/pg11.txt",
    "frankenstein.txt": "https://www.gutenberg.org/cache/epub/84/pg84.txt",
    "the_time_machine.txt": "https://www.gutenberg.org/cache/epub/35/pg35.txt",
    "pride_and_prejudice.txt": "https://www.gutenberg.org/cache/epub/1342/pg1342.txt"
}
for filename, url in BOOKS.items():
    target = OUT / filename
    if target.exists() and target.stat().st_size > 10000:
        print(f"Exists: {target.name}")
        continue
    print(f"Downloading {filename} ...")
    req = Request(url, headers={"User-Agent": "PersonalAIResearch/1.0"})
    with urlopen(req, timeout=45) as response:
        content = response.read()
    target.write_bytes(content)
    print(f"Saved {target.name}: {len(content):,} bytes")
    time.sleep(1)
print(f"Corpus directory: {OUT}")
