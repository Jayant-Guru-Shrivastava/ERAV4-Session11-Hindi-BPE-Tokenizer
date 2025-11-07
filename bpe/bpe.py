import json
from collections import Counter, defaultdict
from typing import List, Tuple, Dict

def _get_stats(ids: List[int]) -> Counter:
    """Count adjacent pair frequencies in `ids`."""
    counts = Counter()
    for a, b in zip(ids, ids[1:]):
        counts[(a, b)] += 1
    return counts

def _merge(ids: List[int], pair: Tuple[int,int], idx: int) -> List[int]:
    """Merge all non-overlapping occurrences of `pair` into `idx`."""
    a, b = pair
    out = []
    i = 0
    n = len(ids)
    while i < n:
        if i < n - 1 and ids[i] == a and ids[i+1] == b:
            out.append(idx)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out

class ByteLevelBPE:
    """
    Byte-level BPE:
    - base tokens: 0..255 (raw bytes)
    - merges learned on bytes -> new token IDs starting at 256 upward
    """
    def __init__(self):
        self.merges: Dict[Tuple[int,int], int] = {}
        self.vocab: Dict[int, bytes] = {}

    def train(self, corpus_bytes: bytes, vocab_size: int, progress=None):
        """
        Train merges to reach `vocab_size` (>=256).
        corpus_bytes: raw bytes from concatenated training text(s).
        """
        assert vocab_size >= 256, "vocab_size must be >= 256"
        ids = list(corpus_bytes)
        self.merges = {}
        # base vocab
        vocab = {i: bytes([i]) for i in range(256)}
        num_merges = vocab_size - 256
        for i in range(num_merges):
            stats = _get_stats(ids)
            if not stats:
                break
            # pick the most frequent pair
            pair = max(stats, key=stats.get)
            new_id = 256 + i
            ids = _merge(ids, pair, new_id)
            self.merges[pair] = new_id
            # construct new vocab entry
            a, b = pair
            vocab[new_id] = vocab[a] + vocab[b]
            if progress and i % max(1, num_merges // 50) == 0:
                progress(i, num_merges, pair, stats[pair])
        self.vocab = vocab

    def encode(self, text: str) -> List[int]:
        ids = list(text.encode('utf-8'))
        # greedy: pick the best-ranked pair present each round
        while len(ids) >= 2:
            stats = _get_stats(ids)
            if not stats:
                break
            # choose min by rank (merges order) via token id; unseen -> inf rank
            best_pair = min(stats, key=lambda p: self.merges.get(p, float('inf')))
            if best_pair not in self.merges:
                break
            new_id = self.merges[best_pair]
            ids = _merge(ids, best_pair, new_id)
        return ids

    def decode(self, ids: List[int], errors: str = "strict") -> str:
        # Precompute vocab if not present (for safety)
        if not self.vocab:
            vocab = {i: bytes([i]) for i in range(256)}
            for (a,b), idx in self.merges.items():
                vocab[idx] = vocab[a] + vocab[b]
            self.vocab = vocab

        out = bytearray()
        for t in ids:
            bs = self.vocab.get(t)
            if bs is None:
                # fallback: try to unmerge recursively
                flipped = {v: k for k, v in self.merges.items()}
                stack = [t]
                while stack:
                    x = stack.pop()
                    if x < 256:
                        out.append(x)
                    else:
                        a, b = flipped[x]
                        stack.append(b)
                        stack.append(a)
            else:
                out.extend(bs)
        return out.decode('utf-8', errors=errors)

    # Persistence
    def save(self, path_dir: str):
        merges_list = [[int(a), int(b), int(idx)] for (a,b), idx in self.merges.items()]
        with open(f"{path_dir}/merges.json", "w", encoding="utf-8") as f:
            json.dump({"merges": merges_list}, f, ensure_ascii=False, indent=2)
        # store vocab as int->hex bytes for portability
        vocab_dump = {int(k): self.vocab[k].hex() for k in self.vocab}
        with open(f"{path_dir}/vocab.json", "w", encoding="utf-8") as f:
            json.dump({"vocab_hex": vocab_dump}, f, ensure_ascii=False, indent=2)

    def load(self, path_dir: str):
        with open(f"{path_dir}/merges.json", "r", encoding="utf-8") as f:
            data = json.load(f)
        self.merges = {}
        for a, b, idx in data["merges"]:
            self.merges[(int(a), int(b))] = int(idx)
        # reconstruct vocab
        vocab = {i: bytes([i]) for i in range(256)}
        for (a,b), idx in self.merges.items():
            vocab[idx] = vocab[a] + vocab[b]
        self.vocab = vocab
