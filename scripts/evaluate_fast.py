# scripts/evaluate_fast.py
import argparse, io
from pathlib import Path
from tokenizers import Tokenizer
from tqdm import tqdm

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_file", required=True, help="Full file to evaluate, e.g., data/hindi_iitb.txt")
    ap.add_argument("--tokenizer", default="artifacts/tokenizer.json")
    ap.add_argument("--errors", default="ignore", choices=["ignore","strict","replace"])
    args = ap.parse_args()

    tok = Tokenizer.from_file(args.tokenizer)

    total_bytes = Path(args.input_file).stat().st_size
    total_utf8_bytes = 0
    total_tokens = 0

    with open(args.input_file, "rb") as f, tqdm(total=total_bytes, unit="B", unit_scale=True, desc="Evaluating CR") as pbar:
        buf = io.BufferedReader(f, buffer_size=1024*1024)
        for raw in buf:
            pbar.update(len(raw))
            text = raw.decode("utf-8", errors=args.errors)
            if not text:
                continue
            enc = tok.encode(text)
            total_utf8_bytes += len(text.encode("utf-8"))
            total_tokens += len(enc.ids)

    cr = total_utf8_bytes / max(1, total_tokens)
    print(f"\nVocab size: {tok.get_vocab_size()}")
    print(f"Compression Ratio (bytes/tokens): {cr:.4f}")
    print("Target met? ", "YES" if cr >= 3.2 and tok.get_vocab_size() > 5000 else "NO")

if __name__ == "__main__":
    main()
