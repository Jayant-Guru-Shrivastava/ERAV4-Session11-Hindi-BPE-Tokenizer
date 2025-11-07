import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import argparse, glob, os, json
from bpe.bpe import ByteLevelBPE

def read_glob_text(pattern: str) -> str:
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise SystemExit(f"No files matched: {pattern}")
    parts = []
    for p in paths:
        with open(p, "r", encoding="utf-8") as f:
            parts.append(f.read())
    return "\n".join(parts)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_glob", required=True, help='e.g. "data/*.txt"')
    ap.add_argument("--artifacts-dir", default="artifacts")
    ap.add_argument("--errors", default="strict", choices=["strict","replace","ignore"])
    args = ap.parse_args()

    bpe = ByteLevelBPE()
    bpe.load(args.artifacts_dir)

    text = read_glob_text(args.input_glob)
    utf8 = text.encode("utf-8")
    ids = bpe.encode(text)
    decoded = bpe.decode(ids, errors=args.errors)

    cr = len(utf8) / max(1, len(ids))
    print(f"Tokens in vocab: {len(bpe.vocab)}")
    print(f"Compression ratio: {cr:.4f}")
    ok = (cr >= 3.2) and (len(bpe.vocab) > 5000)
    print(f"Meets target? {'YES' if ok else 'NO'}")
    if text[:200] != decoded[:200]:
        print("Warning: Decoded preview differs (likely due to errors mode).")

if __name__ == "__main__":
    main()
