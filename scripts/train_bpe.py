import argparse, glob, os
from tqdm import tqdm
from bpe.bpe import ByteLevelBPE

def read_corpus_bytes(pattern: str) -> bytes:
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise SystemExit(f"No files matched: {pattern}")
    chunks = []
    for p in paths:
        with open(p, "rb") as f:
            chunks.append(f.read())
    return b"\n".join(chunks)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_glob", required=True, help='e.g. "data/*.txt"')
    ap.add_argument("--vocab-size", type=int, default=8192)
    ap.add_argument("--save-dir", default="artifacts")
    args = ap.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    corpus = read_corpus_bytes(args.input_glob)

    bpe = ByteLevelBPE()
    def prog(i, total, pair, freq):
        if i == 0 or i == total-1 or i % max(1,total//20)==0:
            print(f"[merge {i+1}/{total}] pair={pair} freq={freq}")
    bpe.train(corpus, vocab_size=args.vocab_size, progress=prog)
    bpe.save(args.save_dir)
    print(f"Saved artifacts to: {args.save_dir}")

if __name__ == "__main__":
    main()
