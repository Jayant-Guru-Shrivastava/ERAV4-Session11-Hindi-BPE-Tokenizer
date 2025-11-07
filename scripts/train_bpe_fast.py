# scripts/train_bpe_fast.py
import argparse
from pathlib import Path
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, processors
from tqdm import tqdm
import io

def iter_lines_with_progress(path: Path):
    # 1) Get total bytes for a byte-accurate progress bar
    total_bytes = path.stat().st_size
    processed = 0
    with path.open("rb") as f, tqdm(total=total_bytes, unit="B", unit_scale=True, desc="Training data read") as pbar:
        buf = io.BufferedReader(f, buffer_size=1024*1024)
        for raw in buf:
            processed += len(raw)
            pbar.update(len(raw))
            # decode line safely; skip un-decodable bytes
            line = raw.decode("utf-8", errors="ignore").strip()
            if line:
                yield line

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_file", required=True, help="Small slice file, e.g., data/hindi_train_10mb.txt")
    ap.add_argument("--vocab_size", type=int, default=8192)
    ap.add_argument("--out", default="artifacts/tokenizer.json")
    args = ap.parse_args()

    inp = Path(args.input_file)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    # Byte-level BPE
    tok = Tokenizer(models.BPE(unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=args.vocab_size, min_frequency=2, special_tokens=["[UNK]"])

    # 2) Train with visible progress over lines (proxy for % completion)
    lines_iter = iter_lines_with_progress(inp)
    tok.train_from_iterator(lines_iter, trainer=trainer, length=None)

    # keep byte-level reversibility
    tok.post_processor = processors.ByteLevel()

    tok.save(args.out)
    print(f"\nSaved tokenizer to {args.out}")
    print(f"Vocab size: {tok.get_vocab_size()}")

if __name__ == "__main__":
    main()
