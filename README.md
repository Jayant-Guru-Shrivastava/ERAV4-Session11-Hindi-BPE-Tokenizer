# Hindi Byte‑Level BPE Tokenizer

A fast **Byte‑Level BPE** tokenizer trained on the **IIT Bombay (IITB) Hindi corpus** (Hindi side).  
Built with the Rust‑backed `tokenizers` library for speed and reproducibility.

- **Vocab size:** **8192**
- **Compression Ratio (held‑out 10%):** **3.8694** (bytes / tokens)
- **Training split:** 90% of ~311 MB (≈293 MB)
- **Evaluation split:** 10% (≈32.6 MB)
- **Hardware:** Apple M1 (CPU), `tokenizers` Rust backend

> We use **GPT‑style Byte‑Level BPE**: merges are learned over raw bytes (0–255).  
> This is robust to any script, punctuation, or emoji and is losslessly reversible.

---

## Why Byte‑Level BPE (and not wordpiece / char BPE)?

- **Script‑agnostic & lossless** – handles Devanagari, mixed Hindi–English, punctuation, emoji.  
- **Strong compression** – frequent Hindi patterns (`में`, `कर`, `भारत`, …) become compact tokens.  
- **Matches modern LLMs** – GPT‑2/3/4 use byte‑level BPE under the hood.

---

## Results

| Metric | Value |
|---|---|
| Vocab size | **8192** |
| Compression Ratio (test 10%) | **3.8694** |
| Targets met? | **Yes** (Vocab \> 5000, CR ≥ 3.2) |

> For rigor we report CR on a **held‑out 10%** split. Reporting CR on train text is also common for compression but can be slightly optimistic.

---

## Repository Structure

```
.
├── artifacts/
│   └── tokenizer.json            # final trained tokenizer (tracked)
├── bpe/
│   ├── __init__.py
│   └── bpe.py                    # reference byte‑level BPE (pure Python)
├── hf_space/
│   └── app.py                    # Gradio demo (encode/decode + CR)
├── notebooks/
│   └── train_bpe.ipynb           # optional, for transparency/repro
├── scripts/
│   ├── clean_hindi.py            # optional cleaner (URLs/emails/ASCII‑heavy lines)
│   ├── train_bpe_fast.py         # Rust‑fast training (Hugging Face tokenizers)
│   └── evaluate_fast.py          # streaming CR evaluation with progress
├── requirements.txt
└── README.md
```

> Large raw corpora should **not** be committed to git. Keep small samples only if needed.

---

## Reproduce Locally

### 1) Environment
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

### 2) Prepare data
Place your corpus under `data/` as UTF‑8 text.  
(We used the Hindi side of the IITB English–Hindi parallel corpus.)

**Optional:** make a small 10 MB slice and (optionally) clean it.
```bash
dd if=data/hindi_iitb.txt of=data/hindi_train_10mb.txt bs=1m count=10
python scripts/clean_hindi.py data/hindi_train_10mb.txt data/hindi_train_10mb.clean.txt
```

### 3) Train (90/10 split example)
```bash
# 90/10 split by bytes (adjust sizes as appropriate)
head -c $((311*1024*1024*9/10)) data/hindi_iitb.txt > data/train_90.txt
tail -c $((311*1024*1024/10))   data/hindi_iitb.txt > data/test_10.txt

# Train tokenizer (byte‑level BPE, progress shown)
python scripts/train_bpe_fast.py \
  --input_file data/train_90.txt \
  --vocab_size 8192 \
  --out artifacts/tokenizer.json
```

### 4) Evaluate on held‑out split
```bash
python scripts/evaluate_fast.py \
  --input_file data/test_10.txt \
  --tokenizer artifacts/tokenizer.json
```
Expected output (≈ values):
```
Vocab size: 8192
Compression Ratio (bytes/tokens): ~3.87
Target met?  YES
```
<img width="675" height="373" alt="image" src="https://github.com/user-attachments/assets/47c32f49-dbb1-434e-a1f1-578019e98afa" />

---

## Run the Demo (locally)

```bash
python hf_space/app.py
```
Open the Gradio URL printed in the console and try Hindi sentences.  
The UI shows token IDs, decoded text, and compression ratio.

> If you ever see odd glyphs in raw vocab strings (`Ġ`, `à¤…`), that’s normal for **byte‑level** storage.  
> Round‑trip via `encode`→`decode` remains exact.

---

## Deploy to Hugging Face Spaces

1. Create a **Gradio** Space (CPU Basic).  
2. Commit:
   - `hf_space/app.py`
   - `artifacts/tokenizer.json`
   - `requirements.txt`
3. Push – the Space will build automatically.

**requirements.txt**
```
gradio>=4.0.0
tokenizers
tqdm
```

---

## Notes / Tips

- `clean_hindi.py` is conservative (drops lines with Devanagari ratio \< 0.6 or many ASCII words).  
  If vocab coverage suffers, lower the threshold (e.g., 0.3) or skip cleaning.  
- Keep `min_frequency` in the trainer small to ensure the target vocab is reachable.  
- For even higher CR you can: train on a bit more data (e.g., 20–50 MB) or try a larger vocab (12k–16k).  
  Expect diminishing returns after that.

---

## License

MIT
