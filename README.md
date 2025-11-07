# Byte-Level BPE Tokenizer – Hindi (or Any Text)

This repo trains **your own byte-level BPE** tokenizer for an Indian language (default: Hindi) and checks that it meets the assignment targets:

- **Vocabulary size:** configurable (default: 8000+ tokens; base 256 bytes + merges)  
- **Compression Ratio (CR):** target **≥ 3.2**

> **Note on the prompt ambiguity:** The lecture text mentions both “more than 5000 tokens” and “less than 5000”. Here we follow the stricter interpretation: **> 5000**. You can set `--vocab-size` accordingly (e.g., 8192).

## Quick Start

### 1) Prepare a corpus
Put one or more UTF-8 text files under `data/` (e.g., Hindi Wikipedia dumps, OSCAR Hindi, news articles, books).  
Example:
```
data/
  hindi1.txt
  hindi2.txt
```

### 2) Install
```
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3) Train
```
python scripts/train_bpe.py \
  --input_glob "data/*.txt" \
  --vocab-size 8192 \
  --save-dir artifacts
```

### 4) Evaluate (Compression Ratio, token count)
```
python scripts/evaluate.py \
  --input_glob "data/*.txt" \
  --artifacts-dir artifacts
```

It prints something like:
```
Tokens in vocab: 8192
Compression ratio: 3.45
```

### 5) Hugging Face Spaces
Use the `hf_space/app.py` Gradio app. After training, copy the `artifacts/` folder (with `merges.json` and `vocab.json`) into your Space repo, then run:
```
pip install -r requirements.txt
python hf_space/app.py
```
On Spaces, set **`app.py`** as the entrypoint (or copy its content to the Space’s `app.py`) and include `requirements.txt`.

---

## Files

- `bpe/bpe.py` — Core BPE: train, encode (greedy), decode, save/load.
- `scripts/train_bpe.py` — CLI to train and save artifacts.
- `scripts/evaluate.py` — CLI to compute compression ratio and print summary.
- `hf_space/app.py` — Gradio demo for encoding/decoding and metrics.
- `notebooks/train_bpe.ipynb` — Minimal notebook to run everything end-to-end.
- `artifacts/` — Saved `merges.json` and `vocab.json` after training (created by the scripts).
- `data/` — Put your Hindi corpus here.

## Compression Ratio (definition used)

We report:
```
CR = total_utf8_bytes / total_token_count
```
Higher is better. Ensure CR ≥ 3.2 per assignment.

## Repro Tips

- Use NFC normalization of input text if your corpus mixes sources.
- Increase `--vocab-size` (e.g., 16k or 32k) to generally improve CR; watch for diminishing returns.
- If CR is below 3.2, try a larger corpus and more merges.

## README Requirements (for your submission)

Once trained, **edit below with your results**:

- **Final Vocabulary Size:** `<fill after training, e.g., 8192>`  
- **Compression Ratio:** `<fill after evaluate.py, e.g., 3.31>`  
- **Language/Corpus:** `<Hindi: source(s) you used>`  
- **GitHub link:** `<your repo>`  
- **HF Space link:** `<your Space>`

---

MIT License
