# scripts/clean_hindi.py
import re, sys, io, pathlib

inp = pathlib.Path(sys.argv[1])
out = pathlib.Path(sys.argv[2])

dev = re.compile(r'[\u0900-\u097F]')
url = re.compile(r'https?://\S+|www\.\S+')
email = re.compile(r'\S+@\S+')
ascii_word = re.compile(r'[A-Za-z]{3,}')

def devanagari_ratio(s):
    if not s: return 0.0
    d = len(dev.findall(s))
    return d / len(s)

with inp.open('r', encoding='utf-8', errors='ignore') as fin, \
     out.open('w', encoding='utf-8') as fout:
    for line in fin:
        line = url.sub('', line)
        line = email.sub('', line)
        # drop line if it’s mostly non-Devanagari OR has many ASCII words
        if devanagari_ratio(line) < 0.6: 
            continue
        if len(ascii_word.findall(line)) > 3:
            continue
        fout.write(line)
