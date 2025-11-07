from tokenizers import Tokenizer
import gradio as gr

TOK_PATH = "artifacts/tokenizer.json"

tok = Tokenizer.from_file(TOK_PATH)

def encode_decode(text, errors):
    # bytes and tokens for CR
    utf8 = text.encode("utf-8", errors=errors)
    enc = tok.encode(text)
    cr = len(utf8) / max(1, len(enc.ids))
    return enc.ids, tok.decode(enc.ids), f"Vocab: {tok.get_vocab_size()} | Tokens: {len(enc.ids)} | CR: {cr:.4f}"

with gr.Blocks() as demo:
    gr.Markdown("# Hindi Byte-Level BPE Tokenizer\nUpload: IITB Hindi | Vocab≈8192 | CR≥3.2")
    with gr.Row():
        errors = gr.Radio(["strict","replace","ignore"], value="ignore", label="UTF-8 decode errors")
    txt = gr.Textbox(label="Input text", value="नमस्ते! यह एक हिंदी वाक्य है। Hello Mumbai 2025.", lines=4)
    btn = gr.Button("Encode → Decode")
    ids = gr.JSON(label="Token IDs")
    dec = gr.Textbox(label="Decoded", lines=4)
    stats = gr.Markdown()
    btn.click(encode_decode, [txt, errors], [ids, dec, stats])

if __name__ == "__main__":
    demo.launch()
