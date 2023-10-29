from flask import Flask, render_template, request, jsonify
import ctranslate2
import sentencepiece as spm

app = Flask(__name__)

# [Modify] Set paths to the CTranslate2 and SentencePiece models
ct_model_path = "nllb-200-distilled-600M-int8"  # Replace with your model path
sp_model_path = "flores200_sacrebleu_tokenizer_spm.model"  # Replace with your model path

device = "cpu"  # or "cuda"

# Load the source SentencePiece model
sp = spm.SentencePieceProcessor()
sp.load(sp_model_path)

src_lang = "fra_Latn"
tgt_lang = "eng_Latn"

beam_size = 6

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/translate', methods=['POST'])
def translate():
    source_text = request.form['source_text']

    source_sentences = [source_text]
    target_prefix = [[tgt_lang]] * len(source_sentences)

    # Subword the source sentences
    source_sents_subworded = sp.encode_as_pieces(source_sentences)
    source_sents_subworded = [[src_lang] + sent + ["</s>"] for sent in source_sents_subworded]

    # Translate the source sentences
    translator = ctranslate2.Translator(ct_model_path, device=device)
    translations_subworded = translator.translate_batch(source_sents_subworded, batch_type="tokens", max_batch_size=2024, beam_size=beam_size, target_prefix=target_prefix)
    translations_subworded = [translation.hypotheses[0] for translation in translations_subworded]
    
    for translation in translations_subworded:
        if tgt_lang in translation:
            translation.remove(tgt_lang)

    # Desubword the target sentences
    translations = sp.decode(translations_subworded)

    return jsonify({'translation': translations[0]})

if __name__ == '__main__':
    app.run(debug=True)
