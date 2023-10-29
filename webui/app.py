import os
from flask import Flask, request, render_template, jsonify
from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI

app = Flask(__name__)
os.environ["OPENAI_API_KEY"] = "sk-xxx"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/query', methods=['POST'])
def query():
    pdf_file = request.files['pdf']
    query = request.form['query']
    pdfreader = PdfReader(pdf_file)

    raw_text = ''
    for i, page in enumerate(pdfreader.pages):
        content = page.extract_text()
        if content:
            raw_text += content

    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=800,
        chunk_overlap=200,
        length_function=len,
    )
    texts = text_splitter.split_text(raw_text)

    embeddings = OpenAIEmbeddings()

    document_search = FAISS.from_texts(texts, embeddings)

    chain = load_qa_chain(OpenAI(), chain_type="stuff")

    docs = document_search.similarity_search(query)
    result = chain.run(input_documents=docs, question=query)

    return jsonify({'answer': result})

if __name__ == '__main__':
    app.run(debug=True)
