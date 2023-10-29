import streamlit as st
import re
import io
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
from googletrans import Translator, LANGUAGES as googletrans_languages
from streamlit_extras.add_vertical_space import add_vertical_space
import pickle
import os
from dotenv import load_dotenv
from os import environ
import openai
from pandas import DataFrame
from keybert import KeyBERT
import seaborn as sns
import ctranslate2
import sentencepiece as spm
import base64  # Add missing import

# Set the background color and text color
st.markdown(
    """
    <style>
    body {
        background-color: #000;
        color: #FFF;
    }
    .sidebar {
        background-color: #000;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Create a left sidebar
st.sidebar.title("Options")

# Add options for different features
selected_option = st.sidebar.radio(
    "Select a Feature",
    ["Translation", "Querying", "Citation", "Summary", "Keyword Extraction", "ChatBot"],
)


# Function to extract text from a PDF file
def extract_text_from_pdf(pdf_file):
    text = ""
    pdf_reader = PdfReader(pdf_file)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text


if selected_option == "Citation":
    st.title("Citation Feature")

    # File upload for the input PDF
    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

    if uploaded_file is not None:
        # Read the uploaded PDF file and extract text
        st.text("Extracting text from the PDF...")
        with io.BytesIO(uploaded_file.read()) as pdf_file:
            legal_text = extract_text_from_pdf(pdf_file)

        # Define a regular expression pattern to match case citations
        case_citation_pattern = (
            r"\b[A-Z][\w\s.]+v\. [A-Z][\w\s.]+\s\d+\s[A-Z]+[a-zA-Z]+\s\d+"
        )

        # Define a regular expression pattern to match Indian legal citation (without dates)
        indian_legal_citation_pattern = r"\b\d+\s[A-Z]+[a-zA-Z]+\s\d+"

        # Define a regular expression pattern to match Constitution of India citations
        constitution_citation_pattern = r"The Constitution of India, Article \d+"

        # Compile regular expressions
        case_citation_regex = re.compile(case_citation_pattern)
        indian_legal_citation_regex = re.compile(indian_legal_citation_pattern)
        constitution_citation_regex = re.compile(constitution_citation_pattern)

        # Extract case citations
        case_citations = case_citation_regex.findall(legal_text)

        # Extract Indian legal citations (without dates)
        indian_legal_citations = indian_legal_citation_regex.findall(legal_text)

        # Extract Constitution of India citations
        constitution_citations = constitution_citation_regex.findall(legal_text)

        # Display the extracted citations
        st.header("Extracted Citations:")
        st.subheader("Case Citations:")
        for citation in case_citations:
            st.write(citation)

        st.subheader("Indian Legal Citations:")
        for citation in indian_legal_citations:
            st.write(citation)

        st.subheader("Constitution of India Citations:")
        for citation in constitution_citations:
            st.write(citation)
ct_model_path = "nllb-200-distilled-600M-int8"  # Replace with your model path
sp_model_path = (
    "flores200_sacrebleu_tokenizer_spm.model"  # Replace with your model path
)

device = "cpu"  # or "cuda"

# Load the source SentencePiece model
sp = spm.SentencePieceProcessor()
sp.load(sp_model_path)

src_lang = "fra_Latn"
tgt_lang = "eng_Latn"

beam_size = 6
if selected_option == "Translation":
    st.title("Text Translation")

    source_text = st.text_area("Enter your text:")
    # target_lang = st.selectbox(
    #     "Select the input language:", ["French", "Hindi", "Spanish"]
    # )

    if st.button("Translate to English"):
        source_sentences = [source_text]
        target_prefix = [[tgt_lang]] * len(source_sentences)

        # Subword the source sentences
        source_sents_subworded = sp.encode_as_pieces(source_sentences)
        source_sents_subworded = [
            [src_lang] + sent + ["</s>"] for sent in source_sents_subworded
        ]

        # Translate the source sentences
        translator = ctranslate2.Translator(ct_model_path, device=device)
        translations_subworded = translator.translate_batch(
            source_sents_subworded,
            batch_type="tokens",
            max_batch_size=2024,
            beam_size=beam_size,
            target_prefix=target_prefix,
        )
        translations_subworded = [
            translation.hypotheses[0] for translation in translations_subworded
        ]

        for translation in translations_subworded:
            if tgt_lang in translation:
                translation.remove(tgt_lang)

        # Desubword the target sentences
        translations = sp.decode(translations_subworded)
        st.header("Translation Result:")
        st.write(translations[0])

if selected_option == "Summary":
    st.title("Summary Feature")

    # Upload a PDF file for summarization
    uploaded_pdf_summary = st.file_uploader(
        "Upload a PDF for Summarization", type=["pdf"]
    )

    if uploaded_pdf_summary is not None:
        # Read the uploaded PDF file and extract text for summarization
        st.text("Extracting text from the PDF for Summarization...")
        with io.BytesIO(uploaded_pdf_summary.read()) as pdf_summary_file:
            legal_text_summary = extract_text_from_pdf(pdf_summary_file)

        # Split the text into smaller sections
        text_sections = [
            section.strip() for section in legal_text_summary.split("\n\n")
        ]

        # Initialize OpenAI
        load_dotenv()
        api_key = environ.get("OPENAI_API_KEY")
        openai.api_key = api_key

        # Generate summaries for each section and append to a list
        summaries = []
        for section in text_sections:
            summarization_prompt = f"Summarize the following text:\n{section}\n---"
            summary = openai.Completion.create(
                engine="davinci",
                prompt=summarization_prompt,
                max_tokens=200,  # Adjust this value as needed
            )
            summaries.append(summary.choices[0].text)

        # Combine section summaries into an overall summary
        overall_summary = "\n\n".join(summaries)

        # Display the generated summary
        st.header("Generated Summary:")
        st.write(overall_summary)

        # Allow users to select a destination language for translation
        dest_lang_summary = st.selectbox(
            "Select destination language for Summary:",
            list(googletrans_languages.values()),
        )

        # Perform translation of the summary
        translator_summary = Translator()
        translation_summary = translator_summary.translate(
            overall_summary, src="en", dest=dest_lang_summary
        )

        if translation_summary and hasattr(translation_summary, "text"):
            # Display the translated summary
            st.write(
                f"**Translated Summary ({dest_lang_summary}):** {translation_summary.text}"
            )
        else:
            st.error(
                "Translation failed for Summary. Please check your input and try again."
            )

if selected_option == "Querying":
    st.title("Querying Feature")

    # Upload a PDF file for querying
    uploaded_pdf_query = st.file_uploader("Upload a PDF for Querying", type=["pdf"])

    if uploaded_pdf_query is not None:
        # Read the uploaded PDF file and extract text for querying
        st.text("Extracting text from the PDF for Querying...")
        with io.BytesIO(uploaded_pdf_query.read()) as pdf_query_file:
            legal_text_query = extract_text_from_pdf(pdf_query_file)

        # Accept user questions/query for the uploaded PDF
        query = st.text_input("Ask questions about the uploaded PDF file for Querying")

        if query:
            text_splitter_query = RecursiveCharacterTextSplitter(
                chunk_size=1000, chunk_overlap=200, length_function=len
            )

            chunks_query = text_splitter_query.split_text(text=legal_text_query)

            # Store PDF name for querying
            store_name_query = uploaded_pdf_query.name[:-4]

            if os.path.exists(f"{store_name_query}_query.pkl"):
                with open(f"{store_name_query}_query.pkl", "rb") as f:
                    vectorstore_query = pickle.load(f)
            else:
                # Embeddings (OpenAI methods) for querying
                load_dotenv()

                # Get the API key from the environment variables
                api_key = environ.get("OPENAI_API_KEY")

                # Now you can use the API key to create OpenAIEmbeddings
                embeddings_query = OpenAIEmbeddings(api_key=api_key)

                # Store the chunks part in the database (vector) for querying
                vectorstore_query = FAISS.from_texts(
                    chunks_query, embedding=embeddings_query
                )

                with open(f"{store_name_query}_query.pkl", "wb") as f:
                    pickle.dump(vectorstore_query, f)

            docs_query = vectorstore_query.similarity_search(query=query, k=3)
            # OpenAI rank LNV process for querying
            llm_query = OpenAI(temperature=0)
            chain_query = load_qa_chain(llm=llm_query, chain_type="stuff")

            with st.form("language_selection_query"):
                st.write("Select language for the answer:")
                # Allow users to select any destination language for querying
                dest_lang_query = st.selectbox(
                    "Select destination language for Querying:",
                    list(googletrans_languages.values()),
                )
                submit_button_query = st.form_submit_button(
                    "Translate and Display Answer for Querying"
                )

            if submit_button_query:
                with get_openai_callback() as cb_query:
                    response_query = chain_query.run(
                        input_documents=docs_query, question=query
                    )
                    st.write("PDF Chatbot Response for Querying:")
                    st.write(response_query)

                try:
                    translator_query = Translator()
                    translation_query = translator_query.translate(
                        response_query, src="en", dest=dest_lang_query
                    )
                    if (
                        translation_query is not None
                        and hasattr(translation_query, "text")
                        and translation_query.text
                    ):
                        st.write(
                            f"**Translated Answer ({dest_lang_query}) for Querying:** {translation_query.text}"
                        )
                    else:
                        st.error(
                            "Translation failed for Querying. Please check your input and try again."
                        )
                except Exception as e:
                    st.error(
                        f"An error occurred during translation for Querying: {str(e)}"
                    )

if selected_option == "Keyword Extraction":
    st.title("Keyword Extraction Feature")

    # Input field for text
    text_for_keywords = st.text_area(
        "Paste your text for keyword extraction (max 500 words)", height=200
    )

    # Settings for keyword extraction
    min_ngram = st.slider("Minimum Ngram", 1, 4, 1)
    max_ngram = st.slider("Maximum Ngram", 1, 4, 2)
    top_n = st.slider("Number of Keywords", 1, 30, 10)
    remove_stopwords = st.checkbox("Remove Stop Words")
    use_mmr = st.checkbox("Use MMR for Diversity")

    # Keyword extraction
    if st.button("Extract Keywords"):
        kw_model = KeyBERT("distilbert-base-nli-mean-tokens")
        keywords = kw_model.extract_keywords(
            text_for_keywords,
            keyphrase_ngram_range=(min_ngram, max_ngram),
            stop_words="english" if remove_stopwords else None,
            use_mmr=use_mmr,
            top_n=top_n,
        )

        st.header("Keywords")
        df = DataFrame(keywords, columns=["Keyword/Keyphrase", "Relevancy"])
        df.index += 1
        st.table(df.style.highlight_max(axis=0))

if selected_option == "ChatBot":
    with st.sidebar:
        openai_api_key = st.text_input(
            "OpenAI API Key", key="chatbot_api_key", type="password"
        )
    # "[Get an OpenAI API key](https://platform.openai.com/account/api-keys)"
    # "[View the source code](https://github.com/streamlit/llm-examples/blob/main/Chatbot.py)"
    # "[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/streamlit/llm-examples?quickstart=1)"

    st.title("💬 Chatbot for LEGAL Queries")
    st.caption(
        "🚀 A streamlit chatbot powered by OpenAI LLM for general LEGAL education"
    )
    if "messages" not in st.session_state:
        st.session_state["messages"] = [
            {"role": "assistant", "content": "How can I help you?"}
        ]

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    if prompt := st.chat_input():
        if not openai_api_key:
            st.info("Please add your OpenAI API key to continue.")
            st.stop()

        openai.api_key = openai_api_key
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo", messages=st.session_state.messages
        )
        msg = response.choices[0].message
        st.session_state.messages.append(msg)
        st.chat_message("assistant").write(msg.content)


if __name__ == "__main__":
    st.button("Re-run")  # Add a button to restart the app
