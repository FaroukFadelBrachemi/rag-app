import os
import tempfile
from pathlib import Path
import pysqlite3
import sys

sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")

import chromadb
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import CharacterTextSplitter
from transformers import AutoTokenizer, AutoModelForCausalLM
import streamlit as st

# Set up directories
TMP_DIR = Path(__file__).resolve().parent.joinpath('data', 'tmp')
LOCAL_VECTOR_STORE_DIR = Path(__file__).resolve().parent.joinpath('data', 'vector_store')

# Ensure directories exist
TMP_DIR.mkdir(parents=True, exist_ok=True)
LOCAL_VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)

st.set_page_config(page_title="RAG")
st.title("Retrieval Augmented Generation Engine")


def load_documents():
    loader = DirectoryLoader(TMP_DIR.as_posix(), glob='**/*.pdf')
    documents = loader.load()
    return documents


def split_documents(documents):
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)
    return texts


def embeddings_on_local_vectordb(texts):
    vectordb = Chroma.from_documents(
        texts,
        embedding=OpenAIEmbeddings(),
        persist_directory=LOCAL_VECTOR_STORE_DIR.as_posix()
    )
    vectordb.persist()
    retriever = vectordb.as_retriever(search_kwargs={'k': 7})
    return retriever


def load_llama_model():
    model_name = "meta-llama/Llama-2-70b-chat-hf"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return model, tokenizer


def query_llm(retriever, query):
    model, tokenizer = load_llama_model()
    relevant_docs = retriever.get_relevant_documents(query)
    context = "\n".join([doc['text'] for doc in relevant_docs])
    input_text = f"Question: {query}\nContext: {context}\nAnswer:"
    inputs = tokenizer(input_text, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=512, num_beams=5, early_stopping=True)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    st.session_state.messages.append((query, response))
    return response


def input_fields():
    with st.sidebar:
        st.session_state.source_docs = st.file_uploader(label="Upload Documents", type="pdf", accept_multiple_files=True)


def process_documents():
    if not st.session_state.source_docs:
        st.warning("Please upload documents.")
    else:
        try:
            for source_doc in st.session_state.source_docs:
                with tempfile.NamedTemporaryFile(delete=False, dir=TMP_DIR.as_posix(), suffix='.pdf') as tmp_file:
                    tmp_file.write(source_doc.read())

            documents = load_documents()
            texts = split_documents(documents)

            if 'retriever' not in st.session_state or st.session_state.retriever is None:
                st.session_state.retriever = embeddings_on_local_vectordb(texts)
            else:
                # Update retriever with new documents
                st.session_state.retriever.add_documents(texts)

        except Exception as e:
            st.error(f"An error occurred while processing documents: {e}")


def boot():
    # Initialize session state variables
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "retriever" not in st.session_state:
        st.session_state.retriever = None

    input_fields()
    st.button("Submit Documents", on_click=process_documents)

    for message in st.session_state.messages:
        st.chat_message('human').write(message[0])
        st.chat_message('ai').write(message[1])

    if query := st.chat_input():
        st.chat_message("human").write(query)
        if st.session_state.retriever is None:
            st.warning("No retriever available. Please upload documents first.")
        else:
            response = query_llm(st.session_state.retriever, query)
            st.chat_message("ai").write(response)


if __name__ == '__main__':
    boot()
