import os
from pathlib import Path
import fitz
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from transformers import AutoTokenizer, AutoModelForCausalLM
import streamlit as st
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.environ['OPENAI_API_KEY']

LOCAL_VECTOR_STORE_DIR = Path("vector_store")
LOCAL_VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)

st.set_page_config(page_title="Chat with PDF using LLaMA")

def get_pdf_text(pdf_docs):
    """
    Extract text from multiple PDF files.
    """
    try:
        text = ""
        for pdf in pdf_docs:
            pdf_reader = fitz.open(pdf)
            for page_num in range(pdf_reader.page_count):  # Iterate through the pages using range
                page = pdf_reader.load_page(page_num)  # Load each page by index
                text += page.get_text() or ""  # Extract text from the page
        return text
    except Exception as e:
        st.error(f"Error reading PDF files: {e}")
        return ""

def get_text_chunks(text):
    """
    Split text into manageable chunks.
    """
    try:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = text_splitter.split_text(text)
        return chunks
    except Exception as e:
        st.error(f"Error splitting text into chunks: {e}")
        return []

def get_vector_store(text_chunks):
    """
    Generate a vector store from text chunks.
    """
    try:
        embeddings = OpenAIEmbeddings()
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        vector_store.save_local(LOCAL_VECTOR_STORE_DIR.as_posix())
    except Exception as e:
        st.error(f"Error creating vector store: {e}")

def get_conversational_chain():
    """
    Create a conversational chain for answering questions.
    """
    try:
        prompt_template = """
        Answer the question as detailed as possible from the provided context. If the answer is not in
        the provided context, respond with "The answer is not available in the context." Avoid providing incorrect information.

        Context:\n {context}\n
        Question: {question}\n
        Answer:
        """
        model_name = "meta-llama/Llama-2-70b-chat-hf"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)

        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
        return chain
    except Exception as e:
        st.error(f"Error loading conversational chain: {e}")
        return None

def user_input(user_question):
    """
    Process user queries and generate responses.
    """
    try:
        embeddings = OpenAIEmbeddings()
        vector_store = FAISS.load_local(LOCAL_VECTOR_STORE_DIR.as_posix(), embeddings, allow_dangerous_deserialization=True)
        docs = vector_store.similarity_search(user_question)

        chain = get_conversational_chain()
        if chain:
            response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
            st.write("Reply: ", response["output_text"])
    except Exception as e:
        st.error(f"Error processing user query: {e}")

def main():
    """
    Main application function.
    """
    st.title("Chat with PDF using LLaMA")

    user_question = st.text_input("Ask a Question about the PDF Files")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu")
        pdf_docs = st.file_uploader("Upload your PDF Files", accept_multiple_files=True, type="pdf")
        if st.button("Submit & Process"):
            if not pdf_docs:
                st.error("Please upload at least one PDF file.")
            else:
                with st.spinner("Processing..."):
                    # Process uploaded PDF files
                    raw_text = get_pdf_text(pdf_docs)
                    if raw_text:
                        text_chunks = get_text_chunks(raw_text)
                        if text_chunks:
                            get_vector_store(text_chunks)
                            st.success("Processing complete!")

if __name__ == "__main__":
    main()
