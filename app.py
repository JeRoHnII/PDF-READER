import os
import streamlit as st
import google.generativeai as genai
from PyPDF2 import PdfReader
from pdfminer.high_level import extract_text
import pytesseract
from PIL import Image
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
import io

# Set Google API key
os.environ["GOOGLE_API_KEY"] = "AIzaSyCKWxgyFTgNkY2n3PJS_pxEJhUa0oke9IY"
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

# Initialize chat history if not present
if "messages" not in st.session_state:
    st.session_state.messages = []

EXAMPLE_PDF_PATH = "example.pdf"

# Extract text with OCR fallback
def extract_pdf_text(pdf_path):
    text = ""
    try:
        # Try using pdfminer
        text = extract_text(pdf_path)
        if not text.strip():
            reader = PdfReader(pdf_path)
            for page in reader.pages:
                if page.extract_text():
                    text += page.extract_text()
                else:
                    # Convert page to image using PyMuPDF or other lib if OCR needed (mock logic)
                    st.warning("OCR fallback needed, but not implemented fully here.")
    except Exception as e:
        st.error(f"Text extraction failed: {e}")
    return text

# Split text into chunks
def split_text_into_chunks(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    return splitter.split_text(text)

# Create and save FAISS vector store
def create_vector_store(text_chunks):
    if not text_chunks:
        st.warning("No text chunks to create vector store.")
        return
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        vector_store.save_local("faiss_index")
        st.success("Vector store created successfully.")
    except Exception as e:
        st.error(f"Error creating vector store: {e}")

# Process PDF on first load
if "pdf_processed" not in st.session_state:
    with st.spinner("Processing PDF..."):
        extracted_text = extract_pdf_text(EXAMPLE_PDF_PATH)
        text_chunks = split_text_into_chunks(extracted_text)
        create_vector_store(text_chunks)
        st.session_state.pdf_processed = True

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
user_prompt = st.chat_input("Ask about the document...")
if user_prompt:
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vector_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        docs = vector_db.similarity_search(user_prompt)

        if docs:
            model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.3)
            chain = load_qa_chain(model, chain_type="stuff")
            response = chain.invoke({"input_documents": docs, "question": user_prompt}).get("output_text", "")
        else:
            model = genai.GenerativeModel("gemini-2.0-flash-lite")
            response = model.generate_content(user_prompt).text

    except Exception as e:
        response = f"Sorry, something went wrong: {e}"

    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.markdown(response)

