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
from dotenv import load_dotenv
import os
os.environ["GOOGLE_API_KEY"] ="AIzaSyAxuRrd_dF349H9rwQynLLDp1WwzvbUOq0"

# Configure API Key
genai.configure(api_key=os.environ['GOOGLE_API_KEY'])

# Initialize chat history if not present
if "messages" not in st.session_state:
    st.session_state.messages = []  # Stores conversation history

# Path to the example PDF
EXAMPLE_PDF_PATH = "example.pdf"

# Function to extract text from PDF with OCR fallback
def extract_pdf_text(pdf_path):
    text = ""
    try:
        text = extract_text(pdf_path)
        if not text.strip():
            with open(pdf_path, "rb") as file:
                reader = PdfReader(file)
                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text
                    else:
                        image = Image.open(file)
                        text += pytesseract.image_to_string(image)
    except Exception as e:
        st.error(f"Error extracting text from {pdf_path}: {e}")
    return text

# Function to split text into chunks
def split_text_into_chunks(text):
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    return text_splitter.split_text(text)

# Function to create FAISS vector store
def create_vector_store(text_chunks):
    if not text_chunks:
        return
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        vector_store.save_local("faiss_index")
    except:
        pass  # Ignore errors to keep it silent

# Process PDF on first load (No messages shown)
if "pdf_processed" not in st.session_state:
    extracted_text = extract_pdf_text(EXAMPLE_PDF_PATH)
    text_chunks = split_text_into_chunks(extracted_text)
    create_vector_store(text_chunks)
    st.session_state.pdf_processed = True

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input field
user_prompt = st.chat_input("Ask about the document...")
if user_prompt:
    # Add user message to chat history and display it
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    # Process query
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vector_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        docs = vector_db.similarity_search(user_prompt)

        if docs:
            model = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", temperature=0.3)
            chain = load_qa_chain(model, chain_type="stuff")
            response = chain.invoke({"input_documents": docs, "question": user_prompt}).get("output_text", "")
        else:
            model = genai.GenerativeModel("gemini-1.5-flash-latest")
            response = model.generate_content(user_prompt).text

    except:
        response = "Sorry, I couldn't process your request."

    # Add AI response to chat history and display it
    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.markdown(response)
