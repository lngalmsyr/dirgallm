# import streamlit as st
# from st_pages import Page, show_pages


# show_pages(
#     [
#         Page("pages/home.py", "Home", "🏠"),
#         Page("pages/chat.py", "Tanya DirgaInsight Sekarangg", ":books:"),
#         Page("pages/visualisasi.py", "Halaman Visualisasi", "🧐")
#     ])

import streamlit as st
from st_pages import add_page_title
from streamlit_extras.colored_header import colored_header
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import os
from langchain_ollama.llms import OllamaLLM
import time
from PyPDF2 import PdfReader  # Tambahkan ini
from docx import Document  # Tambahkan ini

# Fungsi untuk membaca file yang diperbaiki
def load_file_content(file):
    try:
        bytes_data = file.read()
        
        # Cek tipe file
        if file.type == "text/plain":
            # Coba berbagai encoding teks
            encodings = ['utf-8', 'latin-1', 'cp1252']
            for encoding in encodings:
                try:
                    return bytes_data.decode(encoding)
                except UnicodeDecodeError:
                    continue
            raise UnicodeDecodeError
            
        elif file.type == "application/pdf":
            # Handle PDF dengan PyPDF2
            pdf = PdfReader(file)
            text = ""
            for page in pdf.pages:
                text += page.extract_text()
            return text
            
        elif file.type in ["application/vnd.openxmlformats-officedocument.wordprocessingml.document", 
                          "application/msword"]:
            # Handle DOC/DOCX dengan python-docx
            doc = Document(file)
            return "\n".join([para.text for para in doc.paragraphs])
            
        else:
            st.error("Format file tidak didukung")
            return None

    except Exception as e:
        st.error(f"Error membaca file: {e}")
        return None

# Set Ollama API endpoint
os.environ["OLLAMA_API_BASE"] = "http://localhost:11434"

# Streamlit App
st.title("Ollama LLM with Streamlit")

# Sidebar untuk pemilihan model
st.sidebar.header("Ollama Model Settings")
model_name = st.sidebar.selectbox("Select Ollama Model", ["gemma3:1b"])

# Inisialisasi LLM
@st.cache_resource
def initialize_llm(model):
    try:
        return OllamaLLM(model=model)
    except Exception as e:
        st.error(f"Error initializing Ollama LLM: {e}")
        return None

llm = initialize_llm(model_name)

# File Uploader
uploaded_file = st.file_uploader("Add content from a file", 
                               type=["txt", "pdf", "docx", "xlsx", "csv", "xls"])

# Input prompt
prompt_text = st.text_area("Enter your prompt:", "Summarize the following text:")

# Proses konten file
file_content = None
if uploaded_file is not None:
    file_content = load_file_content(uploaded_file)

# Template prompt
template = """
You are a helpful assistant. Answer the following question based on the provided context.

Context:
{context}

Question:
{question}

Answer:
"""
prompt = PromptTemplate(template=template, input_variables=["context", "question"])

# Logika eksekusi
start_time = time.time()

if llm:
    llm_chain = LLMChain(prompt=prompt, llm=llm)
    
    if st.button("Generate"):
        context = file_content if file_content else ""
        question = prompt_text
        
        try:
            input_data = {"context": context, "question": question}
            response = llm_chain.invoke(input=input_data)
            st.write(response['text'])  # Perhatikan perubahan di sini untuk mengekstrak teks
        except Exception as e:
            st.error(f"Error generating response: {e}")

end_time = time.time()
execution_time = end_time - start_time

if 'response' in locals():
    st.warning(f"Memerlukan waktu {execution_time:.2f} detik untuk dieksekusi.")