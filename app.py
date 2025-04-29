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
import streamlit as st
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import os
from langchain_ollama.llms import OllamaLLM
import time



# Load the image
image_path = "/home/lng/dirga/dirgallm/assets/dirgabanner.png"

# Use st.columns to center the image
col1, col2, col3 = st.columns(3)

with col2:
    st.image(
        image_path,
        use_column_width=True,  # Automatically adjust width to column
    )


# Set Ollama API endpoint (important!)
os.environ["OLLAMA_API_BASE"] = "http://localhost:11434"  # Default Ollama port

# Function to load content from a file
def load_file_content(file):
    try:
        content = file.read().decode("utf-8")
        return content
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return None

# Streamlit App
st.title("Ollama LLM with Streamlit")

# Sidebar for Ollama Model Selection
st.sidebar.header("Ollama Model Settings")
model_name = st.sidebar.selectbox("Select Ollama Model", ["gemma3:1b"])  # Add your models here

# Initialize Ollama LLM
@st.cache_resource  # Cache the LLM to avoid reloading on each interaction
def initialize_llm(model):
    try:
        llm = OllamaLLM(model=model)
        return llm
    except Exception as e:
        st.error(f"Error initializing Ollama LLM: {e}")
        return None

llm = initialize_llm(model_name)

# File Upload
uploaded_file = st.file_uploader("Add content from a file", type=["txt", "pdf", "docx","xlsx", "csv", "xlx" ])

# Prompt Input
prompt_text = st.text_area("Enter your prompt:", "Summarize the following text:")

# Process File Content
file_content = None
if uploaded_file is not None:
    file_content = load_file_content(uploaded_file)

# Create Prompt Template
template = """
You are a helpful assistant.  Answer the following question based on the provided context.

Context:
{context}

Question:
{question}

Answer:
"""
prompt = PromptTemplate(template=template, input_variables=["context", "question"])

start_time = time.time()


# Create LLMChain
if llm:
    llm_chain = LLMChain(prompt=prompt, llm=llm)

    # Run the Chain
    if st.button("Generate"):
        context = file_content if file_content else ""  # Use file content as context if available
        question = prompt_text  # Use the user's prompt as the question
        try:
            input_data = {"context": context, "question": question}  # Create a dictionary for input
            response = llm_chain.invoke(input=input_data)
            st.write(response)
        except Exception as e:
            st.error(f"Error generating response: {e}")
else:
    st.error("Ollama LLM not initialized. Check your settings and API key.")

end_time = time.time()
execution_time = end_time - start_time

st.warning(f"Memerlukan waktu {execution_time:.2f} detik untuk dieksekusi.")