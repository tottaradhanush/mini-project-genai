import streamlit as st
import os
import requests
import pdfplumber
from bs4 import BeautifulSoup
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from supabase import create_client
from dotenv import load_dotenv

# Load API Keys
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# Initialize Supabase
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# Configure Gemini AI
genai.configure(api_key=GEMINI_API_KEY)

# Load Embedding Model
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

HEADERS = {"User-Agent": "Mozilla/5.0"}

# Streamlit Page Configuration
st.set_page_config(page_title="AI Assistant", layout="wide")

# ---- SIDEBAR ---- #
st.sidebar.title("📂 Data Processing")
st.sidebar.markdown("Choose how to process knowledge:")

# Data Source Selection
selected_option = st.sidebar.radio("Select:", ["PDF Upload", "Web URL Extraction"])

# ---- MAIN UI ---- #
st.title("💡 AI-Powered Knowledge Assistant")
st.write("Upload a **PDF** or enter a **Website URL** to process and chat with AI.")

# Store user selections
if "active_source" not in st.session_state:
    st.session_state["active_source"] = None

# ---- PDF Upload Section ---- #
if selected_option == "PDF Upload":
    st.session_state["active_source"] = "pdf"
    pdf_file = st.file_uploader("📄 Upload a PDF", type=["pdf"])

    if pdf_file:
        st.session_state["pdf_name"] = pdf_file.name

        # Extract text from PDF
        def extract_pdf_text(pdf):
            text = ""
            with pdfplumber.open(pdf) as pdf_reader:
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
            return text

        extracted_text = extract_pdf_text(pdf_file)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = text_splitter.split_text(extracted_text)

        # Store embeddings in Supabase
        for chunk in chunks:
            embedding = embedding_model.encode(chunk).tolist()
            supabase.table("pdf_store").insert(
                {"content": chunk, "embedding": embedding, "source_name": pdf_file.name}
            ).execute()

        st.success(f"✅ PDF '{pdf_file.name}' processed and stored!")

# ---- Web URL Section ---- #
elif selected_option == "Web URL Extraction":
    st.session_state["active_source"] = "web"
    url = st.text_input("🔗 Enter Website URL")
    search_query = st.text_area("🔍 What information do you need?", height=150)

    if st.button("Extract Content"):
        st.session_state["web_url"] = url
        st.session_state["web_query"] = search_query

        # Extract text from Website
        def extract_website_text(url):
            response = requests.get(url, headers=HEADERS)
            soup = BeautifulSoup(response.text, "html.parser")
            paragraphs = [p.get_text(strip=True) for p in soup.find_all("p")]
            return "\n".join(paragraphs)

        extracted_text = extract_website_text(url)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = text_splitter.split_text(extracted_text)

        # Store embeddings in Supabase
        for chunk in chunks:
            embedding = embedding_model.encode(chunk).tolist()
            supabase.table("web_store").insert(
                {"content": chunk, "embedding": embedding, "source_name": url}
            ).execute()

        st.success(f"✅ Content from '{url}' extracted and stored!")

# ---- Chat Interface ---- #
st.markdown("---")
st.subheader("💬 Chat with AI")

# Display chat history
for msg in st.session_state.get("chat_history", []):
    if msg["sender"] == "user":
        st.markdown(f'<p style="color: #3498db; font-size: 18px;"><b>👤 You:</b> {msg["text"]}</p>', unsafe_allow_html=True)
    else:
        st.markdown(f'<p style="color: #2c3e50; font-size: 18px;"><b>🤖 AI:</b> {msg["text"]}</p>', unsafe_allow_html=True)
        st.markdown(f'<p style="font-size: 14px; color: gray;"><b>📌 Source:</b> {msg["source"]}</p>', unsafe_allow_html=True)

# User Query Input (Increased Size)
user_query = st.text_area("💡 Ask AI anything...", height=100)

# Process Query
if st.button("Send") and user_query:
    # Retrieve stored data from Supabase
    source_table = "web_store" if st.session_state["active_source"] == "web" else "pdf_store"
    response = supabase.table(source_table).select("content", "source_name").execute()
    documents = response.data

    if documents:
        context_texts = [doc["content"] for doc in documents[:5]]
        context_sources = [doc["source_name"] for doc in documents[:5]]

        model = genai.GenerativeModel("gemini-2.0-flash")
        answer = model.generate_content(f"Context:\n{context_texts}\n\nQuestion: {user_query}").text.strip()

        # Store response in chat history
        st.session_state.setdefault("chat_history", []).extend([
            {"sender": "user", "text": user_query},
            {"sender": "bot", "text": answer, "source": ", ".join(set(filter(None, context_sources)))}
        ])
        st.rerun()

# Clear Chat History Button
if st.sidebar.button("🗑 Clear Chat"):
    st.session_state["chat_history"] = []
    st.rerun()
