import streamlit as st
import os
import requests
import pdfplumber
from bs4 import BeautifulSoup
import google.generativeai as genai
from sentence_transformers import SentenceTransformer, util
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

# ---- STREAMLIT UI ---- #
st.set_page_config(page_title="AI Knowledge Assistant", layout="wide")

st.sidebar.title("📂 Data Processing")
st.sidebar.markdown("Choose how to process knowledge:")
selected_option = st.sidebar.radio("Select:", ["PDF Upload", "Web URL Extraction"])

# Store user selections
if "active_source" not in st.session_state:
    st.session_state["active_source"] = None

st.title("💡 AI-Powered Knowledge Assistant")
st.write("Upload a **PDF** or enter a **Website URL** to process and chat with AI.")

# ---- PDF PROCESSING ---- #
if selected_option == "PDF Upload":
    st.session_state["active_source"] = "pdf"
    pdf_file = st.file_uploader("📄 Upload a PDF", type=["pdf"])

    if pdf_file:
        st.session_state["pdf_name"] = pdf_file.name

        def extract_pdf_text(pdf):
            text = ""
            with pdfplumber.open(pdf) as pdf_reader:
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
            return text

        extracted_text = extract_pdf_text(pdf_file)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = text_splitter.split_text(extracted_text)

        for chunk in chunks:
            embedding = embedding_model.encode(chunk).tolist()
            supabase.table("pdf_store").insert(
                {"content": chunk, "embedding": str(embedding), "source_name": pdf_file.name}
            ).execute()

        st.success(f"✅ PDF '{pdf_file.name}' processed and stored!")

# ---- WEB URL PROCESSING ---- #
elif selected_option == "Web URL Extraction":
    st.session_state["active_source"] = "web"
    url = st.text_input("🔗 Enter Website URL")

    if url and st.button("Extract Content"):
        st.session_state["web_url"] = url

        def extract_website_text(url):
            response = requests.get(url, headers=HEADERS)
            soup = BeautifulSoup(response.text, "html.parser")
            paragraphs = [p.get_text(strip=True) for p in soup.find_all("p")]
            return "\n".join(paragraphs)

        extracted_text = extract_website_text(url)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = text_splitter.split_text(extracted_text)

        for chunk in chunks:
            embedding = embedding_model.encode(chunk).tolist()
            supabase.table("web_store").insert(
                {"content": chunk, "embedding": str(embedding), "source_name": url}
            ).execute()

        st.success(f"✅ Homepage content from '{url}' stored!")

# ---- CHAT INTERFACE ---- #
st.markdown("---")
st.subheader("💬 Chat with AI")

for msg in st.session_state.get("chat_history", []):
    if msg["sender"] == "user":
        st.markdown(f'<p style="color: #3498db; font-size: 18px;"><b>👤 You:</b> {msg["text"]}</p>', unsafe_allow_html=True)
    else:
        st.markdown(f'<p style="color: #2c3e50; font-size: 18px;"><b>🤖 AI:</b> {msg["text"]}</p>', unsafe_allow_html=True)
        st.markdown(f'<p style="font-size: 14px; color: gray;"><b>📌 Source:</b> {msg["source"]}</p>', unsafe_allow_html=True)

# ---- FUNCTION TO FETCH TOP 10 CHUNKS ---- #
def fetch_top_chunks(question, table_name):
    question_embedding = embedding_model.encode(question)
    response = supabase.table(table_name).select("content", "embedding", "source_name").execute()
    documents = response.data

    if not documents:
        return []

    similarities = []
    for doc in documents:
        try:
            doc_embedding = eval(doc["embedding"])  # Convert string back to list
        except:
            continue  # Skip if conversion fails

        score = util.pytorch_cos_sim(question_embedding, doc_embedding)[0][0].item()
        similarities.append((score, doc["content"], doc["source_name"]))

    similarities.sort(reverse=True, key=lambda x: x[0])
    return similarities[:10]

# ---- USER QUERY INPUT ---- #
user_query = st.text_area("💡 Ask AI anything...", height=100)

if st.button("Send") and user_query:
    source_table = "web_store" if st.session_state["active_source"] == "web" else "pdf_store"
    top_chunks = fetch_top_chunks(user_query, source_table)

    if top_chunks:
        context_texts = [chunk[1] for chunk in top_chunks]
        context_sources = [chunk[2] for chunk in top_chunks]

        model = genai.GenerativeModel("gemini-2.0-flash")
        answer = model.generate_content(f"Context:\n{context_texts}\n\nQuestion: {user_query}").text.strip()

        st.session_state.setdefault("chat_history", []).extend([
            {"sender": "user", "text": user_query},
            {"sender": "bot", "text": answer, "source": ", ".join(set(filter(None, context_sources)))}
        ])
        st.rerun()
    else:
        st.error("No relevant data found. Proceeding with additional scraping...")

        # ---- PHASE 2: FETCHING HOMEPAGE LINKS ---- #
        response = requests.get(url, headers=HEADERS, timeout=10)
        soup = BeautifulSoup(response.text, "html.parser")

        homepage_links = [a["href"] for a in soup.find_all("a", href=True) if a["href"].startswith(url)]
        st.write(f"Found {len(homepage_links)} homepage links. Sending to LLM for filtering...")

        model = genai.GenerativeModel("gemini-2.0-flash")
        relevant_links = model.generate_content(f"Filter relevant homepage links from: {homepage_links}").text.strip().split("\n")

        for link in relevant_links:
            internal_links = [a["href"] for a in BeautifulSoup(requests.get(link, headers=HEADERS).text, "html.parser").find_all("a", href=True)]
            relevant_internal_links = model.generate_content(f"Filter relevant internal links from: {internal_links}").text.strip().split("\n")

            for int_link in relevant_internal_links:
                page_text = extract_website_text(int_link)
                if page_text:
                    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
                    chunks = text_splitter.split_text(page_text)
                    for chunk in chunks:
                        embedding = embedding_model.encode(chunk).tolist()
                        supabase.table("web_store").insert(
                            {"content": chunk, "embedding": str(embedding), "source_name": int_link}
                        ).execute()

        st.success("Stored internal links' content. Try asking again!")

# ---- CLEAR CHAT BUTTON ---- #
if st.sidebar.button("🗑 Clear Chat"):
    st.session_state["chat_history"] = []
    st.rerun()
