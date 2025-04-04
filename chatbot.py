
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

# ---- Load API Keys ---- #
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# ---- Initialize Supabase ---- #
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# ---- Configure Gemini AI ---- #
genai.configure(api_key=GEMINI_API_KEY)

# ---- Load Embedding Model ---- #
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

HEADERS = {"User-Agent": "Mozilla/5.0"}
homepage_links = set()

# ---- Streamlit UI ---- #
st.set_page_config(page_title="AI Knowledge Assistant", layout="wide")

st.sidebar.title("📂 Data Processing")
selected_option = st.sidebar.radio("Select:", ["PDF Upload", "Web URL Extraction"])

st.title("💡 AI-Powered Knowledge Assistant")
st.write("Upload a *PDF* or enter a *Website URL* to process and chat with AI.")

if "active_source" not in st.session_state:
    st.session_state["active_source"] = None

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

# ---- Function to Extract Text from PDFs ---- #
def extract_pdf_text(pdf_file):
    with pdfplumber.open(pdf_file) as pdf:
        return "\n".join([page.extract_text() or "" for page in pdf.pages]).strip()

# ---- Function to Extract Text from Websites ---- #
def extract_website_text(url):
    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        paragraphs = [p.get_text(strip=True) for p in soup.find_all("p")]
        return "\n".join(paragraphs)
    except requests.RequestException:
        return None

# ---- Function to Extract Links from a Web Page ---- #
def extract_links(url):
    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        links = set()

        for a_tag in soup.find_all("a", href=True):
            link = a_tag["href"]
            if link.startswith("/") or link.startswith(url):
                full_link = link if link.startswith(url) else f"{url.rstrip('/')}/{link.lstrip('/')}"
                links.add(full_link)

        return tuple(links)
    except requests.RequestException:
        return ()

# ---- PDF Processing ---- #
if selected_option == "PDF Upload":
    st.session_state["active_source"] = "pdf"
    uploaded_file = st.file_uploader("📄 Upload a PDF", type="pdf")

    if uploaded_file:
        extracted_text = extract_pdf_text(uploaded_file)
        if extracted_text:
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            chunks = text_splitter.split_text(extracted_text)

            for chunk in chunks:
                embedding = embedding_model.encode(chunk).tolist()
                supabase.table("pdf_store").insert(
                    {"content": chunk, "embedding": str(embedding), "source_name": uploaded_file.name}
                ).execute()

            st.success(f"✅ PDF '{uploaded_file.name}' processed successfully!")
        else:
            st.error("❌ No extractable text found in the PDF.")

# ---- Web URL Processing ---- #
if selected_option == "Web URL Extraction":
    st.session_state["active_source"] = "web"
    url = st.text_input("🔗 Enter Website URL")

    if url and st.button("Extract Content"):
        st.session_state["web_url"] = url
        extracted_text = extract_website_text(url)

        if extracted_text:
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            chunks = text_splitter.split_text(extracted_text)

            for chunk in chunks:
                embedding = embedding_model.encode(chunk).tolist()
                supabase.table("web_store").insert(
                    {"content": chunk, "embedding": str(embedding), "source_name": url}
                ).execute()

            homepage_links.update(extract_links(url))
            st.success(f"✅ Homepage content from '{url}' stored!")

# ---- Function to Retrieve Relevant Chunks ---- #
def get_top_chunks(question, source):
    question_embedding = embedding_model.encode(question)

    # Identify relevant homepages if querying web data
    if source == "web":
        model = genai.GenerativeModel("gemini-2.0-flash")
        relevant_homepages = model.generate_content(f"Given these links: {homepage_links}, which are relevant for: '{question}'? Return only relevant links.")
        relevant_homepages = tuple(relevant_homepages.text.split("\n"))

        for homepage in relevant_homepages:
            extracted_text = extract_website_text(homepage)
            if extracted_text:
                chunks = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50).split_text(extracted_text)
                for chunk in chunks:
                    embedding = embedding_model.encode(chunk).tolist()
                    supabase.table("web_store").insert(
                        {"content": chunk, "embedding": str(embedding), "source_name": homepage}
                    ).execute()

        relevant_internal_links = set()
        for homepage in relevant_homepages:
            relevant_internal_links.update(extract_links(homepage))

        final_internal_links = model.generate_content(f"Given these internal links: {relevant_internal_links}, which are most relevant for: '{question}'? Return only relevant links.")
        final_internal_links = tuple(final_internal_links.text.split("\n"))

        for link in final_internal_links:
            extracted_text = extract_website_text(link)
            if extracted_text:
                chunks = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50).split_text(extracted_text)
                for chunk in chunks:
                    embedding = embedding_model.encode(chunk).tolist()
                    supabase.table("web_store").insert(
                        {"content": chunk, "embedding": str(embedding), "source_name": link}
                    ).execute()

    # Query Supabase for relevant chunks
    table_name = "pdf_store" if source == "pdf" else "web_store"
    response = supabase.table(table_name).select("content", "embedding", "source_name").execute()
    documents = response.data

    if not documents:
        return []

    similarities = []
    for doc in documents:
        doc_embedding = eval(doc["embedding"])
        score = util.pytorch_cos_sim(question_embedding, doc_embedding)[0][0].item()
        similarities.append((score, doc["content"], doc["source_name"]))

    similarities.sort(reverse=True, key=lambda x: x[0])
    return similarities[:10]

# ---- Chat Interface ---- #
st.markdown("---")
st.subheader("💬 Chat with AI")

for msg in st.session_state["chat_history"]:
    st.markdown(f'{"👤 You" if msg["sender"] == "user" else "🤖 AI"}:** {msg["text"]}')
    if msg["sender"] == "ai":
        st.markdown(f'📌 *Source:* {msg["source"]}', unsafe_allow_html=True)

# ---- User Query Input ---- #
user_query = st.text_area("💡 Ask AI anything...", height=100)

if st.button("Send") and user_query:
    top_chunks = get_top_chunks(user_query, st.session_state["active_source"])

    if top_chunks:
        combined_text = "\n\n".join([chunk[1] for chunk in top_chunks])
        model = genai.GenerativeModel("gemini-2.0-flash")
        response = model.generate_content(f"Answer the question based on the following data:\n\n{combined_text}\n\nQuestion: {user_query}")
        answer = response.text

        st.session_state["chat_history"].append({"sender": "ai", "text": answer, "source": top_chunks[0][2]})
    else:
        st.error("No relevant data found.")