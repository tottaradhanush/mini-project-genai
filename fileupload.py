import streamlit as st
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv
import tempfile


load_dotenv()

gemini_api_key = os.getenv('GEMINI_API_KEY')

st.markdown("""
     <style>
        body {
            background-color: #FFFFFF;
            color: #333333;
        }
        .chat-box {
            padding: 20px;
            border-radius: 10px;
            border: 1px solid #ddd;
            margin-bottom: 20px;
            background-color: #f9f9f9;
        }
        .user-message {
            background-color: #e1f5fe;
            padding: 10px;
            border-radius: 10px;
            border: 1px solid #81d4fa;
            margin: 10px 0;
        }
        .ai-response {
            background-color: #f1f8e9;
            padding: 10px;
            border-radius: 10px;
            border: 1px solid #c5e1a5;
            margin: 10px 0;
        }
        .header {
            text-align: center;
            font-size: 2em;
            color: #333333;
        }
        .subheader {
            text-align: center;
            font-size: 1.2em;
            color: #666666;
        }
    </style>
""", unsafe_allow_html=True)

# Display the header and subheader
st.markdown("<h1 class='header'>AI Chatbot for PDF Q&A</h1>", unsafe_allow_html=True)
st.markdown("<h4 class='subheader'>Upload your PDF, create a vector database, and ask questions</h4>", unsafe_allow_html=True)

# Initialize the language model
llm = ChatGoogleGenerativeAI(google_api_key=gemini_api_key, model="gemini-2.0-flash")

# Define prompt template
prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question.
    <context>
    {context}
    <context>
    Questions: {input}
    """
)


# Create vector database out of uploaded PDF
def create_vector_db_out_of_the_uploaded_pdf_file(pdf_file):
    if "vector_store" not in st.session_state:
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(pdf_file.read())
            pdf_file_path = temp_file.name

        #Hugging Face embeddings to convert the text into vector representations.
        st.session_state.embeddings = HuggingFaceEmbeddings(model_name='BAAI/bge-small-en-v1.5', model_kwargs={'device': 'cpu'}, encode_kwargs={'normalize_embeddings': True})

        #Extracts text from the PDF file.
        st.session_state.loader = PyPDFLoader(pdf_file_path)
        st.session_state.text_document_from_pdf = st.session_state.loader.load()

        #Splits the text into smaller chunks
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        st.session_state.final_document_chunks = st.session_state.text_splitter.split_documents(st.session_state.text_document_from_pdf)

        #Stores the document embeddings in a FAISS(Facebook AI Similarity Search) vector database.
        st.session_state.vector_store = FAISS.from_documents(st.session_state.final_document_chunks, st.session_state.embeddings)   


# File uploader UI with instructions
pdf_input_from_user = st.file_uploader("Please Upload your PDF file", type=['pdf'], help="Upload a PDF file to process and create a vector database.")

if pdf_input_from_user is not None:
    st.info("You have uploaded a PDF. Click the button below to create the vector database.")

    if st.button("Create Vector Database"):
        with st.spinner("Processing PDF and creating embeddings..."):
            create_vector_db_out_of_the_uploaded_pdf_file(pdf_input_from_user)
            st.success("PDF file processed and vector database created successfully!")

# Chat interface for asking questions
if "vector_store" in st.session_state:
    user_prompt = st.text_input("Enter your question about the PDF content:")

    if st.button('Submit Prompt'):
        if user_prompt:
            # Display the user question in a chat-like box
            st.markdown(f'<div class="chat-box user-message">You: {user_prompt}</div>', unsafe_allow_html=True)
            #Retrieves relevant document chunks.
            with st.spinner("Fetching the answer..."):
                document_chain = create_stuff_documents_chain(llm, prompt)
                retriever = st.session_state.vector_store.as_retriever()
                retrieval_chain = create_retrieval_chain(retriever, document_chain)

                response = retrieval_chain.invoke({'input': user_prompt})

                st.markdown(f'<div class="chat-box ai-response">AI: {response["answer"]}</div>', unsafe_allow_html=True)
        else:
            st.error("Please enter a question before submitting.")
else:
    st.warning("Please upload a PDF and create the vector database before asking questions.")