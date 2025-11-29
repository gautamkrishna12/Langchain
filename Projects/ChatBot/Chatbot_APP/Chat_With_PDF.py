import os
import time
import tempfile
from pathlib import Path
import streamlit as st


import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import logging
logging.getLogger("torch").setLevel(logging.ERROR)


from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.chains import create_retrieval_chain
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

load_dotenv()
Groq_api_key=os.getenv('GROQ_API_KEY')
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ['HF_API_KEY']=os.getenv('GROQ_API_KEY')
os.environ['GROQ_API_KEY']=os.getenv('GROQ_API_KEY')

llm_model=ChatGroq(groq_api_key=Groq_api_key,model="openai/gpt-oss-20b")

Prompt = ChatPromptTemplate.from_template(
    """
    Use the information in the <context> section to answer the user's question as accurately and concisely as possible.

    <context>
    {context}
    </context>

    Question: {input}

    If the answer is not found in the context, say: "The provided context does not contain that information."
    """
)

def Create_VectorDB(path):

    if "vectordb" not in st.session_state:

        st.session_state.Embeddings=HuggingFaceEmbeddings(model="all-MiniLM-L6-v2")
        st.session_state.loader=PyPDFLoader(file_path=path)
        st.session_state.documents=st.session_state.loader.load()
        st.session_state.text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=250)
        st.session_state.splitted_docs=st.session_state.text_splitter.split_documents(st.session_state.documents)
        st.session_state.vectordb=FAISS.from_documents(st.session_state.splitted_docs,st.session_state.Embeddings)
        print('Vector Database Created Successfully.')

st.set_page_config(page_title="Chat with PDF", layout="wide")

st.markdown(
    """
    <div style="background-color:#4CAF50;padding:25px;border-radius:12px;margin-bottom:25px;">
        <h1 style="color:white;text-align:center;margin-bottom:5px;">Chat with PDF</h1>
        <p style="color:white;text-align:center;margin-top:-5px;font-size:16px;">
            Upload a PDF and start asking questions about it
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    uploaded_file = st.file_uploader(
        "Upload your PDF file", type="pdf", label_visibility="collapsed"
    )

if uploaded_file:
    temp_dir = tempfile.gettempdir()
    temp_path = Path(temp_dir) / uploaded_file.name
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.session_state.pdf_path = str(temp_path)
    st.success(f"Selected PDF: `{uploaded_file.name}`")

if "pdf_path" in st.session_state and st.session_state.pdf_path:
    pdf_file_path = Path(st.session_state.pdf_path)
    if pdf_file_path.exists() and pdf_file_path.suffix.lower() == ".pdf":
        st.info(f"Ready to process PDF: `{pdf_file_path.name}`")
    else:
        st.error(" Selected file is not valid")
else:
    st.info("Please select a PDF file to continue.")


if st.session_state.get("pdf_path"):
    if st.button(' Embed Document'):
        Create_VectorDB(st.session_state.pdf_path)
        st.write('Vector DataBase Created Successfully')

st.markdown(
    """
    <div style="border:2px solid #4CAF50; padding:20px; border-radius:12px; margin-top:25px;">
        <h3 style="margin-bottom:15px;">Ask a question about your PDF</h3>
    </div>
    """,
    unsafe_allow_html=True
)

user_prompt = st.text_input("Enter your query here...")

if user_prompt:
    if "vectordb" not in st.session_state:
        st.warning("Please embed the PDF first!")
    else:
        document_chain = create_stuff_documents_chain(llm_model, Prompt)
        retriever = st.session_state.vectordb.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        start = time.process_time()
        response = retrieval_chain.invoke({'input': user_prompt})
        elapsed_time = time.process_time() - start
        st.info(f"Response time: {elapsed_time:.2f} seconds")

        st.markdown(
        f"""
        <div style="
            border: 1px solid #d9d9d9;
            padding: 12px;
            border-radius: 8px;
            margin-top: 10px;">
            <strong>Response:</strong><br>{response['answer']}
        </div>
        """,
         unsafe_allow_html=True
        )