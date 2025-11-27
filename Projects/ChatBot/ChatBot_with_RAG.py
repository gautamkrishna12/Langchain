import os
import time
import streamlit as st

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains import create_retrieval_chain
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

#Setting Environment Variables
load_dotenv()
os.environ['GROQ_API_KEY']=os.getenv('GROQ_API_KEY')
Groq_api_key=os.getenv('GROQ_API_KEY')
os.environ['HF_API_KEY']=os.getenv('HF_API_KEY')


llm_model=ChatGroq(groq_api_key=Groq_api_key,model="openai/gpt-oss-20b")

Prompt=ChatPromptTemplate.from_template(
    """
    Answer the question based on the provided contents
    Provide most accurate response.
    <context>
    {context}
    <context>
    Question:{input}

    """
)

def Create_VectorDB():
    if "vectorDB" not in st.session_state:
        #st.session_state.embeddings=OllamaEmbeddings(model='llama3.1')
        st.session_state.Embeddings=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        st.session_state.loader=PyPDFLoader('/Users/gasunil/My Folder /Learn/Langchain/Chatbot/4.1-RAG Q&A Conversation/temp.pdf')
        st.session_state.docs=st.session_state.loader.load()
        st.session_state.text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
        st.session_state.splitted_documents=st.session_state.text_splitter.split_documents(st.session_state.docs)
        st.session_state.vectorDB=FAISS.from_documents(st.session_state.splitted_documents,st.session_state.Embeddings)

st.markdown("<h1 style='text-align: center; color: white; font-size: 60px;'>Chat with PDF</h1>",unsafe_allow_html=True)

user_prompt=st.text_input("Enter the query here")

if st.button('Embedd Document'):
    Create_VectorDB()
    st.write('Vector DataBase Created Successfully')

if user_prompt:
    document_chain=create_stuff_documents_chain(llm_model,Prompt)
    retriever=st.session_state.vectorDB.as_retriever()
    retrieval_chain=create_retrieval_chain(retriever,document_chain)

    start=time.process_time()
    response=retrieval_chain.invoke({'input':user_prompt})
    print(f"Response time: {time.process_time()-start}")

    st.write(response['answer'])











