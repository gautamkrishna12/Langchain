import os
import streamlit as st
from dotenv import load_dotenv
from langchain_ollama import OllamaLLM
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
load_dotenv()

#Setting Langsmith tracking
os.environ['LANGCHAIN_API_KEY']=os.getenv('LANGCHAIN_API_KEY')
os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_PROJECT"]="Chatbot With Ollama"

#Creating Prompt Template

Prompt=ChatPromptTemplate.from_messages(
    [
        ('system',"You are a helpful massistant . Please  repsonse to the user queries"),
        ('user',"Question{question}")
    ]
)

def Get_Response(user_input,temperature,max_tokens):
    model=OllamaLLM(model="llama3.1")
    output_parser=StrOutputParser()
    chain=Prompt|model|output_parser
    answer=chain.invoke({'question':user_input})
    return answer

#Setting title for page
st.title('Q&A Chatbot')

#Choosing the LLM Model
llm=st.sidebar.selectbox('Select the model',['mistral'])

#Adding sliders for adjusting the model parameters
temperature=st.sidebar.slider('Temperature',min_value=0.0,max_value=1.0,value=0.6)
max_tokens=st.sidebar.slider('Max Tokens',min_value=25,max_value=250,value=120)

st.write('Give you query!')

user_input=st.text_input('You:')

if user_input:
    response=Get_Response(user_input,temperature,max_tokens)
    st.write(response)
else:
    st.write('Please provide the query.')