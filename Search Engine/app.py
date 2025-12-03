import os
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchRun
from langchain_classic.agents import initialize_agent, AgentType
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler

load_dotenv()
Groq_api_key=os.getenv('GROQ_API_KEY')

st.title("🔎 LangChain - Chat with search")

st.sidebar.title('Settings')
api_key = st.sidebar.text_input('Enter your Groq API Key Here', type='password')

if 'messages' not in st.session_state:
    st.session_state['messages'] = [
        {"role": "assistant", "content": "Hi, I'm a chatbot who can search the web. How can I help you?"}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg['role']).write(msg['content'])

Arxiv_Wrapper = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=200)
Wiki_Wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=200)

Arxiv = ArxivQueryRun(api_wrapper=Arxiv_Wrapper)
Wiki = WikipediaQueryRun(api_wrapper=Wiki_Wrapper)
search = DuckDuckGoSearchRun(name="search")

if prompt := st.chat_input("What is machine learning?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    llm = ChatGroq(
        groq_api_key=api_key,
        model="llama-3.1-8b-instant" 
    )

    tools = [search, Arxiv, Wiki]
    search_agent = initialize_agent(
        tools, llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        handling_parsing_errors=True
    )

    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
        response = search_agent.invoke(prompt, callbacks=[st_cb])
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.write(response)
