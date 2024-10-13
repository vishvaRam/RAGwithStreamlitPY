import os
from dotenv import load_dotenv
from langchain_ollama import OllamaLLM
import streamlit as st
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ['LANGCHAIN_PROJECT'] = os.getenv("OllamaGenAI")


prompt = ChatPromptTemplate.from_messages([
    ("system","You are an helfull assistant, so answer the following questions."),
    ("user","Questions{question}")
])


llm = OllamaLLM(model="llama3.2:1b")

outParse = StrOutputParser()

chain = prompt | llm |outParse

st.title("Langchain Demo with Llama3.2 1b")

input = st.text_input("Ask me anything.")

if input:
    st.write(chain.invoke({'question':input}))
