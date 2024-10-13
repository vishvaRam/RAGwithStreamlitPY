import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
import streamlit as st

# Load environment variables
load_dotenv()
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ['LANGCHAIN_PROJECT'] = "RAG_Ollama"

# Constants
UPLOAD_FOLDER = "uploaded_pdfs"
LLM_MODEL = "llama3.2:1b"

# Ensure the upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize session state variables
if "clear_clicked" not in st.session_state:
    st.session_state.clear_clicked = False

# LLM and system prompt setup
llm = OllamaLLM(model=LLM_MODEL)
system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the answer concise."
    "\n\n{context}"
)
prompt = ChatPromptTemplate.from_messages([("system", system_prompt), ("human", "{input}")])

def create_vector_embeddings(path):
    """Load documents, split them, and create vector embeddings."""
    if "vectors" not in st.session_state or st.session_state.clear_clicked:
        with st.spinner('Loading docs to vectorDB...'):
            embeddings = OllamaEmbeddings(model=LLM_MODEL)
            loader = PyPDFDirectoryLoader(path)
            docs = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=100)
            final_docs = text_splitter.split_documents(docs)
            
            # Store embeddings and vectors in session state
            st.session_state.vectors = FAISS.from_documents(final_docs, embeddings)
            st.session_state.clear_clicked = False
            st.toast("Loaded Vector DB for PDFs.", icon='✅')

def clear_vector_store():
    """Clear vector store and related session state data."""
    st.session_state.clear_clicked = True
    for key in ["vectors", "embeddings", "loader", "docs", "text_splitter", "final_docs", "out"]:
        st.session_state.pop(key, None)
    st.session_state.selected_files = []
    st.toast("Cleared session.", icon='✅')

def delete_files(path):
    """Delete all files in the specified directory."""
    for file_path in [os.path.join(path, f) for f in os.listdir(path)]:
        if os.path.isfile(file_path):
            os.remove(file_path)
    clear_vector_store()

# Streamlit UI setup
st.title("Ollama RAG App")

file_loader = st.sidebar.file_uploader("Choose PDF files", type=["pdf"], accept_multiple_files=True)

if file_loader and not st.session_state.clear_clicked:
    for uploaded_file in file_loader:
        file_path = os.path.join(UPLOAD_FOLDER, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        # st.toast(f"Saved file in {UPLOAD_FOLDER}", icon='✅')
        print(f"Saved file in {UPLOAD_FOLDER}")
    create_vector_embeddings(UPLOAD_FOLDER)

st.sidebar.button(label="Clear", on_click=lambda: delete_files(UPLOAD_FOLDER), type="primary")

text_input = st.text_input("Ask me anything!")

if text_input:
    if "vectors" in st.session_state:
        docs_chain = create_stuff_documents_chain(llm, prompt)
        retriever = st.session_state.vectors.as_retriever()
        rag_chain = create_retrieval_chain(retriever, docs_chain)

        with st.spinner('Generating output...'):
            res = rag_chain.invoke({"input": text_input})
            st.session_state.out = st.write(res['answer'])
    else:
        st.toast("Load the PDF files first!", icon="🚨")
