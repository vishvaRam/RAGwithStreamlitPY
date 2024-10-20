import os
import random
import string

import streamlit as st
from dotenv import load_dotenv
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_groq.chat_models import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from streamlit import session_state

# emb_model ="multi-qa-MiniLM-L6-cos-v1"

# Load environment variables
load_dotenv()
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ['LANGCHAIN_PROJECT'] = "RAG_Grog"

emb_model = "all-mpnet-base-v2"
embeddings = HuggingFaceEmbeddings(model_name=emb_model)

st.title("RAG with chat history")
st.subheader("Upload PDFs and chat with them.")

api_key = st.text_input("Enter your grog API key :", type="password")


def generate_random_string(length):
    characters = string.ascii_letters + string.digits + string.punctuation
    random_string = ''.join(random.choice(characters) for _ in range(length))
    return random_string


save_directory = "uploaded_pdfs"

global retriever

if api_key:
    os.environ['GROQ_API_KEY'] = api_key
    llm = ChatGroq(model_name="llama-3.2-3b-preview", temperature=0.6, )

    if "session_id" not in st.session_state:
        print("Checking session_id ...")
        st.session_state.session_id = generate_random_string(10)

    if 'store' not in session_state:
        print("Checking chat history store ...")
        st.session_state.store = {}

    upload_pdf = st.sidebar.file_uploader("Select a PDF file to chat.", accept_multiple_files=True, type="pdf")

    if upload_pdf:
        document = []
        # Save uploaded files
        for uploaded_file in upload_pdf:
            file_path = os.path.join(save_directory, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            loader = PyPDFDirectoryLoader(save_directory)
            doc = loader.load()
            document.extend(doc)

            # Split documents
        chunk_size = 3000
        chunk_overlap = 250
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        if "retriever" not in st.session_state:
            print("Checking retriever ...")
            with st.spinner("Loading Vector DB ..."):
                Split_doc = text_splitter.split_documents(document)

                db = FAISS.from_documents(Split_doc, embeddings)
                retriever = db.as_retriever(search_kwargs={"k": 7})

                st.session_state.retriever = retriever


        contextualize_q_system_prompt = (
            "Given a chat history and the latest user question "
            "which might reference context in the chat history, "
            "formulate a standalone question which can be understood "
            "without the chat history. Do NOT answer the question, "
            "just reformulate it if needed and otherwise return it as is."
        )

        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("user", "{input}")
            ]
        )

        history_aware_retriever = create_history_aware_retriever(llm, st.session_state.retriever, contextualize_q_prompt)

        system_prompt = (
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer "
            "the question. If you don't know the answer, say that you "
            "don't know. Use three sentences maximum and keep the "
            "answer concise."
            "\n\n"
            "{context}"
        )
        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

        doc_chain = create_stuff_documents_chain(llm, qa_prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever, doc_chain)


        def get_session_id(session):
            if st.session_state.session_id not in st.session_state.store:
                st.session_state.store[st.session_state.session_id] = ChatMessageHistory()
            return st.session_state.store[st.session_state.session_id]


        conv_rag_chain = RunnableWithMessageHistory(
            rag_chain,
            get_session_id,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer"
        )

        input_text = st.text_input("Your Question : ")

        if input_text:
            session_history = get_session_id(st.session_state.session_id)
            res = conv_rag_chain.invoke(
                {"input": input_text},
                config={
                    "configurable":
                        {"session_id": st.session_state.session_id}
                }
            )
            # st.write(st.session_state.store)
            st.write("Assistant :",res['answer'])
            st.write("Chat history : ",session_history.messages )