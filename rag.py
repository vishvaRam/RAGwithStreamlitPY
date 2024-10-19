import streamlit as st
import os
import warnings
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma, FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_ollama import OllamaLLM

from timer import Timer

# ollama_model = "llama3.2:1b-instruct-q4_0"
ollama_model = "qwen2.5:0.5b"
# emb_model ="all-mpnet-base-v2"
emb_model ="multi-qa-MiniLM-L6-cos-v1"

# Load environment variables
load_dotenv()
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ['LANGCHAIN_PROJECT'] = "RAG_Ollama"


# Function to handle document loading and creating the retriever
def load_pdfs_and_create_retriever(directory_path):
    try:
        # Suppress warnings
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        warnings.filterwarnings("ignore", category=UserWarning)

        # Load documents
        loader = PyPDFDirectoryLoader(directory_path)
        documents = loader.load()

        # Split documents
        chunk_size = 3000
        chunk_overlap = 250
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        docs = text_splitter.split_documents(documents)

        # Sentence embeddings

        embeddings = SentenceTransformerEmbeddings(model_name=emb_model)

        # Vector store
        # Chroma DB is not working fine when we do not close the db properly.
        # db = Chroma.from_documents(docs, embeddings)
        db = FAISS.from_documents(docs, embeddings)
        retriever = db.as_retriever(search_kwargs={"k": 7})

        return retriever

    except Exception as e:
        print(e)
        return None


llm = OllamaLLM(model=ollama_model, temperature=0.6)


# Function to answer questions using the loaded retriever
def answer_question(retriever, question):
    try:
        # Language model
        # llm = Ollama(model="llama3.2:1b")

        # QA chain
        chain = load_qa_chain(llm, chain_type="stuff")

        # Retrieve relevant documents
        relevant_docs = retriever.get_relevant_documents(question, )

        # Get answer
        response = chain.run(input_documents=relevant_docs, question=question)
        return response

    except Exception as e:
        print(e)
        return str(e)


# Streamlit Application
def main():
    # Title and description
    st.title("📄 PDF Question-Answering")
    st.write("Upload your PDFs and ask multiple questions to get insights from the documents.")

    # Create a folder to save PDFs if it doesn't exist
    save_directory = "uploaded_pdfs"
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    # File uploader to upload PDF
    uploaded_files = st.sidebar.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)

    # Clear button to reset the state
    if st.sidebar.button("Clear"):
        # Remove files from the directory
        for file in os.listdir(save_directory):
            file_path = os.path.join(save_directory, file)
            if os.path.isfile(file_path):
                os.remove(file_path)
        # Clear the session state
        if "retriever" in st.session_state:
            del st.session_state.retriever
        if "summary" in st.session_state:
            del st.session_state.summary

        st.toast("🗑️ Cleared all uploaded PDFs and session.")

    # Check if files are uploaded
    if uploaded_files:
        # Save uploaded files
        for uploaded_file in uploaded_files:
            file_path = os.path.join(save_directory, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            # st.toast(f"✅ File saved: {uploaded_file.name}")
            print(f"File saved: {uploaded_file.name}")

        # Load documents and create retriever only once
        if "retriever" not in st.session_state:
            with st.spinner("Loading PDF ... "):
                timer = Timer()
                timer.start()
                st.session_state.retriever = load_pdfs_and_create_retriever(save_directory)
                st.toast(f"✅ PDFs processed successfully in {timer.stop()}")
            if st.session_state.retriever:
                print("Retriever Session check")
            else:
                st.toast("Failed to process the PDFs. Please try again.")

    summary_prompt = """
    Generate an executive summary in the following format, using Indian currency (₹) for all monetary references and providing precise numbers:

        1. Performance Overview: Summarize key metrics and trends, including achievements and shortfalls against targets during the review period. Mention specific amounts in crores (₹) as appropriate.

        2. Major Findings: Highlight important positive and negative observations, including significant deviations, irregularities, or non-compliances noticed, with exact figures.

        3. Non-Performing Assets (NPA): Detail the status of NPAs, mentioning accounts involved, outstanding amounts in crores (₹), and any newly added NPAs during the period, with precise values.

        4. Customer Service: Evaluate the quality of customer service and suggest areas for improvement, including measurable criteria where possible.

        5. Income and Profitability: Provide an analysis of income, focusing on exact reductions or growth in non-interest income. Use crores (₹) to report specific changes in revenue.

        6. Housekeeping and Statutory Compliance: Comment on housekeeping standards, compliance with statutory requirements, and any persisting irregularities, specifying precise financial implications in crores (₹), where applicable.

        7. Suggestions for Improvement: Offer recommendations for areas that require attention or actions for future improvement, including specific targets or measures.

    Ensure the summary presents details includes exact figures, bullet-point style, highlighting important positive and negative aspects observed during the audit.
    Organize it under relevant headings and provide concise insights similar to a formal audit executive summary.
    """

    QaA_prompt = """

    Above is the given question: 
        - Using Indian currency (₹) for all monetary references
        - Ensure the content is presented details in a bullet-point style and providing precise numbers.
        - Organize it under relevant headings and provide concise insights similar to a formal auditor format.
    """

    if "retriever" in st.session_state and st.session_state.retriever:
        if "summary" not in st.session_state:
            with st.spinner("Generating summary ... "):
                timer = Timer()
                timer.start()
                st.session_state.summary = answer_question(st.session_state.retriever, summary_prompt)
                st.toast(f"Summary generated successfully in {timer.stop()}")

        expander = st.expander("💡 Summary ")
        expander.write(st.session_state.summary)

    # Ask question input field
    question = st.text_area("❓ Ask a question about the documents", height=80)

    if st.button("Get Answer"):
        if question and "retriever" in st.session_state and st.session_state.retriever:
            with st.spinner("Generating output ..... "):
                timer = Timer()
                timer.start()
                answer = answer_question(st.session_state.retriever, question + QaA_prompt)
                st.toast(f"Answer generated in {timer.stop()}")

            st.subheader("💡 Answer ")
            st.write(answer)

        else:
            st.toast("Please upload PDF files and enter a question.")


if __name__ == "__main__":
    main()