# 📝 Ollama RAG App

A simple **Retrieval-Augmented Generation (RAG)** application using **LangChain** and **Ollama LLM** for question-answering tasks based on PDF documents. This app allows users to upload PDF files, extract their content, generate embeddings, and query the content using a large language model (LLM).

## 🚀 Features

- 📄 **PDF Uploading:** Upload one or more PDF files to the app.
- 🔍 **Vector Store Creation:** Convert PDF content into embeddings using LangChain's OllamaEmbeddings and store them in a vector database (**FAISS**).
- 💬 **Question-Answering:** Ask questions related to the uploaded PDFs, and the app will provide concise answers using the LLM.
- 🔄 **Session Management:** Clear the current vector store and session state to start over with new documents.

## 🛠️ Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/ollama-rag-app.git
   cd ollama-rag-app


## Environment Variables

To store your environment variables, create a `.env` file in the root of the project. You need to define the following variable:


  LANGCHAIN_API_KEY=<your_langchain_api_key>


## Running the Application

To run the application, use the following command:

 ``` bash
pip install -r req.txt
streamlit run rag.py
```

## How to Use

1. Upload your PDF files using the file uploader in the sidebar.
2. After uploading, the app will process the files and load the content into the vector database.
3. Enter your question in the input field and hit enter to receive an answer based on the uploaded PDFs.

## Clearing the Session

You can clear the session and remove all uploaded files by clicking the "Clear" button in the sidebar.

## Contributing

Contributions are welcome! Let's learn together!!
Feel free to open issues or submit pull requests for any enhancements or bug fixes.


## Acknowledgments

- [Streamlit](https://streamlit.io/)
- [LangChain](https://langchain.com/)
- [Ollama](https://ollama.com/)
- [FAISS](https://faiss.ai/)

