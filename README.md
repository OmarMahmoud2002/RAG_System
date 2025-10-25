# RAG Chatbot

This project implements a Retrieval-Augmented Generation (RAG) chatbot using Streamlit and LangChain. The chatbot is designed to answer questions based on the content of a provided PDF document. It uses OpenAI's API for embeddings and language model completions.

## Features
- **PDF Document Loading**: The chatbot processes a PDF file to extract its content.
- **Text Splitting**: The document is split into smaller chunks for efficient processing.
- **Vector Store Creation**: The chunks are embedded and stored in a vector database using Chroma.
- **Contextual Question Answering**: The chatbot retrieves relevant chunks from the vector store and generates answers using OpenAI's language model.
- **Streamlit Interface**: A user-friendly interface for interacting with the chatbot.

## Key Technologies Used
1. **LangChain**:
   - `PyPDFLoader`: For loading and processing PDF documents.
   - `RecursiveCharacterTextSplitter`: For splitting text into manageable chunks.
   - `Chroma`: For creating and querying the vector store.
   - `OpenAIEmbeddings`: For generating embeddings of text chunks.
   - `ChatOpenAI`: For generating responses using OpenAI's language model.
2. **Streamlit**:
   - Provides the web interface for the chatbot.
3. **Python-dotenv**:
   - For managing environment variables securely.

## How It Works
1. **Environment Setup**:
   - The `.env` file contains the OpenAI API key, base URL, and model name.
   - Example:
     ```env
     OPENAI_API_KEY=your_api_key
     OPENAI_BASE_URL=https://your_base_url/v1/
     OPENAI_MODEL=mistral-small
     ```
2. **PDF Loading**:
   - The PDF file is loaded using `PyPDFLoader`.
3. **Text Splitting**:
   - The document is split into smaller chunks using `RecursiveCharacterTextSplitter`.
4. **Vector Store Creation**:
   - The chunks are embedded using `OpenAIEmbeddings` and stored in a Chroma vector store.
5. **Question Answering**:
   - The chatbot retrieves relevant chunks from the vector store and generates answers using `ChatOpenAI`.
6. **Streamlit Interface**:
   - Users can interact with the chatbot through a web interface.

## How to Run
1. **Install Dependencies**:
   - Ensure you have Python installed.
   - Install the required packages:
     ```bash
     pip install -r requirements.txt
     ```
2. **Set Up Environment Variables**:
   - Create a `.env` file with the following content:
     ```env
     OPENAI_API_KEY=your_api_key
     OPENAI_BASE_URL=https://your_base_url/v1/
     OPENAI_MODEL=mistral-small
     ```
3. **Run the Application**:
   - Start the Streamlit app:
     ```bash
     streamlit run rag.py
     ```
4. **Interact with the Chatbot**:
   - Open the provided URL in your browser and start asking questions about the document.

## Notes
- Ensure the PDF file path in `rag.py` is correct.
- The OpenAI API key must have access to the specified model.

## Example
- **Question**: "What is the main topic of the document?"
- **Answer**: The chatbot will provide a concise answer based on the content of the PDF.


<img width="996" height="925" alt="image" src="https://github.com/user-attachments/assets/d470b76b-5272-40de-91d1-ded4fd803d94" />
<img width="996" height="925" alt="image" src="https://github.com/user-attachments/assets/1b7adedb-aab7-42ac-9344-5db83b595813" />



Enjoy using the RAG Chatbot!
