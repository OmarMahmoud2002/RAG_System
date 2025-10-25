from dotenv import load_dotenv
import os

load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')
os.environ["OPENAI_BASE_URL"] = os.getenv('OPENAI_BASE_URL')

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import streamlit as st

st.title("RAG Chatbot")
st.write("AI Assistant for Document Q&A")

# Load env and API key
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
openai_base_url = os.getenv("OPENAI_BASE_URL")
openai_model = os.getenv("OPENAI_MODEL")

# Load PDF (replace with your PDF path)
loader = PyPDFLoader("egypt_tech.pdf")  # Replace with your PDF path
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
splits = text_splitter.split_documents(docs)

vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings(model="mxbai-embed-large", openai_api_key=openai_api_key, openai_api_base=openai_base_url))
retriever = vectorstore.as_retriever()

prompt_template = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template("You are a technology expert specializing in the tech industry in Egypt. Answer questions strictly based on the provided context from the report."),
    HumanMessagePromptTemplate.from_template("""
    Context:
    {context}
    Question:
    {question}
    """)
])

llm = ChatOpenAI(model_name=openai_model, openai_api_key=openai_api_key, openai_api_base=openai_base_url, temperature=0)
parser = StrOutputParser()

rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt_template
    | llm
    | parser
)

# Keep chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Show history
for message in st.session_state.messages:
    st.chat_message(message["role"]).markdown(message["content"])

# User input
user_input = st.chat_input("Ask a question about the document...")
if user_input:
    # Show user message
    st.chat_message("user").markdown(user_input)
    # Save user message
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Show streamed response
    with st.chat_message("ai"):
        response_placeholder = st.empty()
        full_response = ""
        for chunk in rag_chain.stream(user_input):
            full_response += chunk
            response_placeholder.markdown(full_response)
    # Save AI reply
    st.session_state.messages.append({"role": "ai", "content": full_response})
