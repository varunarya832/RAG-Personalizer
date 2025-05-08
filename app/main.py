from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema import Document
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request models
class DocsInput(BaseModel):
    docs: list[str]

class QueryInput(BaseModel):
    query: str

# Globals
vector_store = None
retriever = None
qa_chain = None

@app.post("/upload_docs")
def upload_docs(payload: DocsInput):
    global vector_store, retriever, qa_chain

    # Convert lines into LangChain Document objects
    documents = [Document(page_content=line) for line in payload.docs if line.strip()]

    # Text splitting (optional)
    splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=10)
    split_docs = splitter.split_documents(documents)

    # Create embedding model and vector store
    embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = Chroma.from_documents(split_docs, embedding_model)
    retriever = vector_store.as_retriever()

    # Setup QA chain
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
    )

    return {"message": "Documents processed and vector store created successfully."}

@app.post("/ask")
def ask_question(payload: QueryInput):
    global qa_chain

    if not qa_chain:
        return {"error": "Please upload documents first."}

    result = qa_chain.invoke(payload.query)
    answer_text = result.get('result') or str(result)
    return {"answer": answer_text}
