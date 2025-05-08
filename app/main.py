from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema import Document
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
load_dotenv()
import os

docs = [
    Document(page_content="My name is Varun Arya, and I am from Delhi. I am 28 years old, a software developer, and I love chicken biryani. I am a graduate, married, and I own a pet named Jenny.")
]

splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=10)
split_docs = splitter.split_documents(docs)

embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
vector_store = Chroma.from_documents(split_docs, embedding_model)
retriever = vector_store.as_retriever()

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


print("Ask a question about Varun's introduction:")

while True:
    query = input("Enter your question (or 'exit' to quit): ")
    if query.lower() == 'exit':
        break
    result = qa_chain.invoke(query)
    print("AI Response:")
    print("=======================")
    print()
    print(result)

