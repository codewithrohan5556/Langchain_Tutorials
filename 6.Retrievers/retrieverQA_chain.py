from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings  # For local embeddings (free alternative)
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq  # Change: Use ChatGroq instead of OpenAI
from langchain_classic.chains import retrieval_qa
from dotenv import load_dotenv

load_dotenv()

loader = TextLoader('docs.txt')
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=50)
docs = text_splitter.split_documents(documents)

vectorstore = FAISS.from_documents(docs,HuggingFaceEmbeddings())

retriever = vectorstore.as_retriever()

llm = HuggingFaceEmbeddings(model_name="llama-3.3-70b-versatile",temprature=0.7)

qa_chain = retrieval_qa.from_chain_type(llm=llm,retriever=retriever)

query = "What are key takeaway from the documents?"

answer = qa_chain.run(query)

print("Answer: ",answer)