from langchain_groq import ChatGroq
from langchain_community.document_loaders import DirectoryLoader,PyPDFLoader
from dotenv import load_dotenv
load_dotenv()

model = ChatGroq(model='llama-3.3-70b-versatile')

loader = DirectoryLoader(
    path="directory",
    glob="*.pdf",
    loader_cls=PyPDFLoader)

docs = loader.load()
print(len(docs))


print(docs[354].page_content)
print(docs[354].metadata)