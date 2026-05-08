from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
load_dotenv()

model = ChatGroq(model='llama-3.3-70b-versatile')

loader = PyPDFLoader('/home/ai-with-rohan/Desktop/langchain/Lec10/genai.pdf')

docs = loader.load()
print(len(docs))


print(docs[0].page_content)
print(docs[0].metadata)