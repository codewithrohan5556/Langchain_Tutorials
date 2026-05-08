from langchain_groq import ChatGroq
from langchain_community.document_loaders import TextLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

model = ChatGroq(model='llama-3.3-70b-versatile')

prompt = PromptTemplate(
    template = 'Write a summary for the following poem - \n {poem}',
    input_variables=['poem']
)

parser = StrOutputParser()

loader = TextLoader("/home/ai-with-rohan/Desktop/langchain/Lec10/cricket.txt",encoding='utf-8')

docs = loader.load()
print(type(docs)) # List
print(docs[0].page_content)
print(docs[0].metadata)


chain = prompt | model | parser
print(chain.invoke({'poem':docs[0].page_content}))