from langchain_community.document_loaders import WebBaseLoader
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

model = ChatGroq(model='llama-3.3-70b-versatile')

prompt = PromptTemplate(
    template = 'Write a answer the following question \n {question} from the following - \n{text}',
    input_variables=['question','text']
)

parser = StrOutputParser()

url = ''
loader = WebBaseLoader(url)

docs = loader.load()
# print(len(docs))
# print(docs)
# print(docs[0].page_content)

chain = prompt | model | parser
print(chain.invoke({'question':'What is the product that we are talking about?','text':docs[0].page_content})
)



