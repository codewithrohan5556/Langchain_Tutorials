from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
load_dotenv()
from langchain_core.output_parsers import StrOutputParser

prompt1 = PromptTemplate(
    template='Generate a detailed report on {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='Generate a five pointer summary from the following text \n {text}',
    input_variables=['text']
)


model = ChatGroq(model='llama-3.3-70b-versatile')
parser = StrOutputParser()

chain = prompt1 | model | parser | prompt2 | model | parser

result = chain.invoke({'topic':'Unemployment in india'})

print(result)

chain.get_graph().print_ascii()