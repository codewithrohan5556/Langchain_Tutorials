from langchain_groq import ChatGroq  
from langchain_classic.chains import LLMChain
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
load_dotenv()


llm = ChatGroq(
    model="llama-3.3-70b-versatile", 
    temperature=0.7
)

prompt = PromptTemplate(
    input_variables=['topic'],
    template="Suggest a catchy blog title about {topic}"
)

chain = LLMChain(llm = llm,prompt= prompt)

topic = input("Enter a topic")
output = chain.run(topic)

print("Generated Blog title : ",output)