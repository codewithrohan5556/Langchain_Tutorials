from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
load_dotenv()

llm = ChatGroq(model='openai/gpt-oss-120b',temperature=0.7)

prompt = PromptTemplate(
    input_variables=['topic'],
    template='Suggest a catchy blog title about {topic}.' 
)

topic = input("Enter a topic")

formatted_prompt = prompt.format(topic=topic)
blog_title = llm.predict(formatted_prompt)

print("Generated blog title: ",blog_title)