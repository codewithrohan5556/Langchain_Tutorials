from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
import os
from dotenv import load_dotenv

load_dotenv()

# Get Groq API key from environment variables
api_key = os.getenv("GROQ_API_KEY")

# Initialize Groq LLM (using Llama 3 70B as an example - you can change the model)
llm = ChatGroq(
    temperature=0.7,
    groq_api_key=api_key,
    model_name="llama-3.3-70b-versatile"  # Other options: "mixtral-8x7b-32768", "gemma-7b-it"
)

# 1st prompt -> detailed report
template1 = PromptTemplate(
    template='Write a detailed report on {topic}',
    input_variables=['topic']
)

# 2nd prompt -> summary
template2 = PromptTemplate(
    template='Write a 5 line summary on the following text.\n{text}',
    input_variables=['text']
)

# Generate detailed report
prompt1 = template1.invoke({'topic': 'black hole'})
result1 = llm.invoke(prompt1)

# Generate summary from the report (fixed: was using template1 instead of template2)
prompt2 = template2.invoke({'text': result1.content})
result2 = llm.invoke(prompt2) 

print(result2.content)