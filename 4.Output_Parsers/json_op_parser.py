from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
import os
from dotenv import load_dotenv
load_dotenv()

# Get Groq API key from environment variables
api_key = os.getenv("GROQ_API_KEY")

# Initialize Groq LLM (using Llama 3 70B as an example - you can change the model)
model = ChatGroq(
    groq_api_key=api_key,
    model_name="llama-3.3-70b-versatile"  
)

parser = JsonOutputParser()
template = PromptTemplate(
    # template="Give me name,age and city of a fictional person \n {format_instruction}",
    # schema failing
    template="Give me 5 facts about {topic} \n {format_instruction}",
    input_variables=['topic'],
    partial_variables={'format_instruction':parser.get_format_instructions()}
)

# prompt = template.format()
# result = model.invoke(prompt)
# final_result = parser.parse(result.content)
# print(final_result)
# print(type(final_result))
### Chains
chain = template | model | parser 
result = chain.invoke({'topic':'black hole'})
print(result)