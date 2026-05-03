from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
import os
from dotenv import load_dotenv

load_dotenv()

# Get Hugging Face API key from environment variables
api_key = os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")

# Initialize Hugging Face Endpoint
llm = HuggingFaceEndpoint(
    repo_id="mixtral-8x7b-32768",
    task="text-generation",
    temperature=0.7,
    max_new_tokens=512
)
# Other options: "mixtral-8x7b-32768", "gemma-7b-it"

# Wrap with ChatHuggingFace for chat interface
model = ChatHuggingFace(llm=llm)

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
result1 = model.invoke(prompt1)

# Generate summary from the report
prompt2 = template2.invoke({'text': result1.content})
result2 = model.invoke(prompt2)

print(result2)