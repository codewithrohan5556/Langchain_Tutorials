from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(model='gpt-4',temperature=1.5,max_completion_tokens=10)

result = model.invoke("What is the capital of Maharashtra?")
result1 = model.invoke("Write a five line poem on cricket")

print(result)
print(result.content)


