from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embedding = OpenAIEmbeddings(model='text-embedding-3-large',
        dimensions=32)

documents = [
    "Delhi is the capital of India",
    "Mumbai is the capital of Maharashtra",
    "Bengluru is the capital of Karnataka"
]

result = embedding.embed_documents(documents)

print(str(result))