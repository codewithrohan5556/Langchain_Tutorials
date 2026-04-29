from langchain_huggingface import HuggingFaceEmbeddings

embedding = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

documents = [
    "Mumbai is the capital of Maharashtra"
]

vector = embedding.embed_documents(documents)

print(str(vector))