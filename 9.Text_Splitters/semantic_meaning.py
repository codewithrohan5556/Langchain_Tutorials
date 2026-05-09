from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
from dotenv import load_dotenv

load_dotenv()

text_splitter = SemanticChunker(
    OpenAIEmbeddings(),breakpoint_threshold_type='standard deviation',
    breakpoint_threshold_amount=1
)

sample = """

"""
docs = text_splitter.create_documents([sample])
print(len(docs))
print(docs)
