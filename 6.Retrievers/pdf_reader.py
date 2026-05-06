from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings  # For local embeddings (free alternative)
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq  # Change: Use ChatGroq instead of OpenAI
from dotenv import load_dotenv

load_dotenv()

loader = TextLoader('docs.txt')
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

docs = text_splitter.split_documents(documents)

# Change: Use HuggingFace embeddings (or other alternatives) since Groq doesn't provide embeddings
# Option 1: HuggingFace embeddings (free, local)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Option 2: If you want to use OpenAI embeddings still, you can keep that line
# vectorstore = FAISS.from_documents(docs, OpenAIEmbeddings())

vectorstore = FAISS.from_documents(docs, embeddings)

retriever = vectorstore.as_retriever()

query = "What are key takeaway from the documents?"

retrieved_docs = retriever.invoke(query)  # Note: get_relevant_documents is deprecated, use invoke

retrieved_text = "\n".join([doc.page_content for doc in retrieved_docs])

# Change: Use ChatGroq instead of OpenAI
llm = ChatGroq(
    model="llama-3.3-70b-versatile", 
    temperature=0.7
)

prompt = f"Based on the following text, answer the question: {query}\n\n{retrieved_text}"

# Change: Groq uses invoke() instead of predict()
answer = llm.invoke(prompt)
print('answer: ', answer.content)  # Access .content attribute for the response text