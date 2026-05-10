# ,FAISS,Weaviate,Pinecone,Qdrant,Redis,supabase,PGVector
# Pinecone() # FAISS() # Weaviate() # Qdrant() # Redis() # supabase() # PGVector()

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_classic.schema import Document

doc1 = Document(
    page_content="Maratha empire, early modern Indian empire that rose in the 17th century and dominated much of the Indian subcontinent during the 18th century.",
    metadata= {"team":"RCB"}
)

doc2 = Document(
    page_content="The Marathas were a Marathi-speaking warrior group mostly from what is now the state of Maharashtra in India.",
    metadata= {"tean":"MI"}
)

doc3 = Document(
    page_content=" They became politically active under the leadership of Shivaji, their first king, in opposition to the Islamic rulers of the time.",
    metadata= {"team":"CSK"}
)

doc4 = Document(
    page_content=" The formal Maratha empire began in 1674 with the coronation of Shivaji as Chhatrapati (“Keeper of the Umbrella”) and ended in 1818 after defeat by the English East India Company.",
    metadata= {"team":"SH"}
)

doc5 = Document(
    page_content="The 17th-century politics in the Indian subcontinent were dominated by multiple Islamic kingdoms, with the Mughal Empire controlling most of north India.",
    metadata= {"team":"KKR"}
)

docs = [doc1,doc2,doc3,doc4,doc5]

vector_store = Chroma(
    embedding_function=OpenAIEmbeddings(),
    persist_directory='my_chroma_db', #sqlite3 db format
    collection_name='sample'
)

vector_store.add_documents(docs)

vector_store.get(include=['embeddings','documents','metadatas'])


vector_store.similarity_search(
    query='who was the first Chhatrapati of Maratha empire?',
    k=2
)

vector_store.similarity_search_with_score(
    query='who was the first Chhatrapati of Maratha empire?',
    k=2
)


vector_store.similarity_search_with_score(
    query='',
    filter={'team':'CSK'}
)

updated_doc1 = Document(page_content="Maratha empire, early modern Indian empire that rose in the 17th century and dominated much of the Indian subcontinent during the 18th century.",
    metadata= {"team":"RCB"}
)

vector_store.update_document(document_id='',document=updated_doc1)

vector_store.get(include=['embeddings','documents','metadatas'])


vector_store.delete(ids=[''])

vector_store.get(include=['embeddings','documents','metadatas'])
