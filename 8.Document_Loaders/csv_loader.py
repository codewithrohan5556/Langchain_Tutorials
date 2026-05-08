from langchain_community.document_loaders import CSVLoader

loader = CSVLoader(file_path='/home/ai-with-rohan/Desktop/langchain/Lec10/titanic.csv')

docs = loader.lazy_load()

print(len(docs))
print(docs[0])

print(docs[0].page_content)
print(docs[0].metadata)