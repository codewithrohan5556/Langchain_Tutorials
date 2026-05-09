from langchain_classic.text_splitter import RecursiveCharacterTextSplitter, Language

text = """

"""


splitter = RecursiveCharacterTextSplitter.from_language(language=Language.MARKDOWN,
                                                        chunk_size=300,
                                                        chunk_overlap=0)

chunks = splitter.split_text(text)

print(len(chunks))
print(chunks[1])