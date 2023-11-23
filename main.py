import os
from langchain.llms import Ollama
from langchain.docstore.document import Document
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import GPT4AllEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA


# Replace 'base_url' with the actual Ollama base URL
ollama = Ollama(base_url='http://localhost:11434', model="mistral-openorca")

# Path to the local directory containing the .mdx files
docs_path = 'meshtastic/docs/'

# Recursively find all .mdx files, load their content, and wrap them in Document objects
all_documents = {}
for subdir, dirs, files in os.walk(docs_path):
    for file in files:
        if file.endswith('.mdx'):
            file_path = os.path.join(subdir, file)
            with open(file_path, 'r', encoding='utf-8') as f:
                # Read the content and create a Document object
                content = f.read()
                all_documents[file_path] = Document(page_content=content, metadata={})

# Now that you have a dictionary of Document objects, you can split them into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
all_splits = text_splitter.split_documents(all_documents.values())

# Flatten the splits for embedding
all_splits_flat = [chunk.page_content for chunk in all_splits]

# Create embeddings
embedder = GPT4AllEmbeddings()
vectorstore = Chroma.from_documents(documents=all_splits_flat, embedding=embedder)

# Create a RetrievalQA chain
qa_chain = RetrievalQA.from_chain_type(ollama, retriever=vectorstore.as_retriever())

# Example question
question = "What is the range of a Meshtastic device?"

# Get answer
answer = qa_chain({"query": question})
print(answer)