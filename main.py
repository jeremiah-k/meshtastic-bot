import os
from langchain.llms import Ollama
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import GPT4AllEmbeddings
from langchain.vectorstores import chroma
from langchain.chains import RetrievalQA

# Replace 'base_url' with the actual Ollama base URL
ollama = Ollama(base_url='http://localhost:11434', model="llama2")

# Path to the local directory containing the .mdx files
docs_path = 'meshtastic/docs/'

# Recursively find all .mdx files and load their content
all_documents = {}
for subdir, dirs, files in os.walk(docs_path):
    for file in files:
        if file.endswith('.mdx'):
            file_path = os.path.join(subdir, file)
            with open(file_path, 'r', encoding='utf-8') as f:
                all_documents[file_path] = f.read()

# Split documents into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
all_splits = {path: text_splitter.split_documents(doc) for path, doc in all_documents.items()}

# Flatten the splits for embedding
all_splits_flat = [chunk for chunks in all_splits.values() for chunk in chunks]

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