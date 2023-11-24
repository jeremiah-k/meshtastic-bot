import os
import discord
from discord.ext import commands
from langchain.llms import Ollama
from langchain.docstore.document import Document
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import GPT4AllEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA

# Bot command prefix
BOT_PREFIX = '!'

# Replace 'your_discord_bot_token' with your actual Discord bot token
DISCORD_BOT_TOKEN = 'your_discord_bot_token'

# Replace 'base_url' with the actual Ollama base URL
ollama = Ollama(base_url='http://localhost:11434', model="orca-mini")

# Persistence directory for ChromaDB
persist_directory = './chroma_db'

# Initialize Chroma with or without loading from disk
if os.path.exists(persist_directory) and os.listdir(persist_directory):
    # Load the vector store from disk if it exists
    vectorstore = Chroma(persist_directory=persist_directory, embedding_function=GPT4AllEmbeddings())
else:
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

    # Create embeddings and the vector store, then save to disk
    embedder = GPT4AllEmbeddings()
    vectorstore = Chroma.from_documents(documents=all_splits, embedding=embedder, persist_directory=persist_directory)

# Create a RetrievalQA chain
qa_chain = RetrievalQA.from_chain_type(ollama, retriever=vectorstore.as_retriever())

# Initialize the Discord bot
bot = commands.Bot(command_prefix=BOT_PREFIX)

@bot.event
async def on_ready():
    print(f'{bot.user.name} has connected to Discord!')

@bot.command(name='ask', help='Ask a Meshtastic-related question.')
async def ask_question(ctx, *, question):
    # Get and send the answer
    answer = qa_chain({"query": question})
    await ctx.send(answer)

# Run the bot
bot.run(DISCORD_BOT_TOKEN)