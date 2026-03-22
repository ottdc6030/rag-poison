from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from argparse import ArgumentParser

# This is basically our "hello world" version of using Ollama. It takes the files from a local "data" folder and answers a single query.

parser = ArgumentParser()

parser.add_argument("llm", help="The LLM repository on HuggingFace. MAKE SURE YOU'VE ADDED THE MODEL TO OLLAMA BEFOREHAND")
parser.add_argument("--data_folder", help="The data folder to load attached files from.", default="data")
parser.add_argument("--embedding", help="HuggingFace embedding model, used to assist in semantic searches", default="BAAI/bge-small-en-v1.5")

args = parser.parse_args()

Settings.llm = Ollama(model=args.llm, request_timeout=360.0)
Settings.embed_model = HuggingFaceEmbedding(model_name=args.embedding)

print("Loading documents from the 'data' folder...")
documents = SimpleDirectoryReader(args.data_folder).load_data()

#Build the Local Vector Database (Memory only)
print("Chunking text and building the vector index...")
index = VectorStoreIndex.from_documents(documents)

#Set up the Query Engine and Ask a Question
query_engine = index.as_query_engine()

print("Ready! Sending query to local LLM...")
user_query = "Within this folder is a resume. Explain what kind of industry this resume is tailored to, and what the resume's best attributes are."
response = query_engine.query(user_query)

print("\n--- RESPONSE ---")
print(f"<think>\n{response}")