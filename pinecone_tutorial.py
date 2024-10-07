import os
from pinecone import Pinecone, ServerlessSpec
import openai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up API keys
openai.api_key = os.getenv("OPENAI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")

# Initialize Pinecone
pc = Pinecone(
    api_key=pinecone_api_key
)

# Check if the index exists, if not, create it
index_name = "openai-gen-qa-demo"
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=1536,  # For the text-embedding-ada-002 model
        metric="cosine",  # Change metric if necessary
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"  # Make sure this is the correct region for your use case
        )
    )

# Connect to the index
index = pc.Index(index_name)

# Example documents to index
documents = [
    "OpenAI develops advanced language models like GPT-3.",
    "Pinecone is a vector database that allows fast and scalable similarity searches.",
    "Python is one of the most popular programming languages, especially for data science and machine learning.",
    "Semantic search is a way to understand the meaning of a query rather than just matching keywords."
]

# Function to generate embeddings using OpenAI
def generate_embedding(text):
    response = openai.Embedding.create(
        input=text,
        model="text-embedding-ada-002"
    )
    return response['data'][0]['embedding']

# Insert documents into Pinecone index
for i, doc in enumerate(documents):
    embedding = generate_embedding(doc)
    index.upsert(vectors=[(str(i), embedding)])

print("Documents inserted into Pinecone.")

# Query function to get the most relevant document using Pinecone
def query_pinecone(query, top_k=3):
    # Generate query embedding
    query_embedding = generate_embedding(query)
    # Search the vector database for the most similar documents
    result = index.query(vector=query_embedding, top_k=top_k)
    return result['matches']

# Function to generate answers based on the retrieved documents
def generate_answer(query, context_documents):
    prompt = f"Answer the question based on the following documents:\n\n"
    for doc in context_documents:
        prompt += f"- {doc}\n"
    prompt += f"\nQuestion: {query}\nAnswer:"
    
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    return response['choices'][0]['message']['content']

# Example question
query = "What is Pinecone used for?"
print(f"Query: {query}")

# Query Pinecone for relevant documents
matches = query_pinecone(query)

# Retrieve matched documents
matched_docs = [documents[int(match['id'])] for match in matches]

# Generate the answer using OpenAI GPT model
answer = generate_answer(query, matched_docs)

print(f"Answer: {answer}")
