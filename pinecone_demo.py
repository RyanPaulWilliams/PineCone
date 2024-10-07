import os
from pinecone import Pinecone, ServerlessSpec
import openai
import numpy as np
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get Pinecone API key from environment
pinecone_api_key = os.getenv("PINECONE_API_KEY")

# Initialize Pinecone
pc = Pinecone(api_key=pinecone_api_key)

# Check if the index exists, if not, create it
index_name = 'openai-pinecone-demo'
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=1536,  # For the text-embedding-ada-002 model
        metric='euclidean',
        spec=ServerlessSpec(
            cloud='aws',
            region='us-east-1'  # Use the appropriate region
        )
    )

# Connect to the index
index = pc.Index(index_name)

# Example documents
documents = [
    "OpenAI develops powerful language models like GPT-3.",
    "Pinecone is a vector database for fast similarity search.",
    "Python is a popular programming language used in data science.",
    "Semantic search matches the meaning, not just keywords."
]

# Set OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Function to generate embeddings using OpenAI
def generate_embedding(text):
    foo = None
    try:
    
        response = openai.Embedding.create( 
        input=text,
        model="text-embedding-ada-002"
        ) 
        foo = response 
    except Exception as e:
        print (e)
        breakpoint()
        print (type(e).__name__)          
    return foo['data'][0]['embedding']

# Function to normalize embeddings using Min-Max normalization
def normalize_embedding(embedding):
    min_val = np.min(embedding)
    max_val = np.max(embedding)
    if max_val - min_val == 0:
        return embedding
    return (embedding - min_val) / (max_val - min_val)  

# Insert documents into Pinecone index with normalized embeddings   
for i, doc in enumerate(documents):
    embedding = generate_embedding(doc)
    normalized_embedding = normalize_embedding(np.array(embedding))  # Normalize using Min-Max
    index.upsert(vectors=[(str(i), normalized_embedding.tolist())])  # Convert to list before upserting

print("Documents inserted into Pinecone.")

# Function to query Pinecone with a question
def query_pinecone(query, top_k=2):
    # Generate embedding for the query
    query_embedding = generate_embedding(query)
    
    # Normalize the embedding using Min-Max normalization
    query_embedding = normalize_embedding(np.array(query_embedding))
    
    # Convert NumPy array to list
    query_embedding = query_embedding.tolist()
    
    # Query Pinecone for the most similar documents using the `vector` parameter
    result = index.query(vector=query_embedding, top_k=top_k)
    
    # Print the results
    for match in result['matches']:
        doc_id = match['id']
        print(f"Matched Document {doc_id}: {documents[int(doc_id)]} (Score: {match['score']})")


# Query example
query = "What is a language model?"
print(f"Query: {query}")
query_pinecone(query)
