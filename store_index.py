from src.helper import load_pdf, text_split, download_huggingface_embedings
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
import os

load_dotenv()

PINECODE_API_KEY = os.environ.get("PINECODE_API_KEY")

extracted_data = load_pdf("data/")
text_chunks = text_split(extracted_data)
embeddings = download_huggingface_embedings()

pc = Pinecone(api_key=PINECODE_API_KEY)

index_name = "copyia"

pc.create_index(
  name = index_name,
  dimension = 384,
  metric = "cosine",
  spec = ServerlessSpec(
    cloud = "aws",
    region = "us-east-1"
  ) 
)

docSearch = PineconeVectorStore.from_documents( 
    documents = text_chunks, 
    embedding = embeddings, 
    index_name = index_name
)