import os
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Load environment variables from .env file
load_dotenv()

# --- CONFIGURATION ---
PDF_PATH = "constitution.pdf"
VECTOR_DB_PATH = "faiss_index_constitution"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

def create_vector_db():
    """
    Creates and saves a FAISS vector database from the Indian Constitution PDF.

    This function loads the PDF, splits it into manageable text chunks, 
    generates embeddings for each chunk using a HuggingFace model, and 
    stores them in a FAISS index, which is then saved to the local disk.
    """
    if not os.path.exists(PDF_PATH):
        print(f"Error: PDF file not found at {PDF_PATH}")
        print("Please download the 'Constitution-of-India.pdf' and place it in the root directory.")
        return

    print("Starting the data ingestion and vector DB creation process...")
    
    #Load the document
    print(f"Loading PDF document from {PDF_PATH}...")
    loader = PyPDFLoader(file_path=PDF_PATH)
    data = loader.load()
    print("PDF loaded successfully.")

    #Split the document into chunks
    print("Splitting document into text chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    docs = text_splitter.split_documents(data)
    print(f"Document split into {len(docs)} chunks.")

    #Create embeddings
    print(f"Initializing embedding model: {EMBEDDING_MODEL}...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL, model_kwargs={'device': 'cpu'},encode_kwargs={'normalize_embeddings': True})
    print("Embedding model initialized.")

    #Create a FAISS vector store and save it
    print("Creating FAISS vector store from chunks...")
    db = FAISS.from_documents(docs, embeddings)
    db.save_local(VECTOR_DB_PATH)
    print(f"Vector DB created and saved successfully at {VECTOR_DB_PATH}.")

if __name__ == "__main__":
    create_vector_db()