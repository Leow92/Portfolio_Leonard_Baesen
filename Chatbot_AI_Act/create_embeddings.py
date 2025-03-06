from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import torch
from google.cloud import storage

# Define the bucket name and file path
BUCKET_NAME = "embeddings_chat_ai"  # Replace with your bucket name
EMBEDDINGS_FILE = "embeddings.pkl"
LOCAL_EMBEDDINGS_FILE = "/tmp/embeddings.pkl"  # Temporary local file for saving

def load_pdf(pdf_doc):
    loader = PyMuPDFLoader(pdf_doc)
    documents = loader.load()
    print(len(documents))
    return documents

def save_embeddings(vectorstore):
    torch.save(vectorstore, LOCAL_EMBEDDINGS_FILE)
    print(f"Embeddings saved locally to {LOCAL_EMBEDDINGS_FILE}")

    # Now upload the local file to Google Cloud Storage
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob(EMBEDDINGS_FILE)
    blob.upload_from_filename(LOCAL_EMBEDDINGS_FILE)
    #blob.upload_from_string(bs.getvalue())
    print(f"Embeddings uploaded to gs://{BUCKET_NAME}/{EMBEDDINGS_FILE}")

def main():
    # Load and process document
    AI_Act = load_pdf("AI_Act.pdf")
    print("Document loaded.")

    # Split document into chunks
    #text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=70) #chunk_size=250, chunk_overlap=40
    text_splitter = CharacterTextSplitter(
        separator="\n\n"
    )
    splits = text_splitter.split_documents(AI_Act)
    print("Text split.")

    # Create embeddings
    embeddings_hugging = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2") #sentence-transformers/all-mpnet-base-v2 / sentence-transformers/all-MiniLM-L6-v2
    vectorstore = FAISS.from_documents(splits, embeddings_hugging)
    
    print("Vectorstore created.")

    # Optionally save the embeddings object if you need to reload the same embeddings later
    save_embeddings(vectorstore)

if __name__ == "__main__":
    main()