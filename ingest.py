import os
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.vectorstores import FAISS

load_dotenv()

KB_DIR  = "Knowledge_base"
INDEX_DIR = "faiss_index"
def build_index() -> None:
    # 1. file and folder seach in knowledge_base folder
    if not os.path.isdir(KB_DIR):
        raise FileNotFoundError(f" '{KB_DIR} /' folder not found. please check ")
    
    loader = DirectoryLoader(KB_DIR, glob="*.txt", loader_cls=TextLoader)
    documents = loader.load()
  
    if not documents:
        raise ValueError(f"no .txt file present or found in the directory '{KB_DIR}'. Add file and try again")
    
    # 2. recusrive text spliter is used for chunking
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap = 50,
    )
    
    chunks = splitter.split_documents(documents)
    
    # 3. embedding of the documendts using FAISS vector index and open AI embeddings
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(INDEX_DIR)
    
    print(f"FAISS embedding is completed for the knowledge base and  stored in '{INDEX_DIR}'")
    
    
if  __name__ == "__main__":
    build_index()