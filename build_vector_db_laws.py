import os 
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings

LAWS_DIR= "data/laws"
INDEX_DIR= "vector_stores/faiss_laws"

def build_law_index():
    print("Building law index...")
    docs=[]
    for f in os.listdir(LAWS_DIR):
        if f.endswith(".pdf"):
            path = os.path.join(LAWS_DIR, f)
            print("→", f)
            docs.extend(PyPDFLoader(path).load_and_split())


    print(f"Loaded {len(docs)} chunks")
    emb = OllamaEmbeddings(model="llama3")
    bd = FAISS.from_documents(docs, emb)
    bd.save_local(INDEX_DIR)
    print(f"Saved to {INDEX_DIR}")

if __name__ == "__main__":
    build_law_index()