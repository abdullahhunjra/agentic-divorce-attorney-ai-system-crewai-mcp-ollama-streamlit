from fastmcp import FastMCP
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA


from langchain_ollama import ChatOllama  # ✅ new correct import

llm = ChatOllama(
    model="llama3",
    base_url="http://127.0.0.1:11434",  # Ollama default port
)


INDEX_DIR= "vector_stores/faiss_laws"

mcp = FastMCP("ClauseRetrieval")

#llm= ChatOllama(model="llama3")

emb = OllamaEmbeddings(model="llama3")
db = FAISS.load_local(INDEX_DIR, emb, allow_dangerous_deserialization=True)
retriever = db.as_retriever(search_kwargs={"k": 3})



@mcp.tool()
def find_relevant_clauses(query: str):
    """Retrieve relevant statutory clauses from legislation PDFs."""
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    answer = qa.run(query)
    return {"query": query, "answer": answer}

if __name__ == "__main__":
    mcp.run(transport="sse", host="127.0.0.1", port=8001, path="/sse")


