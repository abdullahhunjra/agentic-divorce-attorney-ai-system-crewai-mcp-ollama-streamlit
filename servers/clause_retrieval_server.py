# fastmcp_server.py
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from fastmcp import FastMCP
from langchain_ollama import ChatOllama
from retrieval_pipeline import retrieve_relevant_chunks



llm = ChatOllama(
    model="llama3",
    base_url="http://127.0.0.1:11434", 
)

mcp = FastMCP("ClauseRetrieval")




@mcp.tool() 
def find_relevant_clauses(query: str, top_k: int = 5):
    """
    Retrieve top relevant statutory clauses from the Matrimonial Causes Act 1973.
    Returns both the retrieved chunks and a concise LLM summary.
    """
    results = retrieve_relevant_chunks(query, top_k=top_k)

    # Generate concise legal summary
    context = "\n\n".join([r["text"] for r in results])
    prompt = f"""
    You are a legal summarization assistant.
    Based on the following retrieved sections of the Matrimonial Causes Act 1973,
    provide a concise, factual answer to the query.

    Query: {query}

    Relevant clauses:
    {context}
    """

    answer = llm.invoke(prompt)
    return {
        "query": query,
        "answer": answer.content.strip() if hasattr(answer, "content") else str(answer).strip(),
        "results": results
    }



if __name__ == "__main__":
    mcp.run(transport="sse", host="127.0.0.1", port=8001, path="/sse")
