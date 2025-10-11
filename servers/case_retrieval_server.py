from fastmcp import FastMCP
from langchain_community.tools import TavilySearchResults
from langchain_community.chat_models import ChatOllama
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import os


load_dotenv()

# Initialize FastMCP and LLM
mcp = FastMCP("CaseRetrieval")
llm = ChatOllama(model="llama3")

@mcp.tool()
def find_similar_cases(query: str):
    """
    Fast version: use Tavily Search to find UK legal cases and summarize them.
    Avoids embeddings and FAISS to keep response fast.
    """
    print(f"🔍 Searching for: {query}")

    # Tavily search
    try:
        search = TavilySearchResults(
            k=3, 
            tavily_api_key=os.getenv("TAVILY_API_KEY")
        )
        results = search.run(f"{query} UK family law judgment OR divorce case OR financial provision case")
    except Exception as e:
        return {"error": f"Tavily search failed: {str(e)}"}

    if not results:
        return {"message": "No cases found", "sources": []}

    
    snippets = []
    sources = []
    for r in results:
        if isinstance(r, dict):
            snippet = r.get("content") or r.get("snippet") or ""
            if snippet.strip():
                snippets.append(snippet)
            if r.get("url"):
                sources.append(r.get("url"))

    if not snippets:
        return {"message": "No retrievable content", "sources": sources}

    combined_text = "\n\n".join(snippets[:3])  # use top 3 snippets for speed

    
    prompt = PromptTemplate(
        template=(
            "You are a legal AI assistant. Based on the following retrieved case summaries, "
            "give a concise overview related to the query.\n\n"
            "Query: {query}\n\n"
            "Cases:\n{cases}\n\n"
            "Summary:"
        ),
        input_variables=["query", "cases"]
    )

    chain = LLMChain(llm=llm, prompt=prompt)
    summary = chain.run(query=query, cases=combined_text).strip()

    # Return summary + URLs
    return {
        "query": query,
        "summary": summary,
        "sources": sources
    }

if __name__ == "__main__":
    mcp.run(transport="sse", host="127.0.0.1", port=8002, path="/sse")


