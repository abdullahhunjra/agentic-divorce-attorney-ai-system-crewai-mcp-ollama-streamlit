# retrieval_pipeline.py
import json
import re
import numpy as np
from pathlib import Path
from typing import List, Optional
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate


# CONFIGURATION
# =====================================
CHUNKS_JSON = Path("data/laws/UK_Divorce_Act_chunks_metadata.json")
INDEX_DIR = Path("vector_stores/faiss_divorce_act")

EMBED_MODEL = "BAAI/bge-base-en-v1.5"
RERANKER_MODEL = "BAAI/bge-reranker-base"
OLLAMA_MODEL = "llama3"

TOP_K = 30
ALPHA = 0.7



# MODEL LOAD
# =====================================
embedder = SentenceTransformer(EMBED_MODEL)
reranker = CrossEncoder(RERANKER_MODEL)
llm = OllamaLLM(model=OLLAMA_MODEL)
embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

with open(CHUNKS_JSON, "r", encoding="utf-8") as f:
    CHUNKS = json.load(f)

db = FAISS.load_local(str(INDEX_DIR), embeddings, allow_dangerous_deserialization=True)
bm25 = BM25Okapi([c["text"].lower().split() for c in CHUNKS])



# QUERY REWRITER
# =====================================
rewrite_prompt = ChatPromptTemplate.from_template("""
You are a query rewriter for a legal retrieval system.
Rewrite the query to be specific and legally contextualized within the Matrimonial Causes Act 1973 only.
Return JSON:
{{ "rewritten": "..." }}
Query: {user_query}
""")

rewrite_chain = rewrite_prompt | llm


def rewrite_query(user_query: str) -> str:
    """Use LLM to clarify or expand the query."""
    try:
        raw = rewrite_chain.invoke({"user_query": user_query}).strip()
        match = re.search(r"\{.*\}", raw, re.S)
        data = json.loads(match.group(0)) if match else {"rewritten": user_query}
        return data.get("rewritten", user_query)
    except Exception:
        return user_query


# RETRIEVAL FUNCTION
# =====================================
def retrieve_relevant_chunks(query: str, top_k: int = 5, use_rewriter: bool = True) -> List[dict]:
    """Return top relevant chunks (with metadata) using hybrid + reranking pipeline."""

    # Rewrite query (optional)
    rewritten = rewrite_query(query) if use_rewriter else query

    # Lexical (BM25)
    tokenized_query = rewritten.lower().split()
    bm25_scores = bm25.get_scores(tokenized_query)

    # Dense (FAISS)
    dense_results = db.similarity_search_with_score(rewritten, k=len(CHUNKS))
    dense_scores = np.zeros(len(CHUNKS))
    for doc, score in dense_results:
        idx = next(
            (i for i, c in enumerate(CHUNKS) if c["text"].strip() == doc.page_content.strip()), None
        )
        if idx is not None:
            dense_scores[idx] = 1 / (1 + score)

    
    hybrid_scores = ALPHA * dense_scores + (1 - ALPHA) * (bm25_scores / np.max(bm25_scores))

   
    KEY_TERMS = [
        "unreasonable behaviour", "adultery", "separation",
        "desertion", "financial relief", "maintenance", "decree nisi"
    ]
    for i, c in enumerate(CHUNKS):
        text_lower = c["text"].lower()
        if any(term in rewritten.lower() for term in KEY_TERMS if term in text_lower):
            hybrid_scores[i] *= 1.2

    
    top_idx = np.argsort(hybrid_scores)[::-1][:top_k * 3]
    candidates = [(CHUNKS[i], hybrid_scores[i]) for i in top_idx if hybrid_scores[i] > 0]

    
    if candidates:
        texts = [c[0]["text"] for c in candidates]
        pairs = [(rewritten, t) for t in texts]
        rerank_scores = reranker.predict(pairs, convert_to_numpy=True)

        for i, s in enumerate(rerank_scores):
            candidates[i][0]["rerank_score"] = float(s)
            candidates[i] = (candidates[i][0], candidates[i][1], s)

        sorted_final = sorted(candidates, key=lambda x: x[2], reverse=True)[:top_k]
    else:
        sorted_final = []

    
    results = []
    for chunk, hybrid_score, rerank_score in sorted_final:
        results.append({
            "text": chunk["text"],
            "metadata": chunk["metadata"],
            "hybrid_score": float(hybrid_score),
            "rerank_score": float(rerank_score),
        })
    return results


if __name__ == "__main__":
    q = "What are the grounds for divorce under the Matrimonial Causes Act 1973?"
    for r in retrieve_relevant_chunks(q):
        print(r["metadata"])
        print(r["text"][:400], "...\n")
