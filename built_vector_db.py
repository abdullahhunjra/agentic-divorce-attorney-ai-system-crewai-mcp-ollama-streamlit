import os
import re
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
from unstructured.partition.pdf import partition_pdf
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document


# Configurations of paths and model details
# ====================================
PDF_PATH = Path("data/laws/legal_document.pdf")
CLEAN_JSON = Path("data/laws/UK_Divorce_Act_clean.json")
CHUNKS_JSON = Path("data/laws/UK_Divorce_Act_chunks_metadata.json")
INDEX_DIR = Path("vector_stores/faiss_divorce_act")

SEMANTIC_MODEL = "all-MiniLM-L6-v2"
EMBED_MODEL = "BAAI/bge-base-en-v1.5"
OLLAMA_MODEL = "llama3"
SIMILARITY_THRESHOLD = 0.82



# Cleaning PDF using Unstructred
# ====================================
def clean_text_block(text: str) -> str:
    """
    Cleans a single text block from the UK Divorce Act PDF.
    Removes headers/footers, page markers, and stray formatting,
    but keeps all substantive legal text.
    """
    text = text.strip()

    #  Removing the common repeated headers/footers
    text = re.sub(r"c\.?\s*18\s*Matrimonial\s*Causes\s*Act\s*1973", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"Matrimonial\s*Causes\s*Act\s*1973", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"Page\s*\d+", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"Chapter\s*\d+", " ", text, flags=re.IGNORECASE)

    # Fixing the  formatting issues
    text = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", text)  # split jammed words
    text = re.sub(r"(\d)([A-Za-z])", r"\1 \2", text)   # 1973Act → 1973 Act
    text = re.sub(r"\s+", " ", text)
    text = text.replace(" .", ".").replace(" ,", ",")

    # Removing stray characters or long dashes
    text = re.sub(r"—+", "-", text)
    text = re.sub(r"–+", "-", text)
    text = text.strip(" -\t\n\r")

    return text


def clean_legal_pdf(pdf_path, output_json=CLEAN_JSON):
    """Cleans legal PDF and extracts structured elements."""
    pdf_path = Path(pdf_path)
    output_json = Path(output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)

    print(f"📄 Processing and cleaning: {pdf_path.name}")

    elements = partition_pdf(
        filename=str(pdf_path),
        strategy="hi_res",
        include_page_breaks=True,
        infer_table_structure=True
    )

    structured_data = []
    for i, el in enumerate(elements):
        if el.category in ["Header", "Footer"]:
            continue

        text = el.text.strip()
        if not text:
            continue

        text = clean_text_block(text)
        if not text:
            continue

        meta = el.metadata.to_dict() if el.metadata else {}
        structured_data.append({
            "id": i,
            "category": el.category,
            "text": text,
            "page_number": meta.get("page_number", None),
            "filename": pdf_path.name,
        })

    print(f"✅ Extracted {len(structured_data)} structured text elements")

    # Filter out trivial lines unless they’re titles or list items
    clean_data = []
    for item in structured_data:
        if len(item["text"]) < 25 and item["category"] not in ["Title", "ListItem"]:
            continue
        clean_data.append(item)

    print(f"✅ Retained {len(clean_data)} meaningful data elements")

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(clean_data, f, indent=2, ensure_ascii=False)

    print(f"💾 Clean structured data saved → {output_json}")
    return clean_data



# Semnatic chunking 
# ====================================
def detect_semantic_boundaries(text_blocks, embeddings, threshold=0.82):
    sims = cosine_similarity(embeddings[:-1], embeddings[1:]).diagonal()
    return [i + 1 for i, sim in enumerate(sims) if sim < threshold]


def group_by_boundaries(text_blocks, boundaries):
    chunks, start = [], 0
    for b in boundaries + [len(text_blocks)]:
        joined = " ".join([tb["text"] for tb in text_blocks[start:b]])
        chunks.append(joined)
        start = b
    return chunks



#  improved LLM assisted metadata extraction
# ====================================
def refine_chunks_and_extract_metadata(chunks, llm):
    """
    Uses the LLM to:
      - Ensure text chunks are self-contained
      - Generate structured metadata
      - Handles small batches to avoid context loss
    """

    # Few-shot examples
    EXAMPLES = [
        {
            "text": "Section 1 – Grounds for Divorce. A petition for divorce may be presented by either party to a marriage...",
            "metadata": {
                "clause_number": "Section 1",
                "title": "Grounds for Divorce",
                "summary": "Defines the grounds under which a divorce petition can be filed.",
                "category": "Grounds for Divorce",
                "legal_concepts": "petition, marriage, divorce, court",
                "page_reference": 2
            }
        },
        {
            "text": "Section 24 – Financial provision in connection with divorce proceedings...",
            "metadata": {
                "clause_number": "Section 24",
                "title": "Financial Relief",
                "summary": "Outlines financial remedies such as maintenance and property adjustment orders.",
                "category": "Financial Relief",
                "legal_concepts": "maintenance, property adjustment, court order",
                "page_reference": 12
            }
        }
    ]

    prompt_template = PromptTemplate.from_template("""
You are an expert UK legal analyst structuring sections of the Matrimonial Causes Act 1973.

Analyze each text chunk and return JSON ONLY with this exact format:
[
  {{
    "text": "<cleaned, self-contained text>",
    "metadata": {{
      "clause_number": "Section or Clause number if found",
      "title": "Short descriptive title",
      "summary": "1–2 line summary of content",
      "category": "Definitions | Grounds for Divorce | Judicial Separation | Financial Relief | Procedural | Miscellaneous | General Provisions",
      "legal_concepts": "comma-separated legal terms or entities",
      "page_reference": <integer or null>
    }}
  }}
]

Guidelines:
- Do NOT explain reasoning.
- If a field is not explicitly stated, infer logically.
- Use British legal language.
- Output STRICT valid JSON only.

Examples:
{examples}

Now process this chunk:
{chunk}
""")

    results = []
    for i, chunk in enumerate(tqdm(chunks, desc="🧠 Extracting metadata per chunk")):
        prompt = prompt_template.format(
            examples=json.dumps(EXAMPLES, ensure_ascii=False, indent=2),
            chunk=json.dumps(chunk, ensure_ascii=False)
        )

        response = llm.invoke(prompt)
        try:
            data = json.loads(response.content)
            if isinstance(data, list) and "metadata" in data[0]:
                results.append(data[0])
            else:
                print(f"⚠️ Invalid structure in chunk {i}, fallback used.")
                raise ValueError
        except Exception:
            results.append({
                "text": chunk,
                "metadata": {
                    "clause_number": None,
                    "title": None,
                    "summary": None,
                    "category": "Uncategorized",
                    "legal_concepts": "",
                    "page_reference": None
                }
            })

    return results


# ====================================
# Hybrid chuNking pipeLine
# ====================================
def hybrid_chunking_with_metadata():
    print("⚙️ Starting hybrid semantic + metadata chunking...")

    data = clean_legal_pdf(PDF_PATH, CLEAN_JSON)
    text_blocks = [d for d in data if len(d["text"].strip()) > 30]
    print(f"📘 Using {len(text_blocks)} clean text blocks for chunking")

    # Embeddings for semantic boundary detection
    print("🧠 Generating sentence embeddings...")
    embed_model = SentenceTransformer(SEMANTIC_MODEL)
    embeddings = embed_model.encode(
        [tb["text"] for tb in text_blocks],
        normalize_embeddings=True,
        show_progress_bar=True
    )

    print("🔍 Detecting semantic boundaries...")
    boundaries = detect_semantic_boundaries(text_blocks, embeddings, SIMILARITY_THRESHOLD)
    semantic_chunks = group_by_boundaries(text_blocks, boundaries)
    print(f"🧩 Created {len(semantic_chunks)} semantic chunks")

    print("🤖 Refining chunks + extracting metadata with LLM (Ollama)...")
    llm = ChatOllama(model=OLLAMA_MODEL, temperature=0.1)
    refined_chunks = refine_chunks_and_extract_metadata(semantic_chunks, llm)
    print(f"✅ Final refined chunks: {len(refined_chunks)}")

    final_data = []
    for i, c in enumerate(refined_chunks):
        src_idx = min(i, len(text_blocks) - 1)
        base_meta = text_blocks[src_idx]
        c["metadata"]["page_number"] = base_meta["page_number"]
        c["metadata"]["source"] = base_meta["filename"]
        final_data.append(c)

    CHUNKS_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(CHUNKS_JSON, "w", encoding="utf-8") as f:
        json.dump(final_data, f, indent=2, ensure_ascii=False)

    print(f"💾 Saved refined metadata-enriched chunks → {CHUNKS_JSON}")
    return final_data


# ====================================
#  Building here the FAISS Index
# ====================================
def build_faiss_index(chunks):
    print("📦 Building FAISS vector store with metadata...")
    docs = [Document(page_content=c["text"], metadata=c["metadata"]) for c in chunks]

    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    vector_db = FAISS.from_documents(docs, embeddings)
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    vector_db.save_local(str(INDEX_DIR))
    print(f"✅ FAISS index saved to {INDEX_DIR}")





# ====================================
if __name__ == "__main__":
    chunks = hybrid_chunking_with_metadata()
    build_faiss_index(chunks)
    print("\n🎯 RAG-ready FAISS index with clean text + structured metadata built successfully.")
