"""
memory/vector_memory.py
------------------------
PURPOSE:
    Provides long-term memory for the sales pipeline agent using ChromaDB.
    
    This module implements a Retrieval-Augmented Generation (RAG) pattern:
    1. Retrieval: Before researching a company, we query ChromaDB for similar past
       leads using semantic search.
    2. Augmentation: If similar leads are found, their profiles (industry, pain
       points, overview, and messaging templates) are passed as context to the 
       researcher agent.
    3. Generation: The researcher uses this context to align research, extract
       domain-specific insights, and formulate higher-quality briefs.
    4. Storage: After a pipeline completes, the new company profile is saved
       into ChromaDB as a vector embedding for future recall.

DEPENDENCY:
    ChromaDB client stores persistent vector data in `./data/chroma_db/`.
"""

import os
import sys
import chromadb
from pathlib import Path
from datetime import datetime, timezone

# Ensure data directory exists
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = _PROJECT_ROOT / "data" / "chroma_db"
os.makedirs(DB_PATH, exist_ok=True)

# Initialize ChromaDB persistent client
# We use chromadb.PersistentClient to persist embeddings to disk across script runs.
# By default, ChromaDB uses the local sentence-transformer model for embedding generation.
client = chromadb.PersistentClient(path=str(DB_PATH))
collection = client.get_or_create_collection(name="lead_memory")


def log_action(agent_name: str, action: str, details: str = "") -> None:
    """Log vector memory actions to agent_log.txt with a UTC timestamp."""
    timestamp = datetime.now(timezone.utc).isoformat()
    log_line = f"[{timestamp}] {agent_name} | {action} | {details}\n"
    try:
        log_file = _PROJECT_ROOT / "agent_log.txt"
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(log_line)
    except Exception as e:
        sys.stderr.write(f"Logging error in vector memory: {e}\n")


def store_lead_memory(company: str, research_dict: dict, score: int, email_draft: str) -> None:
    """Stores a company profile as a vector embedding for future recall.
    
    This function compiles the structured research results, pain points, score,
    and output outreach drafts into a single comprehensive text document. 
    ChromaDB embeds this document, allowing future semantic queries to match
    against any of its contents.
    
    Args:
        company: The name of the company (e.g. "Stripe").
        research_dict: The ResearchResult dict from lead_researcher.
        score: The priority score assigned by lead_scorer.
        email_draft: The outreach email body draft text.
    """
    overview = research_dict.get("overview", "")
    industry = research_dict.get("industry", "")
    pain_points = research_dict.get("pain_points", [])
    
    # Clean pain points formatting
    if isinstance(pain_points, list):
        pain_points_str = ", ".join(pain_points)
    else:
        pain_points_str = str(pain_points)

    # Construct document text for embedding
    document = (
        f"Company: {company}\n"
        f"Industry: {industry}\n"
        f"Overview: {overview}\n"
        f"Pain Points: {pain_points_str}\n"
        f"Lead Score: {score}\n"
        f"Email Draft Preview: {email_draft[:300]}"
    )

    # Construct metadata for structured filtering if needed in the future
    metadata = {
        "company": company,
        "industry": industry,
        "score": int(score)
    }

    # Upsert into ChromaDB collection.
    # We use upsert so that re-running the pipeline on the same company updates
    # the existing vector rather than producing duplicates or throwing errors.
    collection.upsert(
        documents=[document],
        metadatas=[metadata],
        ids=[company]
    )
    
    log_action("VectorMemory", "STORE_LEAD_SUCCESS", f"company={company}, industry={industry}")


def recall_similar_leads(company_name: str, n: int = 3) -> list[dict]:
    """Searches for similar past leads by company name, industry, or pain points.
    
    Uses semantic similarity search (cosine distance/L2) over the embedded vector
    space to return the top n leads that resemble the queried company profile.
    
    Args:
        company_name: The company name to search similar leads for.
        n: Maximum number of similar leads to return.
        
    Returns:
        List of matching leads, each containing:
            - id: The company ID (name)
            - document: The stored textual profile
            - metadata: Stored key-value metadata dict
    """
    # Query ChromaDB collection by text
    results = collection.query(
        query_texts=[company_name],
        n_results=n + 1  # Query n+1 in case the company itself is returned
    )

    matches = []
    if results and results.get("documents") and results["documents"][0]:
        docs = results["documents"][0]
        metas = results["metadatas"][0] if results.get("metadatas") else [None] * len(docs)
        ids = results["ids"][0] if results.get("ids") else [None] * len(docs)

        for doc, meta, doc_id in zip(docs, metas, ids):
            # Skip self-match (if the past lead we find is the same company we are researching now)
            if doc_id.lower() == company_name.lower():
                continue
            matches.append({
                "id": doc_id,
                "document": doc,
                "metadata": meta
            })

    # Return top n results
    return matches[:n]


def get_memory_stats() -> int:
    """Returns the total number of leads stored in memory."""
    try:
        return collection.count()
    except Exception:
        return 0
