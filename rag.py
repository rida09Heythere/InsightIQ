
import os
import time
import hashlib
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from google import genai
from pypdf import PdfReader

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise ValueError("GEMINI_API_KEY not found in .env")

client = genai.Client(api_key=API_KEY)

BASE_DIR = Path(__file__).resolve().parent

# Persistent vector database
db = chromadb.PersistentClient(
    path=str(BASE_DIR / "chroma_db")
)

collection = db.get_or_create_collection(
    name="insightiq_documents"
)

GENERATION_MODELS = [
    "gemini-3.5-flash",
    "gemini-3.8-flash",
]

CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200


def get_embedding(text):
    """Generate an embedding for a text passage."""
    result = client.models.embed_content(
        model="gemini-embedding-001",
        contents=text
    )
    return result.embeddings[0].values


def split_into_chunks(
    text,
    chunk_size=CHUNK_SIZE,
    overlap=CHUNK_OVERLAP
):
    """Split text into overlapping chunks."""
    text = text.strip()

    if not text:
        return []

    if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
        raise ValueError(
            "Chunk size must be positive and overlap smaller than chunk size."
        )

    chunks = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))

        if end < len(text):
            boundary = text.rfind(" ", start, end)
            if boundary > start:
                end = boundary

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        start = end - overlap

    return chunks


def add_document(text, source, page_number=None):
    """Chunk and index text with source metadata."""
    chunks = split_into_chunks(text)

    if not chunks:
        return 0

    ids = []
    documents = []
    metadatas = []
    embeddings = []

    for index, chunk in enumerate(chunks):
        unique_key = f"{source}:{page_number}:{index}"

        chunk_id = hashlib.sha256(
            unique_key.encode("utf-8")
        ).hexdigest()

        metadata = {
            "source": source,
            "chunk": index + 1
        }

        if page_number is not None:
            metadata["page"] = page_number

        ids.append(chunk_id)
        documents.append(chunk)
        metadatas.append(metadata)
        embeddings.append(get_embedding(chunk))

    collection.upsert(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas
    )

    return len(chunks)


def ingest_pdf(pdf_path):
    """Extract PDF pages and index their chunks."""
    pdf_path = Path(pdf_path)
    reader = PdfReader(str(pdf_path))

    extracted_pages = []

    # Extract first so unreadable PDFs don't erase existing records.
    for page_number, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text() or ""

        if page_text.strip():
            extracted_pages.append((page_number, page_text))

    if not extracted_pages:
        return (
            "No extractable text found. The PDF may be scanned. "
            "Existing indexed data was preserved."
        )

    source = pdf_path.name

    # Remove previous chunks for this source filename.
    existing = collection.get(
        where={"source": source},
        include=["metadatas"]
    )

    old_ids = existing.get("ids", [])

    if old_ids:
        collection.delete(ids=old_ids)

    try:
        total_chunks = 0

        for page_number, page_text in extracted_pages:
            total_chunks += add_document(
                page_text,
                source=source,
                page_number=page_number
            )

    except Exception:
        # Remove partial new records if indexing fails.
        current = collection.get(
            where={"source": source},
            include=["metadatas"]
        )

        current_ids = current.get("ids", [])

        if current_ids:
            collection.delete(ids=current_ids)

        raise

    return (
        f"Successfully indexed {source}: {total_chunks} chunks "
        f"from {len(reader.pages)} pages."
    )


def search_documents(question, n=5):
    """Retrieve relevant chunks plus opening document context."""
    if collection.count() == 0:
        return []

    embedding = get_embedding(question)

    results = collection.query(
        query_embeddings=[embedding],
        n_results=min(n, collection.count())
    )

    retrieved = {}

    result_ids = results.get("ids", [[]])[0]
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]

    for chunk_id, document, metadata in zip(
        result_ids, documents, metadatas
    ):
        retrieved[chunk_id] = {
            "text": document,
            "metadata": metadata or {}
        }

    # Include opening chunks to capture titles and key details.
    opening = collection.get(
        where={"chunk": 1},
        include=["documents", "metadatas"]
    )

    for chunk_id, document, metadata in zip(
        opening.get("ids", []),
        opening.get("documents", []),
        opening.get("metadatas", [])
    ):
        metadata = metadata or {}
        page = metadata.get("page")

        # Include document openings, not every page's first chunk.
        if page is None or page == 1:
            retrieved.setdefault(
                chunk_id,
                {
                    "text": document,
                    "metadata": metadata
                }
            )

    return list(retrieved.values())


def generate_answer(prompt):
    """Retry temporary generation errors and try a fallback model."""
    last_error = None

    for model in GENERATION_MODELS:
        for attempt in range(3):
            try:
                response = client.models.generate_content(
                    model=model,
                    contents=prompt
                )

                if response.text:
                    return response.text

                last_error = "Gemini returned an empty response."
                break

            except Exception as error:
                last_error = error
                error_text = str(error).lower()

                retryable = any(
                    term in error_text
                    for term in (
                        "503",
                        "unavailable",
                        "429",
                        "resource_exhausted",
                        "500",
                        "internal server error",
                        "deadline exceeded"
                    )
                )

                if not retryable:
                    raise

                if attempt < 2:
                    time.sleep(2 ** (attempt + 1))

    return (
        "InsightIQ could not generate an answer because Gemini "
        "is temporarily unavailable or overloaded. Please try "
        f"again later. Technical details: {last_error}"
    )


def answer_question(question):
    """Answer using retrieved evidence and general document understanding."""
    retrieved = search_documents(question)

    if not retrieved:
        return "No documents are available to answer this question."

    context_parts = []

    for item in retrieved:
        metadata = item["metadata"]
        source = metadata.get("source", "Unknown")
        page = metadata.get("page")

        citation = f"Source: {source}"

        if page is not None:
            citation += f", page {page}"

        context_parts.append(
            f"{citation}\n{item['text']}"
        )

    context = "\n\n".join(context_parts)

    prompt = f"""
You are InsightIQ, a document understanding and business
intelligence assistant.

Answer the user's question using all relevant supplied context.

DOCUMENT UNDERSTANDING RULES:
- Interpret headings, titles, tables, captions, URLs, usernames,
  labels, and surrounding text as meaningful evidence.
- Do not require information to appear under an explicitly
  labelled field. A name may appear as a heading; a company
  may be identified by a report title; a date may appear in a
  sentence or table.
- Understand the meaning and structure of the document rather
  than searching only for exact words from the question.
- Connect related information across passages when supported.
- Answer directly when sufficient evidence exists.
- If a conclusion is plausible but uncertain, explain the
  uncertainty instead of presenting it as a confirmed fact.
- If evidence is insufficient, say what is missing rather
  than inventing an answer.
- Never fabricate names, dates, statistics, or other details.
- Cite the source filename and page number when available.

CONTEXT:
{context}

QUESTION:
{question}
"""

    return generate_answer(prompt)


if __name__ == "__main__":
    sample_path = (
        BASE_DIR / "data" / "documents" / "company.txt"
    )

    if sample_path.exists():
        sample_text = sample_path.read_text(encoding="utf-8")
        count = add_document(
            sample_text,
            source=sample_path.name
        )
        print(f"Sample document indexed into {count} chunks.")
    else:
        print("Sample document not found.")

    print("\nInsightIQ RAG pipeline is ready.")
    print("PDF files can be indexed using ingest_pdf(pdf_path).")

    question = "Why did revenue decline in Q3?"
    print("\nQuestion:", question)
    print("\nInsightIQ answer:")
    print(answer_question(question))
