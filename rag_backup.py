
import os
import time
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

# Primary model first; fallback is attempted if it is unavailable.
GENERATION_MODELS = [
    "gemini-3.5-flash",
    "gemini-3.8-flash",
]


def get_embedding(text):
    """Convert text into a Gemini embedding."""
    result = client.models.embed_content(
        model="gemini-embedding-001",
        contents=text
    )
    return result.embeddings[0].values


def add_document(text, source):
    """Store a document and its embedding."""
    if not text.strip():
        return

    embedding = get_embedding(text)

    collection.upsert(
        ids=[source],
        documents=[text],
        embeddings=[embedding],
        metadatas=[{"source": source}]
    )


def ingest_pdf(pdf_path):
    """Extract text from a PDF and index it."""
    pdf_path = Path(pdf_path)
    reader = PdfReader(str(pdf_path))

    pages = []

    for page_number, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text() or ""

        if page_text.strip():
            pages.append(
                f"[Page {page_number}]\n{page_text}"
            )

    full_text = "\n\n".join(pages)

    if not full_text.strip():
        return (
            "No extractable text found. "
            "The PDF may contain scanned images."
        )

    add_document(full_text, source=pdf_path.name)

    return (
        f"Successfully indexed {pdf_path.name} "
        f"({len(reader.pages)} pages)."
    )


def search_documents(question, n=3):
    """Retrieve relevant documents from ChromaDB."""
    count = collection.count()

    if count == 0:
        return []

    embedding = get_embedding(question)

    results = collection.query(
        query_embeddings=[embedding],
        n_results=min(n, count)
    )

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]

    return [
        {
            "text": document,
            "metadata": metadata or {}
        }
        for document, metadata in zip(documents, metadatas)
    ]


def generate_answer(prompt):
    """Retry temporary failures and try the fallback model."""
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

                # Exponential backoff: 2 seconds, then 4 seconds.
                if attempt < 2:
                    wait_seconds = 2 ** (attempt + 1)
                    time.sleep(wait_seconds)

                else:
                    # Move to the next model after the final retry.
                    break

    return (
        "InsightIQ could not generate an answer because Gemini "
        "is temporarily unavailable or overloaded. Please try "
        f"again later. Technical details: {last_error}"
    )


def answer_question(question):
    """Answer using retrieved document context."""
    retrieved = search_documents(question)

    if not retrieved:
        return "No documents are available to answer this question."

    context = "\n\n".join(
        f"Source: {item['metadata'].get('source', 'Unknown')}\n"
        f"{item['text']}"
        for item in retrieved
    )

    prompt = f"""
You are InsightIQ, a business intelligence assistant.

Answer the question using only the supplied context.
If the context does not contain the answer, say so.
Do not invent facts or numerical results.
Cite the source filename in your answer.

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
        add_document(sample_text, source=sample_path.name)
        print("Sample document indexed.")
    else:
        print("Sample document not found.")

    print("\nInsightIQ RAG pipeline is ready.")
    print(
        "PDF files can be indexed using "
        "ingest_pdf(pdf_path)."
    )

    question = "Why did revenue decline in Q3?"
    print("\nQuestion:", question)
    print("\nInsightIQ answer:")
    print(answer_question(question))
