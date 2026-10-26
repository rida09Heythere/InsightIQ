
import os
import chromadb
from dotenv import load_dotenv
from google import genai

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise ValueError("GEMINI_API_KEY not found in .env")

client = genai.Client(api_key=API_KEY)

# Persistent vector database
db = chromadb.PersistentClient(path="./chroma_db")

collection = db.get_or_create_collection(
    name="insightiq_documents"
)


def get_embedding(text):
    """Convert text into an embedding using Gemini."""
    result = client.models.embed_content(
        model="gemini-embedding-001",
        contents=text
    )
    return result.embeddings[0].values


def add_document(text, source):
    """Store a document and its embedding."""
    embedding = get_embedding(text)

    collection.upsert(
        ids=[source],
        documents=[text],
        embeddings=[embedding],
        metadatas=[{"source": source}]
    )


def search_documents(question, n=3):
    """Retrieve relevant document passages."""
    if collection.count() == 0:
        return []

    embedding = get_embedding(question)

    results = collection.query(
        query_embeddings=[embedding],
        n_results=min(n, collection.count())
    )

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]

    return [
        {"text": doc, "metadata": meta}
        for doc, meta in zip(documents, metadatas)
    ]


def answer_question(question):
    """Answer using retrieved context and Gemini."""
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

    response = client.models.generate_content(
        model="gemini-3.5-flash",
        contents=prompt
    )

    return response.text or "The model returned an empty response."


if __name__ == "__main__":
    file_path = "data/documents/company.txt"

    # Ingest the sample document if it exists
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as file:
            text = file.read()

        add_document(text, source="company.txt")
        print("Document indexed successfully.")
    else:
        print(f"Sample document not found: {file_path}")

    # Test the RAG pipeline
    question = "Why did revenue decline in Q3?"

    print("\nQuestion:", question)
    print("\nInsightIQ answer:")
    print(answer_question(question))
