# cli_rag/cli.py
import argparse
import os
from .pdf_loader import load_pdfs
from .store import get_db, is_doc_processed
from .agent import ask


def process_pdfs(pdf_paths):
    """Process and store any new PDFs not already in DB."""
    conn = get_db()

    for path in pdf_paths:
        if not os.path.exists(path):
            print(f"⚠️ File not found: {path}")
            continue

        chunks = load_pdfs([path])  # computes doc_id
        if not chunks:
            print(f"⚠️ No text extracted from: {path}")
            continue

        doc_id = chunks[0]["doc_id"]
        if is_doc_processed(doc_id):
            print(f"✅ Already processed: {path} ({doc_id})")
        else:
            print(f"📄 Processing new PDF: {path} ({doc_id})")
            # load_pdfs internally stores chunks + embeddings in DB via store.add_chunks_with_embeddings

    conn.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("pdfs", nargs="+", help="Paths to PDF files")
    args = parser.parse_args()

    print("Loading PDFs into database...")
    process_pdfs(args.pdfs)

    print("\nInteractive RAG CLI. Type 'exit' to quit.")
    while True:
        q = input("\n> ").strip()
        if q.lower() in ["exit", "quit"]:
            break
        ans = ask(q, show_sources=True)
        print("\n" + ans)


if __name__ == "__main__":
    main()
