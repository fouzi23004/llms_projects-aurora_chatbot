import json
import os
import tracemalloc
from dotenv import load_dotenv

from loaders.html_scraper import scrape_ubuntu_documentation
from utils.convert import convert_to_documents
from vectorstore.embed_store import store_documents


# Load .env variables
load_dotenv()

# Load values from environment
MAX_PAGES = int(os.getenv("MAX_PAGES", 100))
CHUNK_OUTPUT_FILE = os.getenv("CHUNK_OUTPUT_FILE", "ubuntu_microcloud_chunks2.json")
UBUNTU_DOC_URL = os.getenv("UBUNTU_DOC_URL", "https://documentation.ubuntu.com/lxd/latest/")

try:
    # Start memory tracking
    tracemalloc.start()

    output_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, CHUNK_OUTPUT_FILE)

    # Initialize chunks
    chunks = []

    if os.path.isfile(output_file) and os.path.getsize(output_file) > 0:
        
        print(f"[INFO] File {output_file} exists. Loading chunks from file...")
        with open(output_file, 'r', encoding='utf-8') as f:
            chunks = json.load(f)
    else:
        print(output_file)
        print(f"[INFO] File not found or empty. Scraping documentation from: {UBUNTU_DOC_URL}")
        chunks = scrape_ubuntu_documentation(UBUNTU_DOC_URL, max_pages=MAX_PAGES)

        if chunks:
            print(f"\n[INFO] Sample chunks:")
            for i, chunk in enumerate(chunks[:3]):
                print(f"\n--- Chunk {i + 1} ---")
                print(f"Source: {chunk['source_url']}")
                print(f"Title: {chunk['title']}")
                print(f"Text: {chunk['text'][:200]}...")

            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(chunks, f, indent=2, ensure_ascii=False)

            print(f"\n[INFO] Saved {len(chunks)} chunks to {output_file}")
        else:
            print("[WARN] No chunks were generated. Check the scraping process.")

    # Convert and store
    docs = convert_to_documents(chunks)
    print(f"[INFO] Converted to {len(docs)} LangChain documents.")

    store_documents(docs)

    current, peak = tracemalloc.get_traced_memory()
    print(f"\n[MEMORY] Current: {current / 1_000_000:.2f} MB; Peak: {peak / 1_000_000:.2f} MB")
    tracemalloc.stop()

except MemoryError:
    print("[ERROR] Memory error occurred. Try reducing MAX_PAGES.")
except Exception as e:
    print(f"[ERROR] Unexpected error: {e}")
    import traceback
    traceback.print_exc()
