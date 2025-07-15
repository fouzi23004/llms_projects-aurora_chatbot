import json
import os
from loaders.html_scraper import scrape_ubuntu_documentation
from utils.convert import convert_to_documents
from vectorstore.embed_store import store_documents


url = "https://documentation.ubuntu.com/microcloud/latest/lxd/"

try:
        # Start with a smaller number of pages to test
        chunks = scrape_ubuntu_documentation(url, max_pages=169)

        if chunks:
            # Display sample results
            print(f"\nSample chunks:")
            for i, chunk in enumerate(chunks[:3]):
                print(f"\n--- Chunk {i + 1} ---")
                print(f"Source: {chunk['source_url']}")
                print(f"Title: {chunk['title']}")
                print(f"Text: {chunk['text'][:200]}...")

            # Save to file (optional)

            output_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
            os.makedirs(output_dir, exist_ok=True)  # <-- ensures the directory exists

            output_file = os.path.join(output_dir, 'ubuntu_microcloud_chunks2.json')
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(chunks, f, indent=2, ensure_ascii=False)
            print(f"\nSaved {len(chunks)} chunks to {output_file}")
            docs = convert_to_documents(chunks)

            # Show memory usage

        else:
            print("No chunks were generated. Check the scraping process.")

except MemoryError:
        print("Memory error occurred. Try reducing max_pages or chunk_size.")
except Exception as e:
        print(f"Unexpected error: {e}")
        import traceback

        traceback.print_exc()

store_documents(docs)
