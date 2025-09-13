# enrich_chroma.py

import logging
import sys
import chromadb

# This script prepares the ChromaDB knowledge base for the advanced Retrieval Agent.
# It iterates through all stored documents and adds a 'domain' metadata field
# based on the document's source URL. This enables filtered vector searches,
# significantly improving retrieval speed and relevance.

# try:
#     # Fix for SQLite version issues in some environments
#     __import__('pysqlite3')
#     sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
#     import chromadb
# except (ImportError, RuntimeError) as e:
#     print(f"FATAL: ChromaDB or pysqlite3 could not be imported. Error: {e}")
#     print("Please ensure you have run 'pip install -r requirements.txt'.")
#     sys.exit(1)

# --- Configuration ---
# Replace with your ChromaDB credentials
CHROMADB_CONFIG = {
    'api_key': 'ck-GgDLCLEeXKAEhpWCWMDwcFP1hEVH4gpqhii25vw98XSC',
    'tenant': '94df3293-175e-443f-994a-22655697ffc9',
    'database': 'atlan'
}
COLLECTION_NAME = "new_atlan_docs"

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_domain_from_url(url: str) -> str:
    """Assigns a domain topic based on URL patterns."""
    if not isinstance(url, str) or not url:
        return "Product" # Default domain

    url_lower = url.lower()

    if 'developer.atlan.com' in url_lower:
        return 'API/SDK'
    if '/sso' in url_lower:
        return 'SSO'
    if '/connector' in url_lower:
        return 'Connector'
    if '/lineage' in url_lower:
        return 'Lineage'
    if '/glossary' in url_lower:
        return 'Glossary'
    if '/best-practices' in url_lower or '/playbooks' in url_lower:
        return 'Best Practices'
    if '/sensitive-data' in url_lower or '/pii' in url_lower:
        return 'Sensitive Data'
    
    return 'Product' # A safe fallback for general documentation

def main():
    """
    Main function to connect to ChromaDB, process documents, and add domain metadata.
    """
    logging.info("Starting ChromaDB metadata enrichment process...")

    try:
        client = chromadb.CloudClient(**CHROMADB_CONFIG)
        collection = client.get_collection(name=COLLECTION_NAME)
        logging.info(f"Successfully connected to ChromaDB and retrieved collection '{COLLECTION_NAME}'.")
    except Exception as e:
        logging.critical(f"Failed to connect to ChromaDB. Aborting. Error: {e}")
        return

    total_docs = collection.count()
    if total_docs == 0:
        logging.warning("The collection is empty. There are no documents to enrich.")
        return

    logging.info(f"Found {total_docs} documents to process.")

    # Fetch all documents in one go. For extremely large collections (>100k),
    # this should be done in batches using the offset parameter in collection.get().
    try:
        documents = collection.get(include=["metadatas"])
    except Exception as e:
        logging.error(f"Failed to retrieve documents from the collection. Error: {e}")
        return

    ids_to_update = []
    metadatas_to_update = []
    updated_count = 0

    for doc_id, metadata in zip(documents['ids'], documents['metadatas']):
        if metadata is None:
            continue

        source_url = metadata.get('url', '')
        current_domain = metadata.get('domain')
        
        new_domain = get_domain_from_url(source_url)
        
        # Only update if the domain is new or has changed
        if new_domain != current_domain:
            new_metadata = metadata.copy()
            new_metadata['domain'] = new_domain
            ids_to_update.append(doc_id)
            metadatas_to_update.append(new_metadata)
            updated_count += 1

    if not ids_to_update:
        logging.info("All documents already have up-to-date domain metadata. No updates needed.")
        return

    logging.info(f"Found {updated_count} documents that require a metadata update.")

    try:
        # Perform a single batch update for efficiency
        collection.update(
            ids=ids_to_update,
            metadatas=metadatas_to_update
        )
        logging.info(f"Successfully updated {len(ids_to_update)} documents in ChromaDB.")
    except Exception as e:
        logging.error(f"An error occurred during the batch update process. Error: {e}")

    logging.info("Metadata enrichment process finished.")

if __name__ == "__main__":
    main()