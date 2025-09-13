# populate_chroma.py

import json
import time
import logging
from typing import List, Dict
import google.generativeai as genai
import chromadb
from tqdm import tqdm

# --- Configuration ---
# Your API keys from the Google AI Studio
GEMINI_API_KEYS = [
    "AIzaSyAFHriOAJQFwaVcSgAXpdyUW_DvIPdWQd4",
    "AIzaSyA2eGfn-HYFgVVU3146LQMqD_QVIf_7snY", 
    "AIzaSyBCN3LRmoajoEbmd9rxBO7cfDugoWqQG40",
    "AIzaSyAwjBzdYJVQUehCCLigvjNKOEb3Szo6HkY",
    "AIzaSyCWvK_GYiy2ZpITZxpWb7453zFzoN_VqmM",
    "AIzaSyAWEZuMIu7Yn1ISlXG9SKDVEaE96ACSjHo"
]

# Your ChromaDB Cloud credentials
CHROMADB_CONFIG = {
    'api_key': 'ck-GgDLCLEeXKAEhpWCWMDwcFP1hEVH4gpqhii25vw98XSC',
    'tenant': '94df3293-175e-443f-994a-22655697ffc9',
    'database': 'atlan'
}
COLLECTION_NAME = "new_atlan_docs"

# The correct, latest model for creating embeddings for retrieval.
EMBEDDING_MODEL = "gemini-embedding-exp-03-07"

# Gemini API free tier allows about 15 requests per minute.
# We'll set it to 12 to be safe, meaning 1 request every 5 seconds per key.
REQUESTS_PER_MINUTE_PER_KEY = 12

# Processing settings
BATCH_SIZE = 50 # How many documents to embed in a single API call
CHUNK_FILES_TO_PROCESS = ["docs_chunks.jsonl", "dev_chunks.jsonl"]

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class RateLimitedEmbeddingGenerator:
    """
    Manages embedding generation with intelligent rate limiting and API key rotation.
    """
    def __init__(self, api_keys: List[str], model_name: str, rpm_per_key: int):
        self.api_keys = api_keys
        self.model_name = model_name
        self.key_request_times = {key: [] for key in api_keys}
        self.requests_per_minute = rpm_per_key
        self.current_key_index = 0

    def _get_next_api_key(self) -> str:
        """Rotates to the next API key."""
        self.current_key_index = (self.current_key_index + 1) % len(self.api_keys)
        return self.api_keys[self.current_key_index]

    def _enforce_rate_limit(self, api_key: str):
        """Waits if the current key has exceeded its per-minute request limit."""
        now = time.time()
        # Get timestamps of requests made in the last 60 seconds
        recent_requests = [t for t in self.key_request_times[api_key] if now - t < 60]
        self.key_request_times[api_key] = recent_requests
        
        if len(recent_requests) >= self.requests_per_minute:
            # Time to wait is until the oldest request is more than 60s old
            time_to_wait = 60 - (now - recent_requests[0])
            logger.info(f"Rate limit for key ending in ...{api_key[-4:]} reached. Waiting for {time_to_wait:.2f} seconds.")
            time.sleep(time_to_wait)
        
        # Record the current request time
        self.key_request_times[api_key].append(time.time())

    def generate_embeddings(self, texts: List[str], max_retries: int = 3) -> List[List[float]]:
        """Generates embeddings for a batch of texts with retries and rate limiting."""
        for attempt in range(max_retries):
            api_key = self._get_next_api_key()
            try:
                self._enforce_rate_limit(api_key)
                genai.configure(api_key=api_key)
                
                response = genai.embed_content(
                    model=self.model_name,
                    content=texts,
                    task_type="RETRIEVAL_DOCUMENT",
                    output_dimensionality=768 # Crucial for consistency
                )
                return response['embedding']
                
            except Exception as e:
                logger.error(f"Error on attempt {attempt + 1} with key ...{api_key[-4:]}: {e}")
                if "quota" in str(e).lower():
                    logger.warning("Quota error detected. Trying next key.")
                    continue # Immediately try the next key
                time.sleep(2 ** attempt) # Exponential backoff for other errors
        
        raise Exception(f"Failed to generate embeddings for batch after {max_retries} retries.")


def populate_chromadb_pipeline():
    """
    Main pipeline to load, embed, and store documents in ChromaDB.
    """
    logger.info("🚀 Starting ChromaDB Population Pipeline...")

    # --- 1. Load and Prepare Chunks ---
    chunks_to_process = []
    for filename in CHUNK_FILES_TO_PROCESS:
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                file_chunks = [json.loads(line) for line in f]
                chunks_to_process.extend(file_chunks)
                logger.info(f"Loaded {len(file_chunks)} chunks from {filename}.")
        except FileNotFoundError:
            logger.error(f"❌ Critical Error: Chunk file '{filename}' not found. Please run the scraper first.")
            return

    if not chunks_to_process:
        logger.warning("No chunks found to process. Exiting.")
        return

    total_chunks = len(chunks_to_process)
    logger.info(f"Total chunks to process: {total_chunks}")

    # --- 2. Connect to ChromaDB and Handle Existing Collection ---
    try:
        chroma_client = chromadb.CloudClient(**CHROMADB_CONFIG)
        logger.info("Successfully connected to ChromaDB Cloud.")
        
        if COLLECTION_NAME in [c.name for c in chroma_client.list_collections()]:
            logger.warning(f"Collection '{COLLECTION_NAME}' already exists.")
            user_input = input(f"Do you want to DELETE and recreate the collection '{COLLECTION_NAME}'? This is permanent. (yes/no): ").lower()
            if user_input == 'yes':
                logger.info(f"Deleting existing collection: {COLLECTION_NAME}")
                chroma_client.delete_collection(name=COLLECTION_NAME)
                collection = chroma_client.create_collection(name=COLLECTION_NAME)
                logger.info(f"Successfully created new empty collection: {COLLECTION_NAME}")
            else:
                logger.info("Exiting pipeline. Please run again when ready.")
                return
        else:
            collection = chroma_client.create_collection(name=COLLECTION_NAME)
            logger.info(f"Successfully created new collection: {COLLECTION_NAME}")

    except Exception as e:
        logger.critical(f"❌ Failed to connect to or configure ChromaDB: {e}")
        return

    # --- 3. Process Batches and Upload ---
    embedding_generator = RateLimitedEmbeddingGenerator(GEMINI_API_KEYS, EMBEDDING_MODEL, REQUESTS_PER_MINUTE_PER_KEY)
    
    with tqdm(total=total_chunks, desc="Embedding and Uploading Chunks") as pbar:
        for i in range(0, total_chunks, BATCH_SIZE):
            batch_chunks = chunks_to_process[i:i + BATCH_SIZE]
            batch_texts = [chunk['content'] for chunk in batch_chunks]

            try:
                # Generate embeddings for the current batch
                embeddings = embedding_generator.generate_embeddings(batch_texts)

                # Prepare data for ChromaDB
                ids = [chunk['chunk_id'] for chunk in batch_chunks]
                documents = [chunk['content'] for chunk in batch_chunks]
                metadatas = [{"url": chunk['url'], "title": chunk['title']} for chunk in batch_chunks]

                # Add the batch to the collection
                collection.add(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)
                pbar.update(len(batch_chunks))
                
            except Exception as e:
                logger.error(f"Failed to process batch starting at index {i}. Skipping. Error: {e}")
                pbar.update(len(batch_chunks)) # Still update progress bar to not get stuck

    logger.info(f"✅ Pipeline complete! Processed and uploaded {collection.count()} chunks to ChromaDB.")

if __name__ == "__main__":
    populate_chromadb_pipeline()