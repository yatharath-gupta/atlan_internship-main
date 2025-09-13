# scrpae2.py (New Version)

import requests
from bs4 import BeautifulSoup, NavigableString
from urllib.parse import urljoin, urlparse
import json
import re
from typing import Set, List, Dict, Optional
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def intelligent_content_extractor(soup: BeautifulSoup) -> str:
    """
    Extracts the main textual content from a BeautifulSoup object,
    focusing on article-like content and removing noise.
    """
    # Try to find the main content area using common tags and roles.
    # This list can be expanded based on site structure.
    main_content_selectors = ["main", "article", "[role='main']", ".main-content", ".content"]
    
    main_body = None
    for selector in main_content_selectors:
        main_body = soup.select_one(selector)
        if main_body:
            break
    
    if not main_body:
        main_body = soup.body # Fallback to the whole body if no main content area is found

    # Remove known noise elements like navbars, headers, footers, scripts, etc.
    for tag in main_body.select("nav, header, footer, script, style, .sidebar, .toc, [role='navigation']"):
        tag.decompose()

    # Get text and clean it up
    lines = []
    for element in main_body.find_all(['h1', 'h2', 'h3', 'p', 'li', 'code']):
        text = element.get_text(" ", strip=True)
        if text:
            # Add markdown-like prefixes for structure
            if element.name.startswith('h'):
                lines.append(f"\n## {text}\n")
            else:
                lines.append(text)
    
    full_text = "\n".join(lines)
    
    # Final cleanup of excessive newlines and whitespace
    full_text = re.sub(r'\n{3,}', '\n\n', full_text)
    full_text = re.sub(r'\s{2,}', ' ', full_text)
    
    return full_text.strip()

def semantic_chunker(text: str, max_chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """
    Splits a long text into smaller, semantically meaningful chunks.
    This version tries to split on paragraphs.
    """
    if not text:
        return []

    # Split by paragraphs (double newlines)
    paragraphs = text.split('\n\n')
    chunks = []
    current_chunk = ""

    for p in paragraphs:
        if not p.strip():
            continue
            
        # If adding the next paragraph fits, add it
        if len(current_chunk) + len(p) + 1 <= max_chunk_size:
            current_chunk += ("\n\n" + p if current_chunk else p)
        # If the paragraph itself is too big, split it hard
        elif len(p) > max_chunk_size:
            if current_chunk:
                chunks.append(current_chunk)
            
            # Split the large paragraph by sentences or words
            words = p.split()
            sub_chunk = ""
            for word in words:
                if len(sub_chunk) + len(word) + 1 > max_chunk_size:
                    chunks.append(sub_chunk)
                    sub_chunk = word
                else:
                    sub_chunk += (" " + word if sub_chunk else word)
            if sub_chunk:
                chunks.append(sub_chunk)
            current_chunk = ""
        # Otherwise, the paragraph doesn't fit, so finalize the current chunk
        else:
            chunks.append(current_chunk)
            # Start the new chunk with an overlap from the end of the last one
            overlap_text = current_chunk.split()[-overlap:]
            current_chunk = " ".join(overlap_text) + "\n\n" + p

    if current_chunk:
        chunks.append(current_chunk)
        
    return chunks

def scrape_and_chunk_site(start_url: str, max_pages: int = 500, output_file: str = "chunked_documents.jsonl"):
    """
    Scrapes a website starting from a URL, extracts clean content,
    chunks it, and saves it to a JSONL file.
    """
    domain = urlparse(start_url).netloc
    urls_to_visit = [start_url]
    visited_urls: Set[str] = set()
    all_chunks: List[Dict] = []
    
    logger.info(f"Starting scrape for domain: {domain}")

    while urls_to_visit and len(visited_urls) < max_pages:
        url = urls_to_visit.pop(0)
        if url in visited_urls:
            continue
        
        visited_urls.add(url)
        logger.info(f"Scraping [{len(visited_urls)}/{max_pages}]: {url}")

        try:
            response = requests.get(url, timeout=10)
            if response.status_code != 200 or 'text/html' not in response.headers.get('Content-Type', ''):
                continue
            
            soup = BeautifulSoup(response.text, "html.parser")
            title = soup.title.string.strip() if soup.title else "No Title"
            
            # 1. Intelligent Content Extraction
            clean_text = intelligent_content_extractor(soup)
            
            if not clean_text or len(clean_text) < 100: # Skip pages with very little content
                continue

            # 2. Semantic Chunking
            chunks = semantic_chunker(clean_text)
            
            for i, chunk_content in enumerate(chunks):
                chunk_data = {
                    "url": url,
                    "title": title,
                    "content": chunk_content,
                    "chunk_id": f"{url}#{i}",
                    "token_count": len(chunk_content.split()),
                    "chunk_index": i
                }
                all_chunks.append(chunk_data)

            # Find new links to visit on the same domain
            for link in soup.find_all("a", href=True):
                full_url = urljoin(url, link["href"]).split('#')[0] # Remove fragments
                if urlparse(full_url).netloc == domain and full_url not in visited_urls:
                    urls_to_visit.append(full_url)

        except requests.RequestException as e:
            logger.error(f"Failed to fetch {url}: {e}")
        except Exception as e:
            logger.error(f"Error processing {url}: {e}")

    # 3. Save to file
    with open(output_file, "w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk) + "\n")
    
    logger.info(f"✅ Scraping and chunking complete. Saved {len(all_chunks)} chunks to {output_file}")


if __name__ == "__main__":
    # Define the starting points for scraping
    DOCS_START_URL = "https://docs.atlan.com/"
    DEV_START_URL = "https://developer.atlan.com/concepts/"
    
    # It's better to scrape into separate files to manage them easily
    # scrape_and_chunk_site(DOCS_START_URL, max_pages=1500, output_file="docs_chunks.jsonl")
    scrape_and_chunk_site(DEV_START_URL, max_pages=1500, output_file="dev_chunks.jsonl")
    
    # You would then run your embedding script on BOTH of these files.
    # To combine them for embedding, you can create a small helper script or just run it twice.
    print("\nNext steps:")
    print("1. Review 'docs_chunks.jsonl' and 'dev_chunks.jsonl' to ensure content quality.")
    print("2. Run your embedding script on both files to populate ChromaDB with the new, high-quality chunks.")