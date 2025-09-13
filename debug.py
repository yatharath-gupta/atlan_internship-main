import chromadb
import google.generativeai as genai
from typing import Dict, List



def debug_chromadb_collection(chromadb_config: Dict[str, str], collection_name: str = "new_atlan_docs"):
    """Debug ChromaDB collection to see what's stored"""
    
    print("🔍 Debugging ChromaDB Collection...")
    print("=" * 60)
    
    try:
        # Connect to ChromaDB
        client = chromadb.CloudClient(**chromadb_config)
        collection = client.get_collection(collection_name)
        
        # Get basic collection info
        print(f"Collection Name: {collection.name}")
        print(f"Collection Count: {collection.count()}")
        
        if collection.count() == 0:
            print("❌ Collection is empty! No embeddings found.")
            print("\nTroubleshooting steps:")
            print("1. Check if embedding generation script ran successfully")
            print("2. Verify ChromaDB credentials and collection name")
            print("3. Re-run the embedding generation pipeline")
            return False
        
        # Sample a few documents
        print(f"\n📄 Sample Documents (showing first 5):")
        sample_results = collection.get(limit=5, include=['documents', 'metadatas'])
        
        for i, (doc, metadata) in enumerate(zip(sample_results['documents'], sample_results['metadatas'])):
            print(f"\n--- Document {i+1} ---")
            print(f"URL: {metadata.get('url', 'N/A')}")
            print(f"Title: {metadata.get('title', 'N/A')}")
            print(f"Chunk ID: {metadata.get('chunk_id', 'N/A')}")
            print(f"Content preview: {doc[:200]}...")
        
        # Check if embeddings exist - COMPLETELY FIXED
        try:
            sample_with_embeddings = collection.get(limit=1, include=['embeddings'])
            embeddings_data = sample_with_embeddings.get('embeddings')
            
            if embeddings_data is not None and len(embeddings_data) > 0:
                first_embedding = embeddings_data[0]
                if first_embedding is not None:
                    embedding_dim = len(first_embedding)
                    print(f"\n✅ Embeddings found! Dimension: {embedding_dim}")
                else:
                    print(f"\n❌ No embeddings found in collection!")
                    return False
            else:
                print(f"\n❌ No embeddings found in collection!")
                return False
                
        except Exception as embed_error:
            print(f"\n❌ Error checking embeddings: {embed_error}")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Error accessing ChromaDB: {e}")
        return False

def test_embedding_generation(gemini_api_key: str, test_query: str = "How to create packages"):
    """Test if embedding generation is working"""
    
    print(f"\n🧪 Testing Embedding Generation...")
    print("=" * 60)
    
    try:
        genai.configure(api_key=gemini_api_key)
        
        # Test embedding generation
        response = genai.embed_content(
            model="models/text-embedding-004",
            content=test_query,
            task_type="RETRIEVAL_QUERY",
            output_dimensionality=768
        )
        
        embedding = response['embedding']
        print(f"✅ Embedding generation successful!")
        print(f"Query: {test_query}")
        print(f"Embedding dimension: {len(embedding)}")
        print(f"First 5 values: {embedding[:5]}")
        
        return embedding
        
    except Exception as e:
        print(f"❌ Embedding generation failed: {e}")
        return None

def test_similarity_search(chromadb_config: Dict[str, str], 
                          gemini_api_key: str,
                          test_query: str = "How to create packages",
                          collection_name: str = "new_atlan_docs"):
    """Test similarity search with different thresholds"""
    
    print(f"\n🔍 Testing Similarity Search...")
    print("=" * 60)
    
    try:
        # Generate query embedding
        embedding = test_embedding_generation(gemini_api_key, test_query)
        if not embedding:
            return
        
        # Connect to ChromaDB
        client = chromadb.CloudClient(**chromadb_config)
        collection = client.get_collection(collection_name)
        
        # Test with very low threshold (0.0)
        print(f"\n🔍 Searching for: '{test_query}'")
        
        for threshold in [0.0, 0.1, 0.2, 0.3, 0.5]:
            results = collection.query(
                query_embeddings=[embedding],
                n_results=10,
                include=['documents', 'metadatas', 'distances']
            )
            
            if results['documents'] and results['documents'][0]:
                filtered_results = []
                for doc, metadata, distance in zip(
                    results['documents'][0],
                    results['metadatas'][0],
                    results['distances'][0]
                ):
                    similarity = 1 - distance
                    if similarity >= threshold:
                        filtered_results.append((doc, metadata, similarity))
                
                print(f"Threshold {threshold}: {len(filtered_results)} results")
                
                # Show top result for this threshold
                if filtered_results and threshold == 0.0:  # Show details for lowest threshold
                    print(f"\nTop result details:")
                    doc, metadata, similarity = filtered_results[0]
                    print(f"Similarity: {similarity:.4f}")
                    print(f"URL: {metadata.get('url', 'N/A')}")
                    print(f"Title: {metadata.get('title', 'N/A')}")
                    print(f"Content: {doc[:200]}...")
            else:
                print(f"Threshold {threshold}: 0 results")
        
    except Exception as e:
        print(f"❌ Similarity search failed: {e}")

def main_debug():
    """Main debugging function"""
    
    # Your configuration
    GEMINI_API_KEYS = [
        "AIzaSyAFHriOAJQFwaVcSgAXpdyUW_DvIPdWQd4",
        "AIzaSyA2eGfn-HYFgVVU3146LQMqD_QVIf_7snY", 
        "AIzaSyAwjBzdYJVQUehCCLigvjNKOEb3Szo6HkY",
        "AIzaSyCWvK_GYiy2ZpITZxpWb7453zFzoN_VqmM"
    ]
    
    CHROMADB_CONFIG = {
        'api_key': 'ck-GgDLCLEeXKAEhpWCWMDwcFP1hEVH4gpqhii25vw98XSC',
        'tenant': '94df3293-175e-443f-994a-22655697ffc9',
        'database': 'atlan'
    }
    
    print("🚀 Starting ChromaDB Debug Session...")
    
    # Step 1: Check collection
    collection_ok = debug_chromadb_collection(CHROMADB_CONFIG)
    
    if not collection_ok:
        print("\n❌ Collection issues found. Please fix ChromaDB setup first.")
        return
    
    # Step 2: Test embedding generation
    embedding = test_embedding_generation(GEMINI_API_KEYS[0])
    
    if not embedding:
        print("\n❌ Embedding generation issues found. Please check API key.")
        return
    
    # Step 3: Test similarity search
    test_similarity_search(CHROMADB_CONFIG, GEMINI_API_KEYS[0])
    
    print("\n✅ Debug session complete!")

if __name__ == "__main__":
    main_debug()