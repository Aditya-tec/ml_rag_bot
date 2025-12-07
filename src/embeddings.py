import json
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from typing import List, Dict


class EmbeddingGenerator:
    """Generate and manage embeddings for text chunks."""
    
    def __init__(self, model_name: str = "all-mpnet-base-v2"):
        """
        Initialize embedding model.
        
       
        """
        print(f" Loading embedding model: {model_name}...")
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
        print(f" Model loaded: {self.model.get_sentence_embedding_dimension()} dimensions")
    
    def generate_embeddings(self, chunks: List[Dict[str, any]], batch_size: int = 32) -> np.ndarray:
        """Generate embeddings for all chunks."""
        print(f"\n Generating embeddings for {len(chunks)} chunks...")
        
        texts = [chunk['text'] for chunk in chunks]
        
        # Generate embeddings in batches for efficiency
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True  # Normalize for cosine similarity
        )
        
        print(f" Generated embeddings: shape {embeddings.shape}")
        return embeddings
    
    def save_embeddings(self, embeddings: np.ndarray, output_path: str = "processed/embeddings.npy"):
        """Save embeddings to file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        np.save(output_path, embeddings)
        
        size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f" Saved embeddings to {output_path} ({size_mb:.2f} MB)")
    
    def load_embeddings(self, embeddings_path: str = "processed/embeddings.npy") -> np.ndarray:
        """Load embeddings from file."""
        embeddings_path = Path(embeddings_path)
        
        if not embeddings_path.exists():
            raise FileNotFoundError(f"Embeddings file not found: {embeddings_path}")
        
        embeddings = np.load(embeddings_path)
        print(f" Loaded embeddings: shape {embeddings.shape}")
        return embeddings
    
    def embed_query(self, query: str) -> np.ndarray:
        """Generate embedding for a single query."""
        embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        return embedding[0]


class InMemoryRetriever:
    """In-memory similarity search using NumPy (no database)."""
    
    def __init__(self, chunks: List[Dict[str, any]], embeddings: np.ndarray):
        """Initialize with chunks and their embeddings."""
        self.chunks = chunks
        self.embeddings = embeddings
        
        if len(chunks) != len(embeddings):
            raise ValueError(f"Chunks ({len(chunks)}) and embeddings ({len(embeddings)}) length mismatch!")
        
        print(f" Retriever ready with {len(chunks)} chunks in memory")
    
    def cosine_similarity(self, query_embedding: np.ndarray) -> np.ndarray:
        """Calculate cosine similarity between query and all chunks."""
        # Since embeddings are normalized, dot product = cosine similarity
        similarities = np.dot(self.embeddings, query_embedding)
        return similarities
    
    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Dict[str, any]]:
        """Search for top-k most similar chunks."""
        # Calculate similarities
        similarities = self.cosine_similarity(query_embedding)
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        # Return chunks with similarity scores
        results = []
        for idx in top_indices:
            result = self.chunks[idx].copy()
            result['similarity_score'] = float(similarities[idx])
            results.append(result)
        
        return results


def main():
    """Generate embeddings for all chunks."""
    # Load chunks
    chunks_path = Path("processed/chunks.json")
    if not chunks_path.exists():
        print(" Error: chunks.json not found. Run pdf_processor.py first!")
        return
    
    with open(chunks_path, 'r', encoding='utf-8') as f:
        chunks = json.load(f)
    
    print(f" Loaded {len(chunks)} chunks from {chunks_path}")
    
    # Initialize embedding generator
    generator = EmbeddingGenerator(model_name="all-mpnet-base-v2")
    
    # Generate embeddings
    embeddings = generator.generate_embeddings(chunks, batch_size=32)
    
    # Save embeddings
    generator.save_embeddings(embeddings)
    
    # Test retrieval
    print("\n Testing retrieval...")
    retriever = InMemoryRetriever(chunks, embeddings)
    
    test_query = "What is the triple barrier method?"
    print(f"\nQuery: '{test_query}'")
    
    query_embedding = generator.embed_query(test_query)
    results = retriever.search(query_embedding, top_k=3)
    
    print(f"\n Top 3 results:")
    for i, result in enumerate(results, 1):
        print(f"\n{i}. Similarity: {result['similarity_score']:.4f} | Page: {result['page_number']}")
        print(f"   Text: {result['text'][:150]}...")
    
    print("\n Embeddings generated and tested successfully!")


if __name__ == "__main__":
    main()