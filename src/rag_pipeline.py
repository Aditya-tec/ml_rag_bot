import json
import os
from pathlib import Path
from typing import List, Dict, Optional
import google.generativeai as genai
from dotenv import load_dotenv
from src.embeddings import EmbeddingGenerator, InMemoryRetriever


class RAGPipeline:
    """RAG pipeline combining retrieval and Gemini LLM."""
    
    def __init__(
        self,
        chunks_path: str = "processed/chunks.json",
        embeddings_path: str = "processed/embeddings.npy",
        api_key: Optional[str] = None
    ):
        """Initialize RAG pipeline."""
        
        # Load environment variables
        load_dotenv()
        
        # Configure Gemini
        api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment or .env file")
        
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.5-flash')
        
        print("Gemini model configured")
        
        # Load chunks
        with open(chunks_path, 'r', encoding='utf-8') as f:
            self.chunks = json.load(f)
        print(f"Loaded {len(self.chunks)} chunks")
        
        # Initialize embedding generator and retriever
        self.embedding_generator = EmbeddingGenerator()
        embeddings = self.embedding_generator.load_embeddings(embeddings_path)
        self.retriever = InMemoryRetriever(self.chunks, embeddings)
        
        print("RAG pipeline ready")
    
    def create_context(self, retrieved_chunks: List[Dict[str, any]]) -> str:
        """Format retrieved chunks into context for LLM."""
        context_parts = []
        
        for i, chunk in enumerate(retrieved_chunks, 1):
            context_parts.append(
                f"[Source {i} - Page {chunk['page_number']}]\n{chunk['text']}\n"
            )
        
        return "\n".join(context_parts)
    
    def create_prompt(self, query: str, context: str) -> str:
        """Create prompt for Gemini."""
        prompt = f"""You are an expert assistant for the book "Advances in Financial Machine Learning" by Marcos Lopez de Prado.

Answer the following question using ONLY the information provided in the context below. Be precise and cite page numbers when referencing specific information.

If the answer cannot be found in the context, say "I cannot find this information in the provided book content."

Context from the book:
{context}

Question: {query}

Answer:"""
        return prompt
    
    def query(
        self,
        question: str,
        top_k: int = 5,
        return_sources: bool = True
    ) -> Dict[str, any]:
        """
        Query the RAG system.
        
        Args:
            question: User's question
            top_k: Number of chunks to retrieve
            return_sources: Whether to return source chunks
            
        Returns:
            Dictionary with answer and optional sources
        """
        
        # Step 1: Embed query
        query_embedding = self.embedding_generator.embed_query(question)
        
        # Step 2: Retrieve relevant chunks
        retrieved_chunks = self.retriever.search(query_embedding, top_k=10)
        
        # Step 3: Create context and prompt
        context = self.create_context(retrieved_chunks)
        prompt = self.create_prompt(question, context)
        
        # Step 4: Generate answer with Gemini
        try:
            response = self.model.generate_content(prompt)
            answer = response.text
        except Exception as e:
            answer = f"Error generating answer: {str(e)}"
        
        # Step 5: Prepare result
        result = {
            "question": question,
            "answer": answer,
        }
        
        if return_sources:
            result["sources"] = [
                {
                    "page": chunk["page_number"],
                    "text": "..." + chunk["text"][-300:],
                    "similarity": chunk["similarity_score"]
                }
                for chunk in retrieved_chunks
            ]
        
        return result


def main():
    """Test the RAG pipeline."""
    
    # Initialize pipeline
    print("\nInitializing RAG pipeline...")
    rag = RAGPipeline()
    
    # Test questions
    test_questions = [
        "What is the triple barrier method?",
        "How does cross-validation work in financial ML?",
        "What is meta-labeling?",
    ]
    
    print("\n" + "="*80)
    print("Testing RAG Pipeline")
    print("="*80)
    
    for question in test_questions:
        print(f"\nQuestion: {question}")
        print("-" * 80)
        
        result = rag.query(question, top_k=3)
        
        print(f"\nAnswer:\n{result['answer']}")
        
        print(f"\nSources:")
        for i, source in enumerate(result['sources'], 1):
            print(f"  {i}. Page {source['page']} (similarity: {source['similarity']:.3f})")
            print(f"     {source['text']}\n")
        
        print("="*80)


if __name__ == "__main__":
    main()