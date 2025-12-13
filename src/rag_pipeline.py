import json
import os
from pathlib import Path
from typing import List, Dict, Optional
import google.generativeai as genai
from dotenv import load_dotenv
from src.embeddings import EmbeddingGenerator, InMemoryRetriever
import re


class EnhancedRAGPipeline:
    """Enhanced RAG with better retrieval and context assembly."""
    
    def __init__(
        self,
        chunks_path: str = "processed/chunks.json",
        embeddings_path: str = "processed/embeddings.npy",
        api_key: Optional[str] = None
    ):
        """Initialize enhanced RAG pipeline."""
        
        load_dotenv()
        
        api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found")
        
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.5-flash')
        
        print(" Gemini model configured")
        
        with open(chunks_path, 'r', encoding='utf-8') as f:
            self.chunks = json.load(f)
        print(f" Loaded {len(self.chunks)} chunks")
        
        self.embedding_generator = EmbeddingGenerator()
        embeddings = self.embedding_generator.load_embeddings(embeddings_path)
        self.retriever = InMemoryRetriever(self.chunks, embeddings)
        
        print(" RAG pipeline ready")
    
    def _normalize_term(self, term: str) -> str:
        """Normalize term by removing non-alphanumeric characters."""
        return re.sub(r'[^a-z0-9]', '', term.lower())
    
    def _extract_key_terms(self, query: str) -> List[str]:
        """Extract key terms from query for better matching."""
        # Remove common words
        stop_words = {'what', 'is', 'the', 'how', 'does', 'can', 'you', 
                     'explain', 'describe', 'tell', 'me', 'about', 'a', 'an'}
        
        words = re.findall(r'\b\w+\b', query.lower())
        key_terms = [w for w in words if w not in stop_words and len(w) > 3]
        
        return key_terms
    
    def _rerank_by_relevance(
        self, 
        chunks: List[Dict[str, any]], 
        query: str
    ) -> List[Dict[str, any]]:
        """Re-rank chunks based on multiple relevance signals."""
        key_terms = self._extract_key_terms(query)
        
        for chunk in chunks:
            text_lower = chunk['text'].lower()
            score = chunk.get('similarity_score', 0)
            
            # CHANGE 1: Increased definition boost from 0.1 to 0.25
            # Boost for definition chunks - prioritize heavily for "what is" queries
            if chunk.get('is_definition', False):
                score += 0.25
            
            # CHANGE 2: Normalized term matching to handle variations
            # Boost for term matches with normalization
            norm_text = self._normalize_term(text_lower)
            norm_terms = [self._normalize_term(t) for t in key_terms]
            term_matches = sum(1 for term in norm_terms if term in norm_text)
            # Cap at 4 to prevent repetition gaming
            score += 0.06 * min(term_matches, 4)
            
            # CHANGE 3: More robust phrase matching
            # Boost for exact phrase match (normalized)
            query_clean = re.sub(r'what is (the )?|how does (the )?|explain (the )?', '', query.lower())
            core_phrase = self._normalize_term(query_clean.strip())
            if core_phrase and core_phrase in norm_text:
                score += 0.2
            
            # CHANGE 4: Reduced coherence weight from 0.05 to 0.03
            # Boost for higher coherence (coherence != correctness for factual queries)
            coherence = chunk.get('coherence_score', 0)
            score += 0.03 * coherence
            
            # CHANGE 5: REMOVED length boost entirely
            # (Previously boosted chunks > 600 words by 0.03)
            # For technical definitions, concise chunks are often better
            
            chunk['reranked_score'] = score
        
        # Sort by reranked score
        chunks.sort(key=lambda x: x['reranked_score'], reverse=True)
        
        return chunks
    
    def _find_related_chunks(
        self, 
        selected_chunks: List[Dict[str, any]], 
        all_chunks: List[Dict[str, any]],
        max_related: int = 2
    ) -> List[Dict[str, any]]:
        """Find chunks adjacent to selected ones for better context."""
        related = []
        selected_ids = {c['chunk_id'] for c in selected_chunks}
        
        for chunk in selected_chunks[:3]:  # Only from top 3
            chunk_id = chunk['chunk_id']
            
            # Find adjacent chunks (before and after)
            for other in all_chunks:
                if other['chunk_id'] in selected_ids:
                    continue
                
                # Check if adjacent (within 1-2 chunks)
                id_diff = abs(other['chunk_id'] - chunk_id)
                if id_diff in [1, 2]:
                    # Same chapter
                    if other['chapter'] == chunk['chapter']:
                        other['adjacency_score'] = 0.3 - (0.1 * id_diff)
                        related.append(other)
        
        # Remove duplicates and sort
        seen = set()
        unique_related = []
        for r in related:
            if r['chunk_id'] not in seen:
                seen.add(r['chunk_id'])
                unique_related.append(r)
        
        unique_related.sort(key=lambda x: x.get('adjacency_score', 0), reverse=True)
        
        return unique_related[:max_related]
    
    def _merge_overlapping_context(
        self, 
        chunks: List[Dict[str, any]]
    ) -> List[Dict[str, any]]:
        """Merge chunks that overlap significantly."""
        if len(chunks) <= 1:
            return chunks
        
        # Sort by chunk_id
        sorted_chunks = sorted(chunks, key=lambda x: x['chunk_id'])
        merged = []
        
        i = 0
        while i < len(sorted_chunks):
            current = sorted_chunks[i].copy()
            j = i + 1
            
            # Try to merge with next chunks
            while j < len(sorted_chunks):
                next_chunk = sorted_chunks[j]
                
                # Check if should merge (adjacent and same chapter)
                if (next_chunk['chunk_id'] - current['chunk_id'] <= 1 and
                    next_chunk['chapter'] == current['chapter']):
                    
                    # Merge texts (avoid duplication)
                    current_words = set(current['text'].split())
                    next_words = next_chunk['text'].split()
                    
                    # Calculate overlap
                    overlap = len([w for w in next_words if w in current_words])
                    overlap_ratio = overlap / len(next_words) if next_words else 0
                    
                    if overlap_ratio < 0.7:  # Not too much overlap
                        # Append non-overlapping part
                        current['text'] = current['text'] + " " + next_chunk['text']
                        current['word_count'] += next_chunk['word_count']
                        current['page_end'] = next_chunk['page_end']
                        j += 1
                    else:
                        break
                else:
                    break
            
            merged.append(current)
            i = j if j > i else i + 1
        
        return merged
    
    def create_context(
        self, 
        retrieved_chunks: List[Dict[str, any]], 
        query: str
    ) -> str:
        """Enhanced context creation with better organization."""
        
        # Re-rank chunks
        reranked = self._rerank_by_relevance(retrieved_chunks, query)
        
        # Select top chunks
        top_chunks = reranked[:7]
        
        # Find related adjacent chunks
        related = self._find_related_chunks(top_chunks, self.chunks, max_related=2)
        
        # Combine and deduplicate
        all_context_chunks = top_chunks + related
        seen_ids = set()
        unique_chunks = []
        for c in all_context_chunks:
            if c['chunk_id'] not in seen_ids:
                seen_ids.add(c['chunk_id'])
                unique_chunks.append(c)
        
        # CHANGE 6: Rerank again AFTER adding adjacent chunks
        # This ensures adjacent chunks are properly scored for relevance
        unique_chunks = self._rerank_by_relevance(unique_chunks, query)
        
        # Limit total chunks
        final_chunks = unique_chunks[:8]
        
        # Try to merge adjacent chunks for better continuity
        final_chunks = self._merge_overlapping_context(final_chunks)
        
        # Format context
        context_parts = []
        for i, chunk in enumerate(final_chunks, 1):
            # Add metadata about chunk type
            chunk_type = " [DEFINITION]" if chunk.get('is_definition', False) else ""
            header = f"[Source {i} - Page {chunk['page_number']}{chunk_type}]"
            
            context_parts.append(f"{header}\n{chunk['text']}\n")
        
        return "\n".join(context_parts)
    
    def create_prompt(self, query: str, context: str) -> str:
        """Create enhanced prompt."""
        prompt = f"""You are an expert assistant for "Advances in Financial Machine Learning" by Marcos Lopez de Prado.

**CRITICAL INSTRUCTIONS:**

1. **Answer STRICTLY from the provided context** - Do not use external knowledge
2. **Be COMPREHENSIVE** - If a concept is explained across multiple sources, synthesize them
3. **Cite ALL sources** - Use [Source X] for every claim
4. **Identify when info is INCOMPLETE** - If definition is partial, say "Based on the available context..."
5. **Prioritize [DEFINITION] tagged sources** - These contain formal explanations

**Context from the book:**

{context}

**Question:** {query}

**Instructions for your answer:**
- Start with a clear, direct definition if available
- For definition queries, prioritize clarity over comprehensiveness. Start with a 2-3 sentence core definition, then add key details only.
- Include ALL relevant details from the sources (components, methods, purpose, etc.)
- Use bullet points for multi-part explanations
- Cite each piece of information with [Source X]
- If information spans multiple sources, synthesize them into a coherent explanation
- End with page references for further reading

**Answer:**"""
        
        return prompt
    
    def query(
        self,
        question: str,
        top_k: int = 12,  # Increased for better coverage
        return_sources: bool = True
    ) -> Dict[str, any]:
        """Enhanced query with better retrieval."""
        
        # Embed query
        query_embedding = self.embedding_generator.embed_query(question)
        
        # Retrieve MORE chunks initially
        retrieved_chunks = self.retriever.search(query_embedding, top_k=top_k)
        
        # Enhanced context creation
        context = self.create_context(retrieved_chunks, question)
        
        # Create prompt
        prompt = self.create_prompt(question, context)
        
        # Generate answer
        try:
            response = self.model.generate_content(prompt)
            answer = response.text
        except Exception as e:
            answer = f"Error generating answer: {str(e)}"
        
        # Prepare result
        result = {
            "question": question,
            "answer": answer,
        }
        
        if return_sources:
            # Include top sources with relevance info
            result["sources"] = [
                {
                    "page": chunk["page_number"],
                    "chapter": chunk.get("chapter", "Unknown"),
                    "text_preview": chunk["text"][:250] + "...",
                    "similarity": chunk["similarity_score"],
                    "word_count": chunk["word_count"],
                    "is_definition": chunk.get("is_definition", False)
                }
                for chunk in retrieved_chunks[:top_k]
            ]
        
        return result


def main():
    """Test enhanced RAG pipeline."""
    
    print("\n" + "="*80)
    print(" ENHANCED RAG PIPELINE TEST")
    print("="*80)
    
    rag = EnhancedRAGPipeline()
    
    test_questions = [
        "What is the triple barrier method?",
        "How does the triple barrier method work?",
        "Explain the components of triple barrier labeling",
    ]
    
    for question in test_questions:
        print(f"\n{'='*80}")
        print(f" Question: {question}")
        print("-" * 80)
        
        result = rag.query(question, top_k=12)
        
        print(f"\n Answer:\n{result['answer']}")
        
        print(f"\n Top Sources:")
        for i, source in enumerate(result['sources'], 1):
            def_marker = " [DEFINITION]" if source['is_definition'] else ""
            print(f"\n  {i}. Page {source['page']}{def_marker}")
            print(f"     Chapter: {source['chapter']}")
            print(f"     Similarity: {source['similarity']:.3f} | Words: {source['word_count']}")
            print(f"     Preview: {source['text_preview']}")
        
        print("\n" + "="*80)


if __name__ == "__main__":
    main()