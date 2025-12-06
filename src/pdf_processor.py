import json
import re
from pathlib import Path
from typing import List, Dict
import pdfplumber
from tqdm import tqdm


class PDFProcessor:
    """Extract and chunk text from PDF with metadata."""
    
    def __init__(self, pdf_path: str, chunk_size: int = 800, chunk_overlap: int = 100):
        self.pdf_path = Path(pdf_path)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
    def extract_text_from_pdf(self) -> List[Dict[str, any]]:
        """Extract text from PDF page by page."""
        pages_text = []
        
        print(f" Extracting text from {self.pdf_path.name}...")
        
        with pdfplumber.open(self.pdf_path) as pdf:
            for page_num, page in enumerate(tqdm(pdf.pages, desc="Extracting pages")):
                text = page.extract_text()
                
                if text and text.strip():
                    pages_text.append({
                        'page_number': page_num + 1,
                        'text': text.strip()
                    })
        
        print(f" Extracted {len(pages_text)} pages")
        return pages_text
    
    def clean_text(self, text: str) -> str:
        """Clean extracted text."""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove page numbers and headers/footers (common patterns)
        text = re.sub(r'\n\d+\n', '\n', text)
        
        # Remove special characters but keep punctuation
        text = re.sub(r'[^\w\s.,!?;:()\-\'\"$%]', '', text)
        
        return text.strip()
    
    def create_chunks(self, pages_text: List[Dict[str, any]]) -> List[Dict[str, any]]:
        """Create overlapping chunks from pages."""
        chunks = []
        chunk_id = 0
        
        print(f"  Creating chunks (size={self.chunk_size}, overlap={self.chunk_overlap})...")
        
        for page_data in tqdm(pages_text, desc="Chunking pages"):
            page_num = page_data['page_number']
            text = self.clean_text(page_data['text'])
            
            # Split by sentences to avoid breaking mid-sentence
            sentences = re.split(r'(?<=[.!?])\s+', text)
            
            current_chunk = ""
            current_length = 0
            
            for sentence in sentences:
                sentence_length = len(sentence.split())
                
                # If adding this sentence exceeds chunk_size, save current chunk
                if current_length + sentence_length > self.chunk_size and current_chunk:
                    chunks.append({
                        'chunk_id': chunk_id,
                        'text': current_chunk.strip(),
                        'page_number': page_num,
                        'word_count': current_length
                    })
                    chunk_id += 1
                    
                    # Start new chunk with overlap
                    overlap_words = current_chunk.split()[-self.chunk_overlap:]
                    current_chunk = ' '.join(overlap_words) + ' ' + sentence
                    current_length = len(overlap_words) + sentence_length
                else:
                    current_chunk += ' ' + sentence
                    current_length += sentence_length
            
            # Add remaining text as final chunk for this page
            if current_chunk.strip():
                chunks.append({
                    'chunk_id': chunk_id,
                    'text': current_chunk.strip(),
                    'page_number': page_num,
                    'word_count': current_length
                })
                chunk_id += 1
        
        print(f" Created {len(chunks)} chunks")
        return chunks
    
    def process_and_save(self, output_path: str = "processed/chunks.json"):
        """Full pipeline: extract, chunk, and save."""
        # Extract text
        pages_text = self.extract_text_from_pdf()
        
        # Create chunks
        chunks = self.create_chunks(pages_text)
        
        # Save to JSON
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(chunks, f, indent=2, ensure_ascii=False)
        
        print(f" Saved {len(chunks)} chunks to {output_path}")
        
        # Print statistics
        total_words = sum(chunk['word_count'] for chunk in chunks)
        avg_words = total_words / len(chunks) if chunks else 0
        
        print(f"\n Statistics:")
        print(f"   - Total chunks: {len(chunks)}")
        print(f"   - Total words: {total_words:,}")
        print(f"   - Avg words/chunk: {avg_words:.0f}")
        print(f"   - Pages covered: {len(set(c['page_number'] for c in chunks))}")
        
        return chunks


def main():
    """Run PDF processing."""
    processor = PDFProcessor(
        pdf_path="data/book.pdf",
        chunk_size=800,
        chunk_overlap=100
    )
    
    chunks = processor.process_and_save()
    
    # Show sample chunks
    print(f"\n Sample chunks:")
    for i, chunk in enumerate(chunks[:3]):
        print(f"\n--- Chunk {i} (Page {chunk['page_number']}) ---")
        print(chunk['text'][:200] + "...")


if __name__ == "__main__":
    main()