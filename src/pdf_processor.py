import json
import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import pdfplumber
from tqdm import tqdm


class SemanticPDFProcessor:
    """Extract and chunk text from PDF using semantic boundaries (sentences)."""
    
    def __init__(
        self, 
        pdf_path: str, 
        min_chunk_words: int = 400,
        target_chunk_words: int = 600,
        max_chunk_words: int = 800,
        overlap_sentences: int = 2
    ):
        self.pdf_path = Path(pdf_path)
        self.min_chunk_words = min_chunk_words
        self.target_chunk_words = target_chunk_words
        self.max_chunk_words = max_chunk_words
        self.overlap_sentences = overlap_sentences
        
        # Chapter detection patterns
        self.chapter_patterns = [
            r'^CHAPTER\s+\d+',
            r'^PART\s+\d+',
        ]
        
        self.section_patterns = [
            r'^\d+\.\d+\s+[A-Z]',
        ]
        
        self.known_chapters = {
            'Financial Machine Learning as a Distinct Subject',
            'Financial Data Structures',
            'Labeling',
            'Sample Weights',
            'Fractionally Differentiated Features',
            'Ensemble Methods',
            'Cross-Validation in Finance',
            'Feature Importance',
            'Hyper-Parameter Tuning',
            'Bet Sizing',
            'The Dangers of Backtesting',
            'Backtesting through Cross-Validation',
            'Backtesting on Synthetic Data',
            'Backtest Statistics',
            'Understanding Strategy Risk',
            'Machine Learning Asset Allocation',
            'Structural Breaks',
            'Entropy Features',
            'Microstructural Features',
            'Multiprocessing and Vectorization',
            'Brute Force and Quantum Computers',
        }
    
    def extract_text_from_pdf(self) -> List[Dict[str, any]]:
        """Extract text from PDF page by page."""
        pages_data = []
        
        print(f"Extracting text from {self.pdf_path.name}...")
        
        with pdfplumber.open(self.pdf_path) as pdf:
            for page_num, page in enumerate(tqdm(pdf.pages, desc="Extracting pages")):
                text = page.extract_text()
                
                if text and text.strip():
                    pages_data.append({
                        'page_number': page_num + 1,
                        'text': text.strip()
                    })
        
        print(f"Extracted {len(pages_data)} pages")
        return pages_data
    
    def clean_text(self, text: str) -> str:
        """Clean text while preserving sentence structure."""
        # Remove special characters but keep sentence boundaries
        text = re.sub(r'[^\w\s.,!?;:()\-\'\"$%/\n]', '', text)
        
        # Normalize whitespace but keep newlines
        lines = text.split('\n')
        cleaned_lines = [re.sub(r'[ \t]+', ' ', line.strip()) for line in lines]
        text = '\n'.join(cleaned_lines)
        
        # Remove excessive newlines
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        return text.strip()
    
    def split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences intelligently."""
        # Handle common abbreviations to avoid false splits
        text = text.replace('Dr.', 'Dr<dot>')
        text = text.replace('Mr.', 'Mr<dot>')
        text = text.replace('Mrs.', 'Mrs<dot>')
        text = text.replace('Ms.', 'Ms<dot>')
        text = text.replace('Prof.', 'Prof<dot>')
        text = text.replace('vs.', 'vs<dot>')
        text = text.replace('etc.', 'etc<dot>')
        text = text.replace('e.g.', 'e<dot>g<dot>')
        text = text.replace('i.e.', 'i<dot>e<dot>')
        
        # Split on sentence boundaries
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
        
        # Restore abbreviations
        sentences = [s.replace('<dot>', '.') for s in sentences]
        
        # Filter out very short fragments (likely artifacts)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
        
        return sentences
    
    def is_header_or_footer(self, line: str) -> bool:
        """Detect page headers/footers."""
        line = line.strip()
        
        # Just page numbers
        if re.match(r'^\d+$', line):
            return True
        
        # Headers like "CHAPTER 5 TITLE"
        if len(line) < 100 and line.isupper():
            return True
        
        return False
    
    def detect_chapter(self, text: str) -> Optional[str]:
        """Detect chapter heading."""
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        
        if not lines:
            return None
        
        for i in range(min(5, len(lines))):
            line = lines[i]
            
            # Skip headers/footers
            if self.is_header_or_footer(line):
                continue
            
            # Check chapter patterns
            for pattern in self.chapter_patterns:
                if re.match(pattern, line, re.IGNORECASE):
                    if i + 1 < len(lines):
                        title = lines[i + 1]
                        if any(known.lower() in title.lower() for known in self.known_chapters):
                            return f"{line} - {title}"
                    return line
            
            # Check known chapter names
            if any(known.lower() == line.lower() for known in self.known_chapters):
                return line
        
        return None
    
    def detect_section(self, text: str) -> Optional[str]:
        """Detect section heading."""
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        
        for i in range(min(3, len(lines))):
            if i >= len(lines):
                break
            
            line = lines[i]
            
            if self.is_header_or_footer(line):
                continue
            
            if len(line) > 100:
                continue
            
            for pattern in self.section_patterns:
                if re.match(pattern, line):
                    return line
        
        return None
    
    def combine_pages_with_sentences(self, pages_data: List[Dict[str, any]]) -> List[Dict[str, any]]:
        """Combine pages into sentences with metadata."""
        all_sentences = []
        
        current_chapter = "Front Matter"
        current_section = None
        
        print("\nProcessing pages into sentences with metadata...")
        
        for page_data in tqdm(pages_data, desc="Processing pages"):
            page_num = page_data['page_number']
            raw_text = page_data['text']
            
            # Detect chapter/section
            chapter = self.detect_chapter(raw_text)
            if chapter:
                current_chapter = chapter
                current_section = None
            
            section = self.detect_section(raw_text)
            if section and not chapter:
                current_section = section
            
            # Clean and split into sentences
            cleaned_text = self.clean_text(raw_text)
            sentences = self.split_into_sentences(cleaned_text)
            
            # Add metadata to each sentence
            for sentence in sentences:
                all_sentences.append({
                    'text': sentence,
                    'page': page_num,
                    'chapter': current_chapter,
                    'section': current_section,
                    'word_count': len(sentence.split())
                })
        
        print(f"Extracted {len(all_sentences)} sentences")
        return all_sentences
    
    def create_semantic_chunks(self, sentences: List[Dict[str, any]]) -> List[Dict[str, any]]:
        """Create chunks that respect sentence boundaries."""
        chunks = []
        chunk_id = 0
        
        i = 0
        total_sentences = len(sentences)
        
        print(f"\nCreating semantic chunks...")
        print(f"Total sentences: {total_sentences}")
        print(f"Target chunk size: {self.target_chunk_words} words")
        print(f"Range: {self.min_chunk_words}-{self.max_chunk_words} words")
        
        with tqdm(total=total_sentences, desc="Chunking") as pbar:
            while i < total_sentences:
                chunk_sentences = []
                chunk_words = 0
                chunk_pages = set()
                chunk_chapter = sentences[i]['chapter']
                chunk_section = sentences[i]['section']
                
                start_idx = i
                
                # Add sentences until we reach target or max
                while i < total_sentences:
                    sentence = sentences[i]
                    sentence_words = sentence['word_count']
                    
                    # If adding this sentence keeps us under max, add it
                    if chunk_words + sentence_words <= self.max_chunk_words:
                        chunk_sentences.append(sentence['text'])
                        chunk_words += sentence_words
                        chunk_pages.add(sentence['page'])
                        i += 1
                        
                        # If we've reached target, check if we should stop
                        if chunk_words >= self.target_chunk_words:
                            # Stop at next sentence boundary after target
                            break
                    else:
                        # Would exceed max
                        if chunk_words >= self.min_chunk_words:
                            # We have enough, stop here
                            break
                        else:
                            # Not enough yet, add anyway and stop
                            chunk_sentences.append(sentence['text'])
                            chunk_words += sentence_words
                            chunk_pages.add(sentence['page'])
                            i += 1
                            break
                
                # Create chunk if we have content
                if chunk_sentences:
                    chunk_text = ' '.join(chunk_sentences)
                    pages_list = sorted(list(chunk_pages))
                    
                    chunks.append({
                        'chunk_id': chunk_id,
                        'text': chunk_text,
                        'word_count': chunk_words,
                        'sentence_count': len(chunk_sentences),
                        'page_start': pages_list[0],
                        'page_end': pages_list[-1],
                        'page_number': pages_list[0],
                        'chapter': chunk_chapter,
                        'section': chunk_section
                    })
                    chunk_id += 1
                
                # Move back for overlap (go back N sentences)
                overlap_start = max(start_idx, i - self.overlap_sentences)
                i = overlap_start if overlap_start > start_idx else i
                
                # If no progress made, force move forward
                if i == start_idx:
                    i += 1
                
                pbar.update(i - pbar.n)
        
        print(f"\nCreated {len(chunks)} semantic chunks")
        return chunks
    
    def process_and_save(self, output_path: str = "processed/chunks.json"):
        """Full pipeline: extract, chunk semantically, and save."""
        # Extract pages
        pages_data = self.extract_text_from_pdf()
        
        # Convert to sentences with metadata
        sentences = self.combine_pages_with_sentences(pages_data)
        
        # Create semantic chunks
        chunks = self.create_semantic_chunks(sentences)
        
        # Save
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(chunks, f, indent=2, ensure_ascii=False)
        
        print(f"\nSaved {len(chunks)} chunks to {output_path}")
        
        # Statistics
        if chunks:
            word_counts = [c['word_count'] for c in chunks]
            sentence_counts = [c['sentence_count'] for c in chunks]
            
            print(f"\nStatistics:")
            print(f"   - Total chunks: {len(chunks)}")
            print(f"   - Words per chunk:")
            print(f"     * Min: {min(word_counts)}")
            print(f"     * Max: {max(word_counts)}")
            print(f"     * Mean: {sum(word_counts)/len(word_counts):.1f}")
            print(f"     * Median: {sorted(word_counts)[len(word_counts)//2]}")
            print(f"   - Sentences per chunk:")
            print(f"     * Min: {min(sentence_counts)}")
            print(f"     * Max: {max(sentence_counts)}")
            print(f"     * Mean: {sum(sentence_counts)/len(sentence_counts):.1f}")
            
            chapters = set(c['chapter'] for c in chunks if c['chapter'])
            sections = set(c['section'] for c in chunks if c['section'])
            
            print(f"   - Unique chapters: {len(chapters)}")
            print(f"   - Unique sections: {len(sections)}")
            
            # Verify all chunks end at sentence boundaries
            print(f"\n   Sentence Boundary Verification:")
            sentence_endings = sum(1 for c in chunks if c['text'][-1] in '.!?')
            print(f"   - Chunks ending with punctuation: {sentence_endings}/{len(chunks)} ({100*sentence_endings/len(chunks):.1f}%)")
            
            # Chapter distribution
            print(f"\n   Top 10 Chapters by Chunk Count:")
            chapter_counts = {}
            for chunk in chunks:
                ch = chunk['chapter']
                chapter_counts[ch] = chapter_counts.get(ch, 0) + 1
            
            for chapter, count in sorted(chapter_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
                print(f"     {chapter}: {count} chunks")
        
        return chunks


def main():
    """Run semantic PDF processing."""
    processor = SemanticPDFProcessor(
        pdf_path="data/book.pdf",
        min_chunk_words=400,
        target_chunk_words=600,
        max_chunk_words=800,
        overlap_sentences=2
    )
    
    chunks = processor.process_and_save()
    
    if chunks and len(chunks) >= 3:
        print(f"\n\nSample Chunks (showing first 3):")
        for i in range(min(3, len(chunks))):
            chunk = chunks[i]
            print(f"\n{'='*80}")
            print(f"Chunk {chunk['chunk_id']}:")
            print(f"Chapter: {chunk['chapter']}")
            print(f"Section: {chunk.get('section', 'None')}")
            print(f"Pages: {chunk['page_start']}-{chunk['page_end']}")
            print(f"Words: {chunk['word_count']} | Sentences: {chunk['sentence_count']}")
            print(f"Text starts: {chunk['text'][:150]}...")
            print(f"Text ends: ...{chunk['text'][-150:]}")
            print(f"Last character: '{chunk['text'][-1]}'")


if __name__ == "__main__":
    main()