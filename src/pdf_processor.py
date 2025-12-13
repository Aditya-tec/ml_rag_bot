import json
import re
from pathlib import Path
from typing import List, Dict, Optional
import pdfplumber
import pypdf
from tqdm import tqdm


class EnhancedSemanticPDFProcessor:
    """Enhanced processor with better definition and context handling."""
    
    def __init__(
        self, 
        pdf_path: str, 
        min_chunk_words: int = 100,
        target_chunk_words: int = 300,
        max_chunk_words: int = 500,
        overlap_sentences: int = 5
    ):
        self.pdf_path = Path(pdf_path)
        self.min_chunk_words = min_chunk_words
        self.target_chunk_words = target_chunk_words
        self.max_chunk_words = max_chunk_words
        self.overlap_sentences = overlap_sentences
        
        # Enhanced chapter detection patterns (Regex)
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
        
        # Definition and concept indicators
        self.definition_indicators = [
            r'\bis defined as\b',
            r'\brefers to\b',
            r'\bconsists of\b',
            r'\bcomprises\b',
            r'\bmethod involves\b',
            r'\btechnique involves\b',
            r'\bapproach involves\b',
            r'\bare as follows\b',
            r'\bbarriers are\b',
            r'\bthree barriers\b',
            r'\bsteps are\b',
            r'\bfollowing steps\b',
            r'\bcomponents are\b',
            r'\bkey aspects\b',
            r'\bmain features\b',
            r'\:\s*$',
        ]
        
        # Section/subsection headers that often contain definitions
        self.concept_headers = [
            r'^\d+\.\d+\.\d+\s+',
            r'^Definition',
            r'^Algorithm',
            r'^Method',
            r'^Approach',
        ]

    def _has_spacing_issues(self, text: str) -> bool:
        """Detect if text has word spacing problems."""
        if not text or not text.strip():
            return False
        
        words = text.split()
        if not words:
            return False
        
        long_words = sum(1 for w in words if len(w) > 30)
        if long_words / len(words) > 0.05:
            return True
        
        concat_patterns = [
            r'[a-z]{10,}[A-Z][a-z]{5,}',
            r'[a-z]{15,}',
        ]
        
        for pattern in concat_patterns:
            if len(re.findall(pattern, text)) > 3:
                return True
        
        return False
    
    def _extract_with_pypdf_fallback(self, page_num: int) -> Optional[str]:
        """Fallback extraction using pypdf if pdfplumber fails."""
        try:
            with open(self.pdf_path, 'rb') as f:
                pdf_reader = pypdf.PdfReader(f)
                if page_num < len(pdf_reader.pages):
                    page = pdf_reader.pages[page_num]
                    text = page.extract_text()
                    return text if text and text.strip() else None
        except Exception as e:
            print(f"  pypdf fallback failed for page {page_num + 1}: {e}")
        return None
    
    def is_toc_page(self, text: str) -> bool:
        """Detect if a page is a Table of Contents or Index."""
        lines = text.split('\n')
        if not lines:
            return False
            
        toc_lines = 0
        for line in lines:
            if re.search(r'\.{3,}\s*\d+$', line.strip()) or (re.search(r'\s+\d+$', line.strip()) and len(line) < 80):
                toc_lines += 1
        
        if len(lines) > 5 and (toc_lines / len(lines)) > 0.3:
            return True
            
        first_line = lines[0].strip().upper()
        if "INDEX" in first_line or "CONTENTS" in first_line:
            return True
            
        return False
    
    def is_header_or_footer(self, line: str) -> bool:
        """Detect headers/footers."""
        line = line.strip()
        # Page numbers (e.g., "45")
        if re.match(r'^\d+$', line):
            return True
        # Running headers (usually short, uppercase, e.g., "CHAPTER 3")
        if len(line) < 100 and line.isupper():
            return True
        return False

    def extract_text_from_pdf(self) -> List[Dict[str, any]]:
        """Extract text from PDF with multiple strategies."""
        pages_data = []
        
        print(f"Extracting text from {self.pdf_path.name}...")
        
        with pdfplumber.open(self.pdf_path) as pdf:
            for page_num, page in enumerate(tqdm(pdf.pages, desc="Extracting pages")):
                text = None
                extraction_method = "pdfplumber_default"
                
                try:
                    bbox = page.bbox
                    crop_box = (
                        bbox[0],                      # Left
                        bbox[1] + (page.height * 0.08), # Top + 8%
                        bbox[2],                      # Right
                        bbox[3] - (page.height * 0.10)  # Bottom - 10%
                    )
                    
                    cropped_page = page.crop(crop_box)
                    
                    text = cropped_page.extract_text(
                        x_tolerance=3,
                        y_tolerance=3,
                        layout=True
                    )
                    
                except Exception:
                    pass
                
                if text and self._has_spacing_issues(text):
                    try:
                        text_alt = page.extract_text(
                            x_tolerance=4,
                            y_tolerance=3,
                            layout=False
                        )
                        if text_alt and not self._has_spacing_issues(text_alt):
                            text = text_alt
                            extraction_method = "pdfplumber_alternative"
                    except Exception:
                        pass
                
                if not text or self._has_spacing_issues(text):
                    text_fallback = self._extract_with_pypdf_fallback(page_num)
                    if text_fallback and not self._has_spacing_issues(text_fallback):
                        text = text_fallback
                        extraction_method = "pypdf_fallback"
                
                if not text or not text.strip():
                    try:
                        text = page.extract_text()
                        extraction_method = "pdfplumber_basic"
                    except Exception:
                        continue
                
                if text and text.strip():
                    if self.is_toc_page(text):
                        continue
                    
                    if self._has_spacing_issues(text):
                        print(f"Page {page_num + 1} still has spacing issues")
                    
                    pages_data.append({
                        'page_number': page_num + 1,
                        'text': text.strip(),
                        'extraction_method': extraction_method
                    })
        
        print(f"Extracted {len(pages_data)} pages")
        
        methods = {}
        for page in pages_data:
            method = page.get('extraction_method', 'unknown')
            methods[method] = methods.get(method, 0) + 1
        print(f"Extraction methods: {methods}")
        
        return pages_data
    
    def clean_text(self, text: str) -> str:
        """Enhanced text cleaning. Preserves math symbols."""
        def smart_camel_split(match):
            word = match.group(0)
            if 3 < len(word) < 25 and sum(1 for c in word if c.isupper()) <= 3:
                return re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', word)
            return word
        
        text = re.sub(r'\b[a-z]+[A-Z][a-zA-Z]+\b', smart_camel_split, text)
        text = re.sub(r'(?<=[a-zA-Z])(?=\d)', ' ', text)
        text = re.sub(r'(?<=\d)(?=[a-zA-Z])', ' ', text)
        
        text = re.sub(r' {2,}', ' ', text)
        text = re.sub(r'\t+', ' ', text)
        
        lines = text.split('\n')
        cleaned_lines = [line.strip() for line in lines if line.strip()]
        text = '\n'.join(cleaned_lines)
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        return text.strip()
    
    def split_into_sentences(self, text: str) -> List[str]:
        """Enhanced sentence splitting."""
        abbreviations = {
            'Dr.': 'Dr<dot>', 'Mr.': 'Mr<dot>', 'Mrs.': 'Mrs<dot>',
            'Ms.': 'Ms<dot>', 'Prof.': 'Prof<dot>', 'vs.': 'vs<dot>',
            'etc.': 'etc<dot>', 'e.g.': 'e<dot>g<dot>', 'i.e.': 'i<dot>e<dot>',
            'vol.': 'vol<dot>', 'fig.': 'fig<dot>', 'eq.': 'eq<dot>',
            'no.': 'no<dot>', 'al.': 'al<dot>', 'Inc.': 'Inc<dot>',
        }
        
        for abbr, placeholder in abbreviations.items():
            text = text.replace(abbr, placeholder)
        
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
        sentences = [s.replace('<dot>', '.') for s in sentences]
        sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
        
        return sentences
    
    def detect_chapter_metadata(self) -> Dict[int, str]:
        """Uses PDF outline/bookmarks to map page numbers to chapter titles."""
        chapter_map = {}
        try:
            with open(self.pdf_path, 'rb') as f:
                pdf_reader = pypdf.PdfReader(f)
                outline = pdf_reader.outline
                
                def _process_outline(items):
                    for item in items:
                        if isinstance(item, list):
                            _process_outline(item)
                        else:
                            try:
                                page_num = pdf_reader.get_destination_page_number(item) + 1
                                title = item.title
                                chapter_map[page_num] = title
                            except Exception:
                                pass
                
                _process_outline(outline)
        except Exception as e:
            print(f"Warning: Could not read PDF metadata outline: {e}")
            
        return chapter_map

    def detect_chapter(self, text: str) -> Optional[str]:
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        if not lines: return None
        for i in range(min(5, len(lines))):
            line = lines[i]
            if self.is_header_or_footer(line): continue
            for pattern in self.chapter_patterns:
                if re.match(pattern, line, re.IGNORECASE):
                    if i + 1 < len(lines):
                        title = lines[i + 1]
                        if any(known.lower() in title.lower() for known in self.known_chapters):
                            return f"{line} - {title}"
                    return line
            if any(known.lower() == line.lower() for known in self.known_chapters):
                return line
        return None
    
    def detect_section(self, text: str) -> Optional[str]:
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        for i in range(min(3, len(lines))):
            if i >= len(lines): break
            line = lines[i]
            if self.is_header_or_footer(line): continue
            if len(line) > 100: continue
            for pattern in self.section_patterns:
                if re.match(pattern, line):
                    return line
        return None
    
    def _is_definition_context(self, sentences: List[Dict[str, any]], idx: int, lookahead: int = 5) -> bool:
        """Detect if we're in or near a definition/explanation."""
        if idx >= len(sentences):
            return False
        
        end_idx = min(idx + lookahead, len(sentences))
        
        for i in range(idx, end_idx):
            text = sentences[i]['text'].lower()
            
            for pattern in self.definition_indicators:
                if re.search(pattern, text):
                    return True
            
            if re.search(r'^\s*[\d\-\•\*]\s*\.', text):
                return True
            
            for pattern in self.concept_headers:
                if re.search(pattern, text):
                    return True
        
        return False
    
    def _calculate_coherence_score(self, sentences: List[str]) -> float:
        """Calculate how coherent a group of sentences is."""
        if len(sentences) < 2:
            return 1.0
        
        score = 0.0
        transition_words = ['however', 'therefore', 'moreover', 'furthermore', 
                          'additionally', 'consequently', 'similarly', 'thus']
        
        for i in range(1, len(sentences)):
            text = sentences[i].lower()
            if any(word in text for word in transition_words):
                score += 0.3
            if re.search(r'\b(this|these|that|those|it|they)\b', text[:50]):
                score += 0.2
            if re.search(r'^(chapter|section|\d+\.)', text):
                score -= 0.5
        
        return max(0, min(1, score / len(sentences)))
    
    def combine_pages_with_sentences(self, pages_data: List[Dict[str, any]]) -> List[Dict[str, any]]:
        """Enhanced sentence extraction with metadata using PDF outline."""
        all_sentences = []
        
        chapter_map = self.detect_chapter_metadata()
        current_chapter = "Front Matter"
        current_section = None
        
        print("\nProcessing pages into sentences...")
        
        for page_data in tqdm(pages_data, desc="Processing pages"):
            page_num = page_data['page_number']
            raw_text = page_data['text']
            
            if page_num in chapter_map:
                current_chapter = chapter_map[page_num]
                current_section = None
            elif not chapter_map:
                chapter = self.detect_chapter(raw_text)
                if chapter:
                    current_chapter = chapter
                    current_section = None
            
            section = self.detect_section(raw_text)
            if section:
                current_section = section
            
            cleaned_text = self.clean_text(raw_text)
            sentences = self.split_into_sentences(cleaned_text)
            
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
        """Enhanced chunking that respects definitions and explanations."""
        chunks = []
        chunk_id = 0
        
        i = 0
        total_sentences = len(sentences)
        
        print(f"\nCreating enhanced semantic chunks...")
        print(f"Total sentences: {total_sentences}")
        print(f"Target: {self.target_chunk_words} words (range: {self.min_chunk_words}-{self.max_chunk_words})")
        
        with tqdm(total=total_sentences, desc="Chunking") as pbar:
            while i < total_sentences:
                chunk_sentences = []
                chunk_words = 0
                chunk_pages = set()
                chunk_chapter = sentences[i]['chapter']
                chunk_section = sentences[i]['section']
                
                start_idx = i
                
                in_definition = self._is_definition_context(sentences, i, lookahead=5)
                
                effective_max = self.max_chunk_words
                effective_target = self.target_chunk_words
                
                sentence_texts = []
                while i < total_sentences:
                    sentence = sentences[i]
                    sentence_words = sentence['word_count']
                    
                    potential_words = chunk_words + sentence_words
                    
                    if potential_words <= effective_max:
                        chunk_sentences.append(sentence['text'])
                        sentence_texts.append(sentence['text'])
                        chunk_words += sentence_words
                        chunk_pages.add(sentence['page'])
                        
                        if sentence['chapter'] != chunk_chapter and chunk_words > self.min_chunk_words:
                             break

                        i += 1
                        
                        if chunk_words >= effective_target:
                            if not in_definition:
                                break
                            else:
                                if i < total_sentences:
                                    next_is_definition = self._is_definition_context(sentences, i, lookahead=2)
                                    if not next_is_definition:
                                        break
                    else:
                        if chunk_words >= self.min_chunk_words:
                            break
                        else:
                            chunk_sentences.append(sentence['text'])
                            sentence_texts.append(sentence['text'])
                            chunk_words += sentence_words
                            chunk_pages.add(sentence['page'])
                            i += 1
                            break
                
                if chunk_sentences:
                    chunk_text = ' '.join(chunk_sentences)
                    pages_list = sorted(list(chunk_pages))
                    
                    coherence = self._calculate_coherence_score(sentence_texts)
                    
                    chunks.append({
                        'chunk_id': chunk_id,
                        'text': chunk_text,
                        'word_count': chunk_words,
                        'sentence_count': len(chunk_sentences),
                        'page_start': pages_list[0],
                        'page_end': pages_list[-1],
                        'page_number': pages_list[0],
                        'chapter': chunk_chapter,
                        'section': chunk_section,
                        'is_definition': in_definition,
                        'coherence_score': coherence
                    })
                    chunk_id += 1
                
                overlap_count = self.overlap_sentences if not in_definition else self.overlap_sentences + 1
                overlap_start = max(start_idx, i - overlap_count)
                i = overlap_start if overlap_start > start_idx else i
                if i == start_idx:
                    i += 1
                
                pbar.update(i - pbar.n)
        
        print(f"\nCreated {len(chunks)} chunks")
        
        def_chunks = sum(1 for c in chunks if c.get('is_definition', False))
        print(f"  - Definition/explanation chunks: {def_chunks}")
        
        return chunks
    
    def validate_chunks(self, chunks: List[Dict[str, any]]) -> Dict[str, int]:
        """Validate chunk quality."""
        issues = {
            'spacing_problems': 0,
            'incomplete_sentences': 0,
            'too_short': 0,
            'too_long': 0
        }
        
        for chunk in chunks:
            text = chunk['text']
            
            if self._has_spacing_issues(text):
                issues['spacing_problems'] += 1
            
            if text and text[-1] not in '.!?':
                issues['incomplete_sentences'] += 1
            
            if chunk['word_count'] < self.min_chunk_words * 0.8:
                issues['too_short'] += 1
            
            if chunk['word_count'] > self.max_chunk_words * 1.1:
                issues['too_long'] += 1
        
        return issues
    
    def process_and_save(self, output_path: str = "processed/chunks.json"):
        """Full pipeline with enhanced chunking."""
        pages_data = self.extract_text_from_pdf()
        
        if not pages_data:
            raise ValueError("No pages extracted from PDF")
        
        sentences = self.combine_pages_with_sentences(pages_data)
        
        if not sentences:
            raise ValueError("No sentences extracted")
        
        chunks = self.create_semantic_chunks(sentences)
        
        if not chunks:
            raise ValueError("No chunks created")
        
        issues = self.validate_chunks(chunks)
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(chunks, f, indent=2, ensure_ascii=False)
        
        print(f"\nSaved {len(chunks)} chunks to {output_path}")
        
        if chunks:
            word_counts = [c['word_count'] for c in chunks]
            sentence_counts = [c['sentence_count'] for c in chunks]
            
            print(f"\n Statistics:")
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
            
            def_chunks = [c for c in chunks if c.get('is_definition', False)]
            if def_chunks:
                def_words = [c['word_count'] for c in def_chunks]
                print(f"\n    Definition chunks: {len(def_chunks)}")
                print(f"      * Avg words: {sum(def_words)/len(def_words):.1f}")
            
            print(f"\n Quality Report:")
            for issue, count in issues.items():
                percentage = 100 * count / len(chunks)
                status = "okk" if percentage < 5 else "!!" if percentage < 15 else "noo"
                print(f"   {status} {issue}: {count}/{len(chunks)} ({percentage:.1f}%)")
            
            sentence_endings = sum(1 for c in chunks if c['text'] and c['text'][-1] in '.!?')
            print(f"\n   Sentence boundaries: {sentence_endings}/{len(chunks)} ({100*sentence_endings/len(chunks):.1f}%)")
        
        return chunks


def main():
    """Run enhanced PDF processing."""
    # Updated default parameters for better RAG performance
    processor = EnhancedSemanticPDFProcessor(
        pdf_path="data/book.pdf",
        min_chunk_words=100,
        target_chunk_words=300,
        max_chunk_words=500,
        overlap_sentences=5
    )
    
    try:
        chunks = processor.process_and_save()
        
        if chunks and len(chunks) >= 3:
            print(f"\n\n Sample Chunks:")
            
            print(f"\n{'='*80}")
            print(f"REGULAR CHUNK EXAMPLE:")
            chunk = chunks[0]
            print(f"Chunk {chunk['chunk_id']}:")
            print(f"Chapter: {chunk['chapter']}")
            print(f"Pages: {chunk['page_start']}-{chunk['page_end']}")
            print(f"Words: {chunk['word_count']}")
            print(f"Definition context: {chunk.get('is_definition', False)}")
            print(f"\nText preview:")
            print(f"{chunk['text'][:300]}...")
            
            def_chunks = [c for c in chunks if c.get('is_definition', False)]
            if def_chunks:
                print(f"\n{'='*80}")
                print(f"DEFINITION CHUNK EXAMPLE:")
                chunk = def_chunks[0]
                print(f"Chunk {chunk['chunk_id']}:")
                print(f"Chapter: {chunk['chapter']}")
                print(f"Pages: {chunk['page_start']}-{chunk['page_end']}")
                print(f"Words: {chunk['word_count']}")
                print(f"Coherence: {chunk.get('coherence_score', 0):.2f}")
                print(f"\nText preview:")
                print(f"{chunk['text'][:400]}...")
    
    except Exception as e:
        print(f"\n Error: {e}")
        raise


if __name__ == "__main__":
    main()