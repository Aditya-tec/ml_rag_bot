import os
import json
import time
import re
from src.rag_pipeline import RAGPipeline
from dotenv import load_dotenv
from groq import Groq

# Load environment
load_dotenv()

# Configure Groq
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError("GROQ_API_KEY not found in .env file")

groq_client = Groq(api_key=api_key)

# ALL 28 TEST QUESTIONS
test_questions = [
    "What is the triple barrier method?",
    "Explain fractionally differentiated features",
    "What are the dangers of backtesting?",
    "How does cross-validation work in financial ML?",
    "What is meta-labeling?",
    "Explain the concept of sample weights",
    "What is VPIN and how is it used?",
    "How does feature importance work in financial ML?",
    "What are ensemble methods in machine learning?",
    "How do you handle non-IID samples in financial data?",
    "What is the volume clock and why is it useful?",
    "Explain the concept of uniqueness weighting",
    "What is the significance of the flash crash of 2010?",
    "How does principal component analysis apply to financial features?",
    "What are dollar bars and why are they useful?",
    "Explain the concept of synthetic data in backtesting",
]


class GroqRAGEvaluator:
    """Evaluate RAG using Groq with proper Gemini rate limiting."""
    
    def __init__(self, rag_pipeline):
        self.rag = rag_pipeline
        self.groq = groq_client
        self.model = "llama-3.1-8b-instant"
        
        # Rate limiting for Gemini (used by RAG pipeline)
        self.gemini_requests = 0
        self.gemini_reset_time = time.time()
        self.GEMINI_REQUESTS_PER_MINUTE = 2 # Conservative limit
        self.GEMINI_WAIT_BETWEEN_CALLS = 12 # 7 seconds between calls
    
    def _wait_for_gemini_rate_limit(self):
        """Ensure we don't exceed Gemini free tier limits."""
        current_time = time.time()
        elapsed = current_time - self.gemini_reset_time
        
        # Reset counter every 60 seconds
        if elapsed >= 60:
            self.gemini_requests = 0
            self.gemini_reset_time = current_time
            print("    Rate limit counter reset")
        
        # If approaching limit, wait
        if self.gemini_requests >= self.GEMINI_REQUESTS_PER_MINUTE:
            wait_time = 60 - elapsed + 2
            if wait_time > 0:
                print(f"    Gemini rate limit: waiting {wait_time:.0f}s...")
                time.sleep(wait_time)
                self.gemini_requests = 0
                self.gemini_reset_time = time.time()
        
        # Always wait between calls
        print(f"    Waiting {self.GEMINI_WAIT_BETWEEN_CALLS}s before next RAG query...")
        time.sleep(self.GEMINI_WAIT_BETWEEN_CALLS)
        
        self.gemini_requests += 1
    
    def _call_groq(self, prompt: str) -> str:
        """Call Groq API for evaluation."""
        try:
            chat_completion = self.groq.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert evaluator. Respond only with valid JSON."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                model=self.model,
                temperature=0.1,
                max_tokens=500,
            )
            return chat_completion.choices[0].message.content
        except Exception as e:
            print(f"    Groq Error: {e}")
            return '{"score": 0.0, "error": "Groq API call failed"}'
    
    def evaluate_answer_relevancy(self, question: str, answer: str) -> dict:
        """Evaluate if answer is relevant to question."""
        prompt = f"""Rate the relevance of the answer to the question on a scale of 0 to 1.

Question: {question}

Answer: {answer}

Criteria:
1. Does the answer directly address the question?
2. Is it focused and on-topic?
3. Does it provide useful information?

Respond ONLY with valid JSON:
{{"score": 0.85, "reasoning": "Brief explanation", "is_relevant": true}}"""
        
        response = self._call_groq(prompt)
        result = self._parse_json_response(response)
        result['metric'] = 'Answer Relevancy'
        return result
    
    def evaluate_faithfulness(self, answer: str, context: list) -> dict:
        """Evaluate if answer is faithful to retrieved context."""
        context_text = "\n\n".join(context[:3])[:2000]
        
        prompt = f"""Rate how faithful the answer is to the provided context on a scale of 0 to 1.

Context:
{context_text}

Answer:
{answer}

Criteria:
1. Are all claims supported by the context?
2. No made-up information?

Respond ONLY with valid JSON:
{{"score": 0.9, "reasoning": "Brief explanation", "is_faithful": true}}"""
        
        response = self._call_groq(prompt)
        result = self._parse_json_response(response)
        result['metric'] = 'Faithfulness'
        return result
    
    def evaluate_context_relevancy(self, question: str, context: list) -> dict:
        """Evaluate if retrieved context is relevant to question."""
        context_text = "\n\n".join(context[:3])[:2000]
        
        prompt = f"""Rate how relevant the retrieved context is for answering the question on a scale of 0 to 1.

Question: {question}

Context:
{context_text}

Criteria:
1. Does context contain needed information?
2. Is it focused and not filled with irrelevant info?

Respond ONLY with valid JSON:
{{"score": 0.88, "reasoning": "Brief explanation", "is_relevant": true}}"""
        
        response = self._call_groq(prompt)
        result = self._parse_json_response(response)
        result['metric'] = 'Context Relevancy'
        return result
    
    def evaluate_completeness(self, question: str, answer: str) -> dict:
        """Evaluate if answer is complete."""
        prompt = f"""Rate the completeness of the answer on a scale of 0 to 1.

Question: {question}

Answer: {answer}

Criteria:
1. Does it fully address all aspects?
2. Are key concepts explained adequately?

Respond ONLY with valid JSON:
{{"score": 0.92, "reasoning": "Brief explanation", "is_complete": true}}"""
        
        response = self._call_groq(prompt)
        result = self._parse_json_response(response)
        result['metric'] = 'Completeness'
        return result
    
    def _parse_json_response(self, text: str) -> dict:
        """Parse JSON from LLM response."""
        text = text.strip()
        
        # Remove markdown if present
        if text.startswith('```json'):
            text = text[7:]
        elif text.startswith('```'):
            text = text[3:]
        if text.endswith('```'):
            text = text[:-3]
        
        text = text.strip()
        
        # Extract JSON object
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if json_match:
            text = json_match.group()
        
        try:
            result = json.loads(text)
            if 'score' not in result:
                result['score'] = 0.0
            return result
        except json.JSONDecodeError:
            return {
                "score": 0.0,
                "reasoning": "Failed to parse response",
                "error": "JSON parse error"
            }
    
    def evaluate_question(self, question: str, question_num: int, total: int) -> dict:
        """Run all evaluations for a single question."""
        print(f"\n[{question_num}/{total}] {question}")
        print("-" * 80)
        
        # Get RAG response with rate limiting
        print("  Getting RAG answer (with Gemini rate limiting)...")
        self._wait_for_gemini_rate_limit()
        
        try:
            result = self.rag.query(question, top_k=10, return_sources=True)
            answer = result['answer']
            context = [s['text'] for s in result['sources']]
            
            # Check if answer is an error
            if "Error generating answer" in answer or "429" in answer:
                print("  RAG returned error - skipping evaluation")
                return {
                    'question': question,
                    'answer': answer,
                    'num_sources': 0,
                    'evaluations': {
                        'relevancy': {'score': 0.0, 'error': 'RAG quota error'},
                        'faithfulness': {'score': 0.0, 'error': 'RAG quota error'},
                        'context_relevancy': {'score': 0.0, 'error': 'RAG quota error'},
                        'completeness': {'score': 0.0, 'error': 'RAG quota error'}
                    }
                }
        except Exception as e:
            print(f"  Error getting RAG answer: {e}")
            return {
                'question': question,
                'answer': f"Error: {str(e)}",
                'num_sources': 0,
                'evaluations': {
                    'relevancy': {'score': 0.0, 'error': str(e)},
                    'faithfulness': {'score': 0.0, 'error': str(e)},
                    'context_relevancy': {'score': 0.0, 'error': str(e)},
                    'completeness': {'score': 0.0, 'error': str(e)}
                }
            }
        
        evals = {
            'question': question,
            'answer': answer,
            'num_sources': len(context),
            'evaluations': {}
        }
        
        print("  Evaluating with Groq...")
        
        print("    Answer Relevancy...")
        evals['evaluations']['relevancy'] = self.evaluate_answer_relevancy(question, answer)
        print(f"      Score: {evals['evaluations']['relevancy']['score']:.3f}")
        
        print("    Faithfulness...")
        evals['evaluations']['faithfulness'] = self.evaluate_faithfulness(answer, context)
        print(f"      Score: {evals['evaluations']['faithfulness']['score']:.3f}")
        
        print("    Context Relevancy...")
        evals['evaluations']['context_relevancy'] = self.evaluate_context_relevancy(question, context)
        print(f"      Score: {evals['evaluations']['context_relevancy']['score']:.3f}")
        
        print("    Completeness...")
        evals['evaluations']['completeness'] = self.evaluate_completeness(question, answer)
        print(f"      Score: {evals['evaluations']['completeness']['score']:.3f}")
        
        # Calculate question average
        scores = [evals['evaluations'][m]['score'] for m in ['relevancy', 'faithfulness', 'context_relevancy', 'completeness']]
        avg = sum(scores) / len(scores)
        print(f"  Question Average: {avg:.3f}")
        
        return evals
    
    def evaluate_all(self, questions: list) -> dict:
        """Evaluate all questions."""
        results = []
        
        print("="*80)
        print("RAG EVALUATION USING GROQ")
        print("="*80)
        print(f"Total questions: {len(questions)}")
        print(f"Estimated time: ~{len(questions) * 10}s (with Gemini rate limiting)")
        print("Note: 7 second delay between questions to avoid Gemini quota")
        print()
        
        start_time = time.time()
        
        for i, question in enumerate(questions, 1):
            result = self.evaluate_question(question, i, len(questions))
            results.append(result)
        
        elapsed = time.time() - start_time
        
        # Filter out error results before calculating averages
        valid_results = [r for r in results if r['num_sources'] > 0]
        
        if valid_results:
            avg_scores = self._calculate_averages(valid_results)
        else:
            avg_scores = {
                'relevancy': 0.0,
                'faithfulness': 0.0,
                'context_relevancy': 0.0,
                'completeness': 0.0,
                'overall': 0.0
            }
        
        return {
            'individual_results': results,
            'average_scores': avg_scores,
            'total_time': elapsed,
            'questions_evaluated': len(questions),
            'successful_evaluations': len(valid_results),
            'failed_evaluations': len(questions) - len(valid_results)
        }
    
    def _calculate_averages(self, results: list) -> dict:
        """Calculate average scores from valid results only."""
        metrics = ['relevancy', 'faithfulness', 'context_relevancy', 'completeness']
        averages = {}
        
        for metric in metrics:
            scores = [r['evaluations'][metric]['score'] for r in results 
                     if 'score' in r['evaluations'][metric] and 'error' not in r['evaluations'][metric]]
            averages[metric] = sum(scores) / len(scores) if scores else 0.0
        
        averages['overall'] = sum(averages.values()) / len(averages) if averages else 0.0
        
        return averages
    
    def print_summary(self, results: dict):
        """Print evaluation summary."""
        print("\n" + "="*80)
        print("EVALUATION SUMMARY")
        print("="*80)
        
        avg = results['average_scores']
        
        print(f"\nAverage Scores (from successful evaluations):")
        print(f"  Answer Relevancy:    {avg['relevancy']:.3f}")
        print(f"  Faithfulness:        {avg['faithfulness']:.3f}")
        print(f"  Context Relevancy:   {avg['context_relevancy']:.3f}")
        print(f"  Completeness:        {avg['completeness']:.3f}")
        print(f"  " + "-"*40)
        print(f"  OVERALL:             {avg['overall']:.3f}")
        
        print(f"\nEvaluation Stats:")
        print(f"  Total Questions:     {results['questions_evaluated']}")
        print(f"  Successful:          {results['successful_evaluations']}")
        print(f"  Failed (quota):      {results['failed_evaluations']}")
        print(f"  Total Time:          {results['total_time']:.1f}s")
        print(f"  Avg Time/Question:   {results['total_time']/results['questions_evaluated']:.1f}s")
        
        # Interpretation
        if avg['overall'] >= 0.8:
            grade = "EXCELLENT"
            emoji = "🟢"
        elif avg['overall'] >= 0.7:
            grade = "GOOD"
            emoji = "🟡"
        elif avg['overall'] >= 0.6:
            grade = "FAIR"
            emoji = "🟠"
        else:
            grade = "NEEDS IMPROVEMENT"
            emoji = "🔴"
        
        print(f"\nOverall Grade: {emoji} {grade}")
        
        # Save detailed results
        with open('evaluation_results_groq.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nDetailed results saved to: evaluation_results_groq.json")
        
        # Print per-question summary (successful only)
        print(f"\n" + "="*80)
        print("PER-QUESTION SCORES (Successful Evaluations Only)")
        print("="*80)
        
        for i, result in enumerate(results['individual_results'], 1):
            if result['num_sources'] == 0:
                print(f"✗ Q{i:2d}: SKIPPED (quota error) - {result['question'][:50]}...")
                continue
                
            evals = result['evaluations']
            scores = [evals[m]['score'] for m in ['relevancy', 'faithfulness', 'context_relevancy', 'completeness']]
            avg_score = sum(scores) / len(scores)
            
            status = "✓" if avg_score >= 0.7 else "✗"
            print(f"{status} Q{i:2d}: {avg_score:.3f} - {result['question'][:60]}...")


def main():
    print("Loading RAG pipeline...")
    rag = RAGPipeline()
    print("RAG pipeline loaded!\n")
    
    evaluator = GroqRAGEvaluator(rag)
    
    results = evaluator.evaluate_all(test_questions)
    
    evaluator.print_summary(results)
    
    print("\n" + "="*80)
    print("EVALUATION COMPLETE!")
    print("="*80)


if __name__ == "__main__":
    main()