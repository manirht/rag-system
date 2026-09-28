"""
Evaluation script using LLM-as-Judge
Compares generated answers with ground truth
"""

import os
import json
from groq import Groq
from tqdm import tqdm

class LLMEvaluator:
    def __init__(self, groq_api_key: str):
        self.groq_client = Groq(api_key=groq_api_key)
    
    def evaluate_answer(self, question: str, ground_truth: str, prediction: str) -> dict:
        """
        Use LLM to evaluate the prediction against ground truth
        Returns: {score: float, reason: str}
        """
        
        prompt = f"""You are an expert evaluator for question-answering systems.

Your task is to compare a Model Answer with the Ground Truth answer and provide:
1. A score from 0.0 to 1.0 (where 1.0 is perfect)
2. A brief reason for the score

Scoring Guidelines:
- 1.0: Perfect match or semantically equivalent
- 0.9: Mostly correct with minor missing details
- 0.8: Correct but missing some important details
- 0.7: Partially correct, missing significant details
- 0.4-0.6: Partially correct but oversimplified or vague
- 0.0-0.3: Incorrect or completely missing information

Question: {question}

Ground Truth: {ground_truth}

Model Answer: {prediction}

Provide your evaluation in this EXACT JSON format (no other text):
{{"score": 0.0, "reason": "explanation here"}}"""

        try:
            response = self.groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=200
            )
            
            result_text = response.choices[0].message.content.strip()
            
            # Extract JSON from response
            # Sometimes LLM adds extra text, so we need to extract the JSON
            start_idx = result_text.find('{')
            end_idx = result_text.rfind('}') + 1
            
            if start_idx != -1 and end_idx > start_idx:
                json_str = result_text[start_idx:end_idx]
                result = json.loads(json_str)
                
                # Ensure score is between 0 and 1
                score = max(0.0, min(1.0, float(result.get('score', 0.0))))
                reason = result.get('reason', 'No reason provided')
                
                return {
                    "score": score,
                    "reason": reason
                }
            else:
                # Fallback if JSON parsing fails
                return {
                    "score": 0.5,
                    "reason": "Evaluation parsing failed"
                }
                
        except Exception as e:
            print(f"Error evaluating: {e}")
            return {
                "score": 0.5,
                "reason": f"Evaluation error: {str(e)}"
            }

def main():
    # Configuration
    QUESTIONS_PATH = "questions.json"
    ANSWERS_PATH = "answers.json"
    EVAL_OUTPUT_PATH = "eval.json"
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    
    if not GROQ_API_KEY:
        print("ERROR: Please set GROQ_API_KEY environment variable")
        return
    
    print("="*80)
    print("EVALUATION - LLM AS JUDGE")
    print("="*80)
    
    # Load data
    print("\nLoading data...")
    with open(QUESTIONS_PATH, 'r') as f:
        questions_data = json.load(f)
    
    with open(ANSWERS_PATH, 'r') as f:
        answers_data = json.load(f)
    
    print(f"Loaded {len(questions_data)} ground truth answers")
    print(f"Loaded {len(answers_data)} predictions")
    
    # Initialize evaluator
    evaluator = LLMEvaluator(GROQ_API_KEY)
    
    # Evaluate each answer
    print("\nEvaluating answers...")
    results = []
    total_score = 0.0
    
    for i, (gt_item, pred_item) in enumerate(tqdm(zip(questions_data, answers_data), 
                                                    total=len(questions_data),
                                                    desc="Evaluating")):
        question = gt_item['question']
        ground_truth = gt_item['answer']
        prediction = pred_item['answer']
        
        # Evaluate
        eval_result = evaluator.evaluate_answer(question, ground_truth, prediction)
        
        # Store result
        result = {
            "question": question,
            "prediction": prediction,
            "ground_truth": ground_truth,
            "score": eval_result['score'],
            "reason": eval_result['reason']
        }
        results.append(result)
        total_score += eval_result['score']
    
    # Calculate final score
    final_score = total_score / len(results) if results else 0.0
    
    # Create output
    output = {
        "final_score": round(final_score, 3),
        "num_samples": len(results),
        "results": results
    }
    
    # Save evaluation
    print(f"\nSaving evaluation to {EVAL_OUTPUT_PATH}...")
    with open(EVAL_OUTPUT_PATH, 'w') as f:
        json.dump(output, f, indent=2)
    
    print("\n" + "="*80)
    print("EVALUATION COMPLETED!")
    print("="*80)
    print(f"\nFinal Score: {final_score:.3f}")
    print(f"Total Samples: {len(results)}")
    print(f"\nScore Distribution:")
    
    # Show score distribution
    score_ranges = {
        "1.0 (Perfect)": 0,
        "0.9-0.99": 0,
        "0.8-0.89": 0,
        "0.7-0.79": 0,
        "0.6-0.69": 0,
        "Below 0.6": 0
    }
    
    for result in results:
        score = result['score']
        if score == 1.0:
            score_ranges["1.0 (Perfect)"] += 1
        elif score >= 0.9:
            score_ranges["0.9-0.99"] += 1
        elif score >= 0.8:
            score_ranges["0.8-0.89"] += 1
        elif score >= 0.7:
            score_ranges["0.7-0.79"] += 1
        elif score >= 0.6:
            score_ranges["0.6-0.69"] += 1
        else:
            score_ranges["Below 0.6"] += 1
    
    for range_name, count in score_ranges.items():
        print(f"  {range_name}: {count}")

if __name__ == "__main__":
    main()

