"""
Main script to run the complete RAG pipeline
1. EDA
2. Build RAG system
3. Generate answers
4. Evaluate
"""

import os
import json
from rag_system import ICEngineRAG
from tqdm import tqdm

def main():
    # Configuration
    PDF_PATH = "introduction to internal combustion engines.pdf"
    QUESTIONS_PATH = "questions.json"
    ANSWERS_PATH = "answers.json"
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    
    if not GROQ_API_KEY:
        print("ERROR: Please set GROQ_API_KEY environment variable")
        print("Run: export GROQ_API_KEY='your-api-key-here'")
        return
    
    print("="*80)
    print("IC ENGINE RAG SYSTEM - BUILDING PIPELINE")
    print("="*80)
    
    # Initialize RAG system
    rag = ICEngineRAG(PDF_PATH, GROQ_API_KEY)
    
    # Step 1: Load PDF
    print("\n[STEP 1] Loading PDF...")
    text = rag.load_pdf()
    print(f"First 500 characters:\n{text[:500]}\n")
    
    # Step 2: Chunk text
    print("\n[STEP 2] Chunking text...")
    rag.chunks = rag.chunk_text(text, chunk_size=800, overlap=150)
    print(f"Sample chunk:\n{rag.chunks[0]}\n")
    
    # Step 3: Create embeddings
    print("\n[STEP 3] Creating embeddings...")
    rag.embeddings = rag.create_embeddings(rag.chunks)
    print(f"Embedding shape: {rag.embeddings.shape}")
    
    # Step 4: Build vector store
    print("\n[STEP 4] Building FAISS vector store...")
    rag.build_vector_store(rag.embeddings)
    
    # Step 5: Test retrieval
    print("\n[STEP 5] Testing retrieval with reranking...")
    test_query = "What is an internal combustion engine?"
    retrieved = rag.retrieve(test_query, k=10)
    print(f"Retrieved {len(retrieved)} chunks")
    reranked = rag.rerank(test_query, retrieved, top_k=5)
    print(f"Reranked to top {len(reranked)} chunks")
    
    # Step 6: Test answer generation
    print("\n[STEP 6] Testing answer generation...")
    test_answer = rag.answer_question(test_query)
    print(f"Q: {test_query}")
    print(f"A: {test_answer}\n")
    
    # Step 7: Load questions
    print("\n[STEP 7] Loading questions...")
    with open(QUESTIONS_PATH, 'r') as f:
        questions_data = json.load(f)
    print(f"Loaded {len(questions_data)} questions")
    
    # Step 8: Generate answers for all questions
    print("\n[STEP 8] Generating answers for all questions...")
    answers = []
    
    for item in tqdm(questions_data, desc="Answering questions"):
        question = item['question']
        try:
            answer = rag.answer_question(question, retrieve_k=10, rerank_k=5)
            answers.append({
                "question": question,
                "answer": answer
            })
        except Exception as e:
            print(f"\nError answering '{question}': {e}")
            answers.append({
                "question": question,
                "answer": "Unable to answer from the provided documents."
            })
    
    # Step 9: Save answers
    print("\n[STEP 9] Saving answers...")
    with open(ANSWERS_PATH, 'w') as f:
        json.dump(answers, f, indent=2)
    print(f"Answers saved to {ANSWERS_PATH}")
    
    print("\n" + "="*80)
    print("RAG PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*80)
    print(f"\nGenerated {len(answers)} answers")
    print(f"Next step: Run evaluation script to compare with ground truth")

if __name__ == "__main__":
    main()

