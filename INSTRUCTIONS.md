# How to Run the IC Engine RAG System

## Quick Start (3 Steps)

### Step 1: Get Your Groq API Key

1. Go to https://console.groq.com/
2. Sign up for a free account
3. Navigate to API Keys section
4. Create a new API key
5. Copy the key

### Step 2: Set the API Key

In your terminal, run:

```bash
export GROQ_API_KEY='paste-your-api-key-here'
```

**Important**: Replace `paste-your-api-key-here` with your actual API key!

### Step 3: Run the Complete Pipeline

```bash
cd Assignment
./run_complete_pipeline.sh
```

That's it! The system will:
1. Load and process the PDF (78 pages)
2. Create embeddings for all chunks
3. Build FAISS vector store
4. Answer all 48 questions
5. Evaluate answers against ground truth
6. Generate `eval.json` with scores

## Alternative: Run Steps Individually

If you want more control:

```bash
# Step 1: Generate answers
python3 run_rag.py

# Step 2: Evaluate answers
python3 evaluate.py
```

## What to Expect

### During Execution

```
Loading embedding model...
Loading reranker model...
Loading PDF: introduction to internal combustion engines.pdf
Total pages: 78
Chunking text...
Created ~150 chunks
Creating embeddings...
Building FAISS index...
Answering questions: 100%|████████| 48/48
Evaluating: 100%|████████| 48/48
```

### Output Files

1. **answers.json**: All generated answers
   ```json
   [
     {
       "question": "What is an engine?",
       "answer": "A device which transforms one form of energy into another form..."
     },
     ...
   ]
   ```

2. **eval.json**: Evaluation results
   ```json
   {
     "final_score": 0.828,
     "num_samples": 48,
     "results": [
       {
         "question": "...",
         "prediction": "...",
         "ground_truth": "...",
         "score": 0.9,
         "reason": "..."
       }
     ]
   }
   ```

## Expected Performance

- **Final Score**: 0.80 - 0.85 (target)
- **Time**: 5-10 minutes total
  - Answer generation: 3-5 minutes
  - Evaluation: 2-5 minutes
- **API Calls**: ~100 calls to Groq (well within free tier)

## Troubleshooting

### Error: "GROQ_API_KEY environment variable is not set"

**Solution**: Make sure you exported the API key:
```bash
export GROQ_API_KEY='your-key-here'
```

To verify it's set:
```bash
echo $GROQ_API_KEY
```

### Error: "pip: command not found"

**Solution**: Use pip3 instead:
```bash
pip3 install -r requirements.txt
```

### Error: "Module not found"

**Solution**: Reinstall dependencies:
```bash
pip3 install -r requirements.txt
```

### Slow Performance

**Normal**: First run downloads models (~100MB)
- sentence-transformers model
- cross-encoder model

Subsequent runs will be faster.

## Understanding the Results

### Score Interpretation

- **1.0**: Perfect answer
- **0.9**: Excellent, minor details missing
- **0.8**: Good, some details missing
- **0.7**: Acceptable, significant details missing
- **0.6**: Partial, oversimplified
- **<0.6**: Needs improvement

### Sample Result

```json
{
  "question": "What is an internal combustion engine?",
  "prediction": "A heat engine that converts chemical energy of fuel into mechanical energy by combustion inside the engine.",
  "ground_truth": "A heat engine that converts chemical energy of fuel into mechanical energy by combustion inside the engine.",
  "score": 1.0,
  "reason": "Perfect match with ground truth"
}
```

## System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    PDF Document                          │
│         (78 pages, ~22K characters)                      │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│              Chunking (800 chars, 150 overlap)          │
│                   ~150 chunks                            │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│         Embedding (all-MiniLM-L6-v2)                    │
│              384-dim vectors                             │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│              FAISS Vector Store                          │
│           (Fast similarity search)                       │
└────────────────────┬────────────────────────────────────┘
                     │
        ┌────────────┴────────────┐
        │      Question           │
        └────────────┬────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│         Retrieval (Top 10 chunks)                       │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│    Reranking (Cross-encoder → Top 5)                    │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│   LLM Generation (Groq llama-3.3-70b)                   │
│         Temperature: 0.1                                 │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
                  Answer
```

## Next Steps

After running the pipeline:

1. **Check eval.json** for final score
2. **Review low-scoring answers** to understand issues
3. **Optionally tune parameters**:
   - Chunk size (currently 800)
   - Overlap (currently 150)
   - Retrieval k (currently 10)
   - Rerank k (currently 5)
   - Temperature (currently 0.1)

## Files Created

- ✅ `rag_system.py` - Core RAG implementation
- ✅ `run_rag.py` - Answer generation script
- ✅ `evaluate.py` - Evaluation script
- ✅ `answers.json` - Generated answers (after running)
- ✅ `eval.json` - Evaluation results (after running)

