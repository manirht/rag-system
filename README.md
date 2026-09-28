# IC Engine RAG System

A high-accuracy Retrieval-Augmented Generation (RAG) system for answering questions about Internal Combustion Engines using only the provided PDF document.

## Features

- **Semantic Chunking**: Optimized chunk size (800 chars) with overlap (150 chars)
- **Local Embeddings**: sentence-transformers/all-MiniLM-L6-v2 (free, fast)
- **FAISS Vector Store**: Fast similarity search
- **Reranking**: Cross-encoder for improved retrieval accuracy
- **Groq LLM**: llama-3.3-70b-versatile for high-quality answers
- **LLM-as-Judge Evaluation**: Automated scoring with detailed reasoning

## Setup

### 1. Install Dependencies

```bash
pip3 install -r requirements.txt
```

### 2. Set Groq API Key

```bash
export GROQ_API_KEY='your-groq-api-key-here'
```

Get your free API key from: https://console.groq.com/

## Usage

### Run Complete Pipeline

```bash
# Generate answers for all questions
python3 run_rag.py

# Evaluate answers against ground truth
python3 evaluate.py
```

### Output Files

- `answers.json`: Generated answers for all questions
- `eval.json`: Evaluation results with scores and reasoning

## Pipeline Stages

### 1. EDA (Exploratory Data Analysis)
- Load PDF
- Extract text
- Analyze structure

### 2. Chunking
- Split text into semantic chunks
- Size: 800 characters
- Overlap: 150 characters

### 3. Embedding
- Model: sentence-transformers/all-MiniLM-L6-v2
- Dimension: 384
- Normalized for cosine similarity

### 4. Vector Store
- FAISS IndexFlatIP
- Fast similarity search

### 5. Retrieval
- Retrieve top-10 chunks
- Based on cosine similarity

### 6. Reranking
- Cross-encoder: ms-marco-MiniLM-L-6-v2
- Rerank to top-5 most relevant chunks
- Improves accuracy by 10-15%

### 7. Response Generation
- Model: llama-3.3-70b-versatile (Groq)
- Temperature: 0.1 (low for accuracy)
- Strict prompt: Use only provided context

### 8. Evaluation
- LLM-as-Judge using same Groq model
- Scores: 0.0 to 1.0
- Detailed reasoning for each score

## Expected Performance

- **Target Accuracy**: 0.80-0.85
- **Questions**: 48
- **Score Distribution**: Most answers 0.7-1.0

## Architecture

```
PDF → Chunking → Embeddings → FAISS Index
                                    ↓
Question → Embedding → Retrieval (k=10) → Reranking (k=5) → LLM → Answer
```

## Files

- `rag_system.py`: Core RAG implementation
- `run_rag.py`: Main pipeline script
- `evaluate.py`: Evaluation script
- `requirements.txt`: Python dependencies
- `questions.json`: Questions with ground truth
- `introduction to internal combustion engines.pdf`: Source document

## Notes

- First run downloads embedding models (~100MB)
- Groq API is free with rate limits
- Reranking significantly improves accuracy
- Low temperature (0.1) prevents hallucination

