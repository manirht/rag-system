# 🚀 Quick Start Guide

## Run in 3 Commands

```bash
# 1. Get your free Groq API key from https://console.groq.com/
export GROQ_API_KEY='your-api-key-here'

# 2. Navigate to Assignment folder
cd Assignment

# 3. Run the complete pipeline
./run_complete_pipeline.sh
```

## What Happens

```
✓ Loading PDF (78 pages)
✓ Creating ~150 chunks
✓ Generating embeddings
✓ Building FAISS index
✓ Answering 48 questions (3-5 min)
✓ Evaluating answers (2-5 min)
✓ Final score: 0.80-0.85 (expected)
```

## Output

- **answers.json** - All 48 answers
- **eval.json** - Scores and evaluation

## Check Results

```bash
# View final score
cat eval.json | grep "final_score"

# View first few results
cat eval.json | head -50
```

## That's It! 🎉

The system is fully automated. Just provide your API key and run!

---

**Need help?** Check:
- `INSTRUCTIONS.md` - Detailed guide
- `README.md` - Full documentation
- `SUMMARY.md` - Technical details

