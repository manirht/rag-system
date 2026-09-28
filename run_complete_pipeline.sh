#!/bin/bash

# Complete RAG Pipeline Runner
# This script runs the entire pipeline from start to finish

echo "=========================================="
echo "IC ENGINE RAG - COMPLETE PIPELINE"
echo "=========================================="
echo ""

# Check if GROQ_API_KEY is set
if [ -z "$GROQ_API_KEY" ]; then
    echo "ERROR: GROQ_API_KEY environment variable is not set"
    echo ""
    echo "Please set it using:"
    echo "  export GROQ_API_KEY='your-api-key-here'"
    echo ""
    echo "Get your free API key from: https://console.groq.com/"
    exit 1
fi

echo "✓ GROQ_API_KEY is set"
echo ""

# Step 1: Generate answers
echo "=========================================="
echo "STEP 1: Generating Answers"
echo "=========================================="
python3 run_rag.py

if [ $? -ne 0 ]; then
    echo ""
    echo "ERROR: Answer generation failed"
    exit 1
fi

echo ""
echo "✓ Answers generated successfully"
echo ""

# Step 2: Evaluate answers
echo "=========================================="
echo "STEP 2: Evaluating Answers"
echo "=========================================="
python3 evaluate.py

if [ $? -ne 0 ]; then
    echo ""
    echo "ERROR: Evaluation failed"
    exit 1
fi

echo ""
echo "✓ Evaluation completed successfully"
echo ""

# Summary
echo "=========================================="
echo "PIPELINE COMPLETED SUCCESSFULLY!"
echo "=========================================="
echo ""
echo "Output files:"
echo "  - answers.json: Generated answers"
echo "  - eval.json: Evaluation results"
echo ""
echo "Check eval.json for final score and detailed results"

