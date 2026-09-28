"""
RAG System for IC Engine Q&A
Optimized for accuracy with reranking
"""

import os
import json
from typing import List, Dict, Tuple
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer, CrossEncoder
from langchain.text_splitter import RecursiveCharacterTextSplitter
import faiss
import numpy as np
from groq import Groq
from tqdm import tqdm

class ICEngineRAG:
    def __init__(self, pdf_path: str, groq_api_key: str):
        """Initialize RAG system with PDF and API key"""
        self.pdf_path = pdf_path
        self.groq_client = Groq(api_key=groq_api_key)
        
        # Initialize embedding model (local, free)
        print("Loading embedding model...")
        self.embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        
        # Initialize reranker (cross-encoder for better accuracy)
        print("Loading reranker model...")
        self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        
        self.chunks = []
        self.embeddings = None
        self.index = None
        
    def load_pdf(self) -> str:
        """Load and extract text from PDF"""
        print(f"Loading PDF: {self.pdf_path}")
        reader = PdfReader(self.pdf_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        
        print(f"Total pages: {len(reader.pages)}")
        print(f"Total characters: {len(text)}")
        return text
    
    def chunk_text(self, text: str, chunk_size: int = 800, overlap: int = 150) -> List[str]:
        """Split text into chunks with overlap for better context"""
        print(f"Chunking text (size={chunk_size}, overlap={overlap})...")
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        chunks = text_splitter.split_text(text)
        print(f"Created {len(chunks)} chunks")
        return chunks
    
    def create_embeddings(self, chunks: List[str]) -> np.ndarray:
        """Create embeddings for all chunks"""
        print("Creating embeddings...")
        embeddings = self.embedding_model.encode(
            chunks,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True  # For cosine similarity
        )
        return embeddings
    
    def build_vector_store(self, embeddings: np.ndarray):
        """Build FAISS index for fast similarity search"""
        print("Building FAISS index...")
        dimension = embeddings.shape[1]
        
        # Using IndexFlatIP for inner product (cosine similarity with normalized vectors)
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(embeddings.astype('float32'))
        
        print(f"Index built with {self.index.ntotal} vectors")
    
    def retrieve(self, query: str, k: int = 10) -> List[Tuple[str, float]]:
        """Retrieve top-k most relevant chunks"""
        # Embed query
        query_embedding = self.embedding_model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        
        # Search in FAISS
        scores, indices = self.index.search(query_embedding.astype('float32'), k)
        
        # Return chunks with scores
        results = [(self.chunks[idx], scores[0][i]) for i, idx in enumerate(indices[0])]
        return results
    
    def rerank(self, query: str, chunks: List[Tuple[str, float]], top_k: int = 5) -> List[str]:
        """Rerank retrieved chunks using cross-encoder for better accuracy"""
        # Prepare pairs for reranking
        pairs = [[query, chunk] for chunk, _ in chunks]
        
        # Get reranking scores
        rerank_scores = self.reranker.predict(pairs)
        
        # Sort by reranking scores
        ranked_indices = np.argsort(rerank_scores)[::-1][:top_k]
        
        # Return top-k reranked chunks
        reranked_chunks = [chunks[idx][0] for idx in ranked_indices]
        return reranked_chunks
    
    def generate_answer(self, query: str, context_chunks: List[str], temperature: float = 0.1) -> str:
        """Generate answer using Groq API with retrieved context"""
        # Combine context
        context = "\n\n".join([f"[Context {i+1}]\n{chunk}" for i, chunk in enumerate(context_chunks)])
        
        # Create prompt
        prompt = f"""You are a precise question-answering system for Internal Combustion Engines.

Answer the question using ONLY the context provided below. Be accurate and concise.

IMPORTANT RULES:
1. Use ONLY information from the provided context
2. Do NOT use external knowledge or make assumptions
3. If the answer cannot be found in the context, respond EXACTLY with: "Unable to answer from the provided documents."
4. Be direct and factual
5. Keep answers concise but complete

Context:
{context}

Question: {query}

Answer:"""
        
        # Call Groq API
        response = self.groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",  # High accuracy model
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=300
        )
        
        return response.choices[0].message.content.strip()
    
    def answer_question(self, question: str, retrieve_k: int = 10, rerank_k: int = 5) -> str:
        """Complete RAG pipeline: retrieve -> rerank -> generate"""
        # Step 1: Retrieve
        retrieved_chunks = self.retrieve(question, k=retrieve_k)
        
        # Step 2: Rerank
        reranked_chunks = self.rerank(question, retrieved_chunks, top_k=rerank_k)
        
        # Step 3: Generate
        answer = self.generate_answer(question, reranked_chunks)
        
        return answer

