# Retrieval-Augmented Generation (RAG) System

## 📌 Overview
This project implements a simple Retrieval-Augmented Generation (RAG) system that answers
questions based on a collection of text documents.

The system generates two types of answers for the same question:
1. Transformer-based answer (retrieval-only, no external LLM)
2. LLM-based answer (retrieval + large language model)

This allows direct comparison between traditional transformer-based retrieval
and modern LLM-augmented generation.

---

## 🧠 What is RAG?
Retrieval-Augmented Generation combines:
- Information Retrieval (finding relevant documents)
- Text Generation (generating answers using retrieved context)

This approach reduces hallucinations and ensures answers are grounded in source documents.

---

## 🏗️ System Architecture

1. Document ingestion  
2. Text chunking  
3. Embedding generation using transformer models  
4. Semantic similarity search (cosine similarity)  
5. Answer generation:
   - Transformer-based output
   - LLM-based output (Groq – LLaMA 3)

---



