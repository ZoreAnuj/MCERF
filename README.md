# MCERF - Multimodal Cross-Encoder Retrieval Framework

Modular multimodal retrieval and reasoning framework for question answering on engineering documentation. Combines vision-language models (ColPali) with text retrieval (BM25) for understanding complex engineering drawings, P&IDs, datasheets, and technical specifications.

## Architecture

- **Vision RAG** (`vision_rag_gpt5.py`) - Multimodal pipeline combining ColPali visual document embeddings with GPT-based reasoning for engineering Q&A
- **ColPali Integration** (`colpali.py`) - Late interaction retrieval using ColPali's patch-level visual embeddings for document page ranking
- **Router System** (`Routers/`) - Intelligent query routing between text-only RAG and vision-augmented RAG based on query complexity
- **Evaluation** (`Evaluation/`) - Comprehensive metrics suite (ROUGE, semantic similarity, accuracy) with full evaluation pipeline
- **Fine-Tuning** (`Appendix/GPT-4o-MCERF-FineTuned/`) - Synthetic data generation and model fine-tuning for domain-specific engineering terminology
- **SAM Integration** (`Appendix/Image Segmentation and Attention Refinement Study/`) - Segment Anything Model for region-of-interest extraction from technical drawings

## Stack

Python / PyTorch / ColPali / OpenAI / LangChain / Sentence-Transformers / BM25 / SAM / PDF2Image

## Key Features

- Multimodal document understanding combining text + visual features
- Engineering-specific fine-tuning with synthetic Q&A datasets
- Self-consistency ensemble for improved answer reliability
- Vision-to-text conversion pipeline for legacy document support
