# Modular RAG Assistant

A modular Retrieval-Augmented Generation (RAG) system designed to reflect real-world production pipelines.  
The project combines hybrid retrieval, reranking, and LLM-based generation with full transparency of token usage and cost.

---

## Overview

This system allows users to:

- Query their own documents using natural language  
- Generate answers grounded in retrieved context  
- Create summaries based on document content  
- Inspect retrieved sources and debug the pipeline  
- Track token usage and estimated API costs  
- Switch between local and API-based language models  

The architecture is modular and extensible, following patterns used in modern AI systems.

---


## Demo

<video src="assets/demo.mp4" controls width="800"></video>

### Main application view

![Main application view](assets/main_ui.jpg)

### Cost tracking

![Cost](assets/cost_tracking.jpg)

### Retrieval sources and scoring
![Retrieval sources](assets/retrieval_sources.jpg)

---

## Key Features

### Hybrid Retrieval
- Dense search using FAISS embeddings  
- Sparse search using TF-IDF  
- Score fusion for improved relevance  

### Query Processing
- Basic query normalization  
- Typo correction for improved robustness  

### Reranking
- Combines semantic similarity and keyword overlap  
- Improves precision of top retrieved results  

### Generation
- Supports:
  - OpenAI models (e.g. gpt-4.1-mini, gpt-4o-mini)
  - Local models via Ollama  
- Generates answers strictly based on retrieved context  

### Cost and Token Tracking
- Input and output token tracking  
- Estimated cost per request  
- Aggregated session usage  

### Evaluation
- Retrieval evaluation framework  
- Metrics:
  - Top-1 / Top-3 / Top-5 accuracy  
  - Mean Reciprocal Rank (MRR)  
  - Recall@k  
- Comparison of retrieval modes:
  - dense vs sparse vs hybrid  

### User Interface
- Streamlit-based interface  
- Chat mode and summary mode  
- Adjustable retrieval parameters  
- Source inspection for each answer  

---

## Architecture

Pre-Retrieval  
↓  
Query Processing (normalization + typo correction)  
↓  
Hybrid Retrieval (FAISS + TF-IDF)  
↓  
Reranking and Filtering  
↓  
Context Construction  
↓  
Generation (LLM)  
↓  
Pipeline (chat / summary)  

---

## Project Structure

rag/  
├── indexing/  
├── retrieval/  
├── pre_retrieval/  
├── post_retrieval/  
├── generation/  
├── orchestration/  
├── utils/  
├── config.py  

evaluation/  
├── test_cases.py  
├── evaluate.py  

app.py  

---

## Running the Project

1. Install dependencies  
pip install -r requirements.txt  

2. Configure environment  
cp .env.example .env  

Add your OpenAI API key:  
OPENAI_API_KEY=your_key_here  

3. (Optional) Run local models with Ollama  
ollama pull llama3.1:8b  
ollama pull nomic-embed-text  

4. Build the knowledge base  
python -m rag.indexing.builder  

5. Run the application  
streamlit run app.py  

---

## Retrieval evaluation

python -m evaluation.evaluate

The retrieval component was evaluated using standard information retrieval metrics.

| Mode   | Top-1 | Top-3 | Top-5 | MRR  | Recall@5 |
|--------|------|------|------|------|----------|
| Dense  | 0.90 | 0.90 | 1.00 | 0.92 | 1.00     |
| Sparse | 0.80 | 0.90 | 1.00 | 0.85 | 1.00     |
| Hybrid | 0.90 | 1.00 | 1.00 | 0.95 | 1.00     |

Hybrid retrieval achieves the best ranking performance (MRR),
while all methods reach full recall at top-5.

---

## Technology Stack

Python  
FAISS  
Scikit-learn  
OpenAI API  
Ollama  
Streamlit  

---

## Author

Bartłomiej Jamiołkowski
