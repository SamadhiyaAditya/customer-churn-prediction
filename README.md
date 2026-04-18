# ChurnGuard AI — Customer Churn Prediction & Agentic Retention Strategy

> **From Predictive Analytics to Intelligent Intervention**

An end-to-end AI system that predicts customer churn using classical ML and autonomously generates personalized retention strategies through a RAG-powered agentic workflow.

[![Live Demo](https://img.shields.io/badge/Live_Demo-Hugging_Face_Spaces-blue?style=for-the-badge)](https://offxkavya-customer-churn-prediction.hf.space)
[![Drive](https://img.shields.io/badge/Project_Drive-Google_Drive-green?style=for-the-badge)](https://drive.google.com/drive/folders/12fRWPl85QFgIlwq_fCYZHf4c7m4g3Y7J?usp=sharing)

---

## Quick Links

| Resource | Link |
|----------|------|
| **Live Application** | [offxkavya-customer-churn-prediction.hf.space](https://offxkavya-customer-churn-prediction.hf.space) |
| **Project Drive** | [Google Drive](https://drive.google.com/drive/folders/12fRWPl85QFgIlwq_fCYZHf4c7m4g3Y7J?usp=sharing) |

---

## Table of Contents

- [Project Overview](#project-overview)
- [Key Features](#key-features)
- [System Architecture](#system-architecture)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Setup Instructions](#setup-instructions)
- [Milestone 1 — Classical ML Baseline](#milestone-1--classical-ml-baseline)
- [Milestone 2 — Agentic AI & RAG System](#milestone-2--agentic-ai--rag-system)
- [Deployment](#deployment)
- [Team](#team)

---

## Project Overview

**ChurnGuard AI** is an intelligent customer retention platform built across two milestones:

- **Milestone 1** — A classical ML pipeline (Logistic Regression + Decision Tree) that predicts churn probability from 10 customer features
- **Milestone 2** — A fully agentic AI system that autonomously analyzes risk, retrieves domain-specific retention strategies via RAG, generates personalized action plans through LLM reasoning, and compiles structured reports — all without human intervention

The system is deployed live on **Hugging Face Spaces** as a professional, dark-themed Streamlit application.

---

## Key Features

| Feature | Description |
|---------|-------------|
| **ML Churn Prediction** | Logistic Regression pipeline predicts churn probability from 10 customer features |
| **RAG Pipeline** | FAISS vector store + HuggingFace embeddings retrieve relevant retention strategies from a curated knowledge base |
| **LLM Reasoning** | Groq API (LLaMA 3.3 70B) generates 3 specific, prioritized retention actions per customer |
| **Agentic Workflow** | 5-node LangGraph agent orchestrates the full analysis → retrieval → reasoning → report pipeline |
| **Structured Output** | Strict JSON reports with risk level, actions, outcomes, priority score, and escalation flags |
| **Ethical AI** | Dedicated disclaimer node appended to every agent run; responsible AI compliance built in |
| **Interactive Dashboard** | 3-tab Streamlit UI with Plotly visualizations (gauge chart, radar chart, bar charts) |
| **Live Deployment** | Hosted on Hugging Face Spaces with GROQ_API_KEY managed via secrets |

---

## System Architecture

```
Input (Customer Features via Sidebar)
    │
    ▼
ML Prediction ──► churn_model.pkl (Logistic Regression Pipeline)
    │
    ▼
Risk Factor Extraction ──► generate_reason()
    │
    ▼
RAG Retrieval ──► rag/retriever.py ──► FAISS Index (all-MiniLM-L6-v2)
    │
    ▼
LangGraph Agent (5 nodes) ──► agent/graph.py
    │
    ├── Node 1: analyze_risk       → Risk level classification
    ├── Node 2: retrieve_rag       → FAISS semantic search (top-3)
    ├── Node 3: generate_recs      → Groq LLM retention strategies
    ├── Node 4: structured_report  → JSON report generation
    └── Node 5: add_disclaimer     → Ethical AI compliance
    │
    ▼
Streamlit UI ──► ChurnGuard AI Dashboard (Hugging Face Spaces)
```

---

## Tech Stack

| Category | Technology | Role |
|----------|-----------|------|
| **LLM Provider** | Groq API (llama-3.3-70b-versatile) | Fast LLM inference for retention strategy generation |
| **Agentic Framework** | LangGraph | Stateful 5-node agent workflow orchestration |
| **Orchestration** | LangChain | Prompt templates, text splitters, document loaders |
| **Embeddings** | HuggingFace all-MiniLM-L6-v2 | 384-dim semantic embeddings for strategy retrieval |
| **Vector Store** | FAISS (langchain_community) | In-memory approximate nearest-neighbor search |
| **ML Pipeline** | scikit-learn 1.6.1 | ColumnTransformer + LogisticRegression pipeline |
| **Data Handling** | pandas, NumPy | Data manipulation and feature engineering |
| **Web Interface** | Streamlit | Interactive frontend with tabs, sidebar, CSS theming |
| **Visualization** | Plotly | Gauge chart, radar chart, bar charts |
| **Deployment** | Hugging Face Spaces | Cloud hosting for the live Streamlit application |
| **Training Environment** | Google Colab | Notebook-based model training and evaluation |
| **Serialization** | joblib | Model saving and loading |

---

## Project Structure

```
customer-churn-prediction/
├── app.py                          # Main Streamlit app — ChurnGuard AI
├── app_old_milestone1.py           # Milestone 1 Streamlit app (archived)
├── churn_model.pkl                 # Serialized Logistic Regression pipeline
├── requirements.txt                # Python dependencies
│
├── agent/                          # Agentic workflow module
│   ├── __init__.py                 # Exports AgentState, build_agent_graph
│   ├── graph.py                    # LangGraph 5-node workflow construction
│   ├── nodes.py                    # Node implementations (Groq API, risk analysis)
│   ├── prompts.py                  # System/user prompt templates
│   └── state.py                    # AgentState TypedDict schema
│
├── rag/                            # RAG pipeline module
│   ├── __init__.py                 # Exports build_rag_index
│   ├── ingest.py                   # Knowledge base, text splitting, FAISS indexing
│   └── retriever.py                # FAISS similarity search wrapper
│
├── Data/
│   └── customer_churn_dataset-testing-master.csv   # 64,374 rows
│
├── Model/
│   └── churn_model.pkl             # Serialized model (archive copy)
│
├── Notebook/
│   └── GenAi-Capstone.ipynb        # Training & evaluation notebook
│
├── AssetsMilestone1/               # Milestone 1 screenshots & diagrams
├── Milestone1ProjectReport.tex     # Milestone 1 LaTeX report
├── Milestone2ProjectReport.tex     # Milestone 2 LaTeX report
└── READMEMilestone1.md             # Milestone 1 README (archived)
```

---

## Setup Instructions

### Prerequisites

- **Python 3.12** (or compatible)
- **GROQ_API_KEY** — [Get one free from console.groq.com](https://console.groq.com)

### Local Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/offxkavya/customer-churn-prediction.git
   cd customer-churn-prediction
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set your Groq API key:**
   ```bash
   export GROQ_API_KEY="your-api-key-here"
   ```

4. **Run the app:**
   ```bash
   streamlit run app.py
   ```
   The app opens at `http://localhost:8501`.

> **Note:** The app requires `churn_model.pkl` in the project root and a valid `GROQ_API_KEY` for the agentic workflow to function.

### Retraining the Model

1. Open `Notebook/GenAi-Capstone.ipynb` in Google Colab or Jupyter
2. Run all cells — the last cell exports `churn_model.pkl`
3. Copy the exported `.pkl` file to the project root

---

## Milestone 1 — Classical ML Baseline

Milestone 1 built the predictive foundation using supervised machine learning.

### Model Performance

| Metric | Logistic Regression | Decision Tree (max_depth=5) |
|--------|--------------------|-----------------------------|
| Accuracy | 83.17% | **95.97%** |
| Precision | 81.63% | — |
| Recall | 83.06% | **98.24%** |
| F1-Score | 82.34% | — |

**Deployed Model:** Logistic Regression — chosen for smoother probability estimates suitable for gauge chart visualization.

### Top Risk Factors (Decision Tree Feature Importance)

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | Payment Delay | 47.87% |
| 2 | Support Calls | 14.40% |
| 3 | Tenure | 9.91% |
| 4 | Usage Frequency | 9.10% |
| 5 | Gender (Female) | 8.28% |

---

## Milestone 2 — Agentic AI & RAG System

Milestone 2 transforms the system from a passive predictor into an **autonomous retention strategist**.

### Evolution from Milestone 1

| Capability | Milestone 1 | Milestone 2 |
|------------|-------------|-------------|
| **Risk Prediction** | Logistic Regression / Decision Tree | ML model + LLM reasoning via Groq API |
| **Contextual Understanding** | None — tabular features only | RAG with FAISS + HuggingFace embeddings |
| **Explainability** | Churn probability only | Natural language strategy per risk factor |
| **Autonomy** | None — single prediction step | 5-node agentic loop via LangGraph |
| **Output** | Probability + gauge chart | Structured JSON retention report + action plan |

### RAG Pipeline

- **Knowledge Base:** 8 curated retention strategy domains (payment delay, support calls, tenure, churn risk, contracts, usage, ethics)
- **Embeddings:** HuggingFace `all-MiniLM-L6-v2` (384-dimensional)
- **Vector Store:** FAISS in-memory index with `CharacterTextSplitter` (chunk_size=300, overlap=50)
- **Retrieval:** Top-3 semantically similar strategy chunks per customer query

### Agentic Workflow (LangGraph)

| Node | Function | Description |
|------|----------|-------------|
| **01** | `analyze_risk_node` | Classifies risk as HIGH/MODERATE/LOW; builds structured risk summary |
| **02** | `retrieve_rag_node` | FAISS semantic search; retrieves top-3 strategy chunks |
| **03** | `generate_recommendations_node` | Groq LLM generates 3 prioritized retention actions (temp=0.5) |
| **04** | `generate_structured_report_node` | Groq LLM produces strict JSON report (temp=0.2) |
| **05** | `add_disclaimer_node` | Appends ethical AI compliance disclaimer |

### LLM Configuration

| Parameter | Value |
|-----------|-------|
| Provider | Groq Cloud API |
| Model | llama-3.3-70b-versatile |
| Temperature (Strategy) | 0.5 |
| Temperature (Report) | 0.2 |

---

## Deployment

| Parameter | Details |
|-----------|---------|
| **Platform** | Hugging Face Spaces |
| **Live URL** | [offxkavya-customer-churn-prediction.hf.space](https://offxkavya-customer-churn-prediction.hf.space) |
| **SDK** | Streamlit (auto-detected from requirements.txt) |
| **Secrets** | GROQ_API_KEY stored as HF Space secret |
| **Model File** | churn_model.pkl committed to repo root |
| **Stability** | Groq API error handling with graceful JSON fallback |

---

## Team

| Name | Roll Number |
|------|-------------|
| Saksham Miglani | 2401010401 |
| Kavya Mukhija | 2401010219 |
| Pratyush Parida | 2401010351 |
| Aditya Samadhiya | 2401010037 |

**Course:** GenAI & Agentic AI

---

<p align="center">
  <b>ChurnGuard AI</b> — Powered by LangGraph · FAISS RAG · Groq LLM
</p>
