# 🧠 RAG Intelligence Dashboard

A **Retrieval-Augmented Generation (RAG)** system with a built-in **Confidence Calibration Layer**, served through a professional Streamlit analytics dashboard.

---

## ✨ Features

- 📄 **Document ingestion** — supports `.txt` and `.pdf` files
- 🔍 **Semantic retrieval** via FAISS + `all-MiniLM-L6-v2` embeddings
- 🤖 **Answer generation** using NVIDIA LLM (or a safe fallback)
- 📊 **Confidence scoring** — multi-factor calibration with:
  - Base cosine similarity score
  - Variance detection & penalty
  - Answer ↔ context cross-check
  - Context length bonus
- ⛔ **Fallback mode** when retrieval confidence is too low
- 🖥️ **Streamlit dashboard** with animated progress bar, Plotly chart, and color-coded badges

---

## 📁 Project Structure

```
Confidence_meter/
├── app.py          # Streamlit dashboard (UI)
├── rr.py           # RAGWithConfidence core class
├── .env            # Environment variables (API keys)
└── docs/           # Place your .txt / .pdf documents here
```

---

## ⚙️ Prerequisites

- Python 3.9+
- pip

---

## 🚀 Setup & Installation

### 1. Clone / navigate to the project folder

```powershell
cd "c:\Users\agokh\OneDrive\Documents\Confidence_meter"
```

### 2. (Optional) Create a virtual environment

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
pip install streamlit plotly sentence-transformers faiss-cpu pypdf numpy
```

> If you want NVIDIA LLM support, also run:
> ```powershell
> pip install langchain-nvidia-ai-endpoints
> ```

### 4. Add your documents

Drop `.txt` or `.pdf` files into the `docs/` folder:

```
docs/
├── document1.pdf
├── notes.txt
└── ...
```

### 5. (Optional) Set your API key

Edit `.env` or set the environment variable for NVIDIA LLM:

```powershell
$env:OPENAI_API_KEY = "your-nvidia-api-key-here"
```

---

## ▶️ Running the Dashboard

```powershell
python -m streamlit run app.py
```

Then open your browser at:

```
http://localhost:8501
```

> **Note:** Use `python -m streamlit` instead of `streamlit` directly if the `streamlit` command is not found in your PATH.

---

## 🖥️ Dashboard Overview

| Panel | Description |
|---|---|
| **Query Input** | Enter any question about your documents |
| **Answer Card** | Generated answer from retrieved context |
| **Confidence Badge** | 🟢 High / 🟡 Medium / 🔴 Low |
| **Animated Progress Bar** | Visual confidence score (0–1) |
| **Similarity Chart** | Plotly bar chart — per-document cosine scores |
| **Sources Panel** | Expandable list of retrieved documents |
| **Score Breakdown** | Base score, variance, penalty, chunk count |
| **Fallback Banner** | Shown when retrieval confidence is too low |
| **Sidebar** | Configuration + explainers for each scoring component |

---

## 🧮 Confidence Calibration Logic

```
confidence = base_score − variance_penalty − mismatch_penalty + context_bonus
```

| Component | Details |
|---|---|
| `base_score` | Mean cosine similarity of top-k retrieved chunks |
| `variance_penalty` | `(max − min) × 0.2` — penalises inconsistent retrieval |
| `mismatch_penalty` | `0.2` if answer ↔ context similarity < 0.30 |
| `context_bonus` | Up to `+0.1` for large context (> 100 words) |
| **Fallback trigger** | `base_score < 0.25` → returns safe refusal answer |

---

## 🏷️ Confidence Labels

| Label | Score Range | Meaning |
|---|---|---|
| 🟢 **High** | > 0.75 | Reliable, well-supported answer |
| 🟡 **Medium** | 0.50 – 0.75 | Partially supported, review sources |
| 🔴 **Low** | < 0.50 | Low reliability, treat with caution |

---

## 📦 Key Dependencies

| Package | Purpose |
|---|---|
| `streamlit` | Web dashboard framework |
| `plotly` | Interactive similarity bar chart |
| `sentence-transformers` | Text embedding model (`all-MiniLM-L6-v2`) |
| `faiss-cpu` | Fast vector similarity search |
| `pypdf` | PDF text extraction |
| `numpy` | Numerical computations |

---

## 📝 License

MIT — free to use and modify.
