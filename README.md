# 📄 Smart PDF Chatbot

A Streamlit web application that lets you upload a PDF and ask questions about its content using local AI.

**Tech Stack:** Streamlit · HuggingFace (all-MiniLM-L6-v2) · FAISS · Ollama (phi)

---

## Prerequisites

| Requirement | Details |
|---|---|
| **Python** | 3.9 or higher |
| **Ollama** | Installed and running locally ([download](https://ollama.com)) |
| **phi model** | Pulled via Ollama (see below) |

---

## Installation Steps

### 1. Clone / navigate to the project folder

```bash
cd "smart pdf"
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS / Linux
```

### 3. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 4. Install & start Ollama

Download Ollama from [https://ollama.com](https://ollama.com) and install it.

Then pull the **phi** model:

```bash
ollama pull phi
```

Make sure Ollama is running before launching the app:

```bash
ollama serve
```

> **Note:** On Windows, Ollama typically runs as a background service after installation.

---

## How to Run the App

```bash
streamlit run main.py
```

The app will open in your browser at **http://localhost:8501**.

---

## How It Works

```
PDF Upload  →  Text Extraction (PyPDF2)
            →  Chunking (500 chars, 100 overlap)
            →  Embeddings (all-MiniLM-L6-v2)
            →  FAISS Vector Store

User Question  →  Similarity Search (top 4 chunks)
               →  LLM Prompt Construction
               →  Ollama (phi) generates answer
               →  Answer displayed in UI
```

---

## Project Structure

```
smart pdf/
├── main.py              # Complete application (single file)
├── requirements.txt     # Python dependencies
└── README.md            # This file
```

---

## Troubleshooting

| Issue | Solution |
|---|---|
| **"Connection refused" from Ollama** | Run `ollama serve` or check that the Ollama service is running |
| **"Model not found"** | Run `ollama pull phi` to download the model |
| **No text extracted from PDF** | The PDF may be scanned/image-based — only text PDFs are supported |
| **Slow first query** | The embedding model downloads on first use (~80 MB); subsequent runs are cached |
