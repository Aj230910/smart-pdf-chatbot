cat > README.md << 'EOF'
#  Smart PDF Chatbot

An AI-powered PDF Question Answering application built using Streamlit, LangChain, FAISS, HuggingFace Embeddings, and Google Gemini API.

Users can upload PDF documents and ask questions in natural language. The chatbot retrieves relevant content from the PDF and generates intelligent answers using Retrieval-Augmented Generation (RAG).

---

#  Features

-  Upload PDF documents
-  Ask questions from uploaded PDFs
-  Semantic search using FAISS vector database
-  AI-powered answers using Gemini API
-  Modern chatbot-style UI
-  Fast document retrieval
-  Clean and responsive Streamlit interface

---

#  Tech Stack

- Python
- Streamlit
- LangChain
- FAISS
- HuggingFace Embeddings
- Google Gemini API
- Sentence Transformers
- PyPDF

---

#  How It Works

1. User uploads a PDF document
2. PDF text is extracted and split into chunks
3. Text chunks are converted into embeddings
4. Embeddings are stored in a FAISS vector database
5. User asks a question
6. Relevant chunks are retrieved using similarity search
7. Gemini API generates the final response

---

#  Installation

## 1️ Clone Repository

\`\`\`bash
git clone https://github.com/Aj230910/smart-pdf-chatbot.git
cd smart-pdf-chatbot
\`\`\`

---

## 2️ Create Virtual Environment

\`\`\`bash
python -m venv venv
\`\`\`

### Activate Environment

#### Windows
\`\`\`bash
venv\Scripts\activate
\`\`\`

#### Mac/Linux
\`\`\`bash
source venv/bin/activate
\`\`\`

---

## 3️ Install Dependencies

\`\`\`bash
pip install -r requirements.txt
\`\`\`

---

#  Setup Gemini API Key

Create a `.env` file:

\`\`\`env
GOOGLE_API_KEY=your_api_key_here
\`\`\`

Or use Streamlit Secrets:

\`\`\`toml
GOOGLE_API_KEY = "your_api_key_here"
\`\`\`

---

#  Run Application

\`\`\`bash
streamlit run main.py
\`\`\`

---

#  Project Structure

\`\`\`bash
smart-pdf-chatbot/
│
├── main.py
├── requirements.txt
├── README.md
├── .gitignore
└── assets/
\`\`\`

---

#  Future Improvements

-  Dark mode toggle
-  Chat history memory
-  Multiple PDF support
-  Voice input
-  Cloud deployment
-  Source highlighting
-  Mobile responsive UI

---

#  Author

Ambrish Jeyan

---

