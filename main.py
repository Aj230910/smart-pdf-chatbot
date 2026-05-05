"""
Smart PDF Chatbot
=================
A Streamlit web application that lets users upload PDF files and ask questions
about their content. Uses HuggingFace embeddings, FAISS vector search, and
a local Ollama LLM to generate answers.
"""

import streamlit as st
import os
import tempfile
from typing import List, Optional

# ---------------------------------------------------------------------------
# PDF Processing
# ---------------------------------------------------------------------------

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract all text from a PDF file."""
    from PyPDF2 import PdfReader

    reader = PdfReader(pdf_path)
    text_parts: List[str] = []
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text_parts.append(page_text)
    return "\n".join(text_parts)


def split_text_into_chunks(text: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
    """
    Split text into overlapping chunks for better context retrieval.

    Args:
        text: The full document text.
        chunk_size: Maximum number of characters per chunk.
        overlap: Number of overlapping characters between consecutive chunks.

    Returns:
        A list of text chunks.
    """
    if not text.strip():
        return []

    chunks: List[str] = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk.strip())
        start += chunk_size - overlap
    return chunks


# ---------------------------------------------------------------------------
# Embeddings & Vector Store
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def load_embedding_model():
    """Load and cache the HuggingFace embedding model."""
    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


def build_vector_store(chunks: List[str]):
    """
    Create a FAISS vector store from a list of text chunks.

    Returns:
        A FAISS vector store instance.
    """
    from langchain_community.vectorstores import FAISS

    embeddings = load_embedding_model()
    vector_store = FAISS.from_texts(chunks, embedding=embeddings)
    return vector_store


def similarity_search(vector_store, query: str, k: int = 4) -> List[str]:
    """
    Retrieve the top-k most relevant chunks for a given query.

    Args:
        vector_store: The FAISS vector store.
        query: The user's question.
        k: Number of results to return.

    Returns:
        A list of relevant text chunks.
    """
    results = vector_store.similarity_search(query, k=k)
    return [doc.page_content for doc in results]


# ---------------------------------------------------------------------------
# LLM Answer Generation
# ---------------------------------------------------------------------------

def build_prompt(context: str, question: str) -> str:
    """Build a prompt that instructs the LLM to answer from context."""
    return (
        "You are a helpful assistant. Use ONLY the following context to answer "
        "the question. If the answer is not in the context, say "
        "\"I couldn't find that information in the document.\"\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer:"
    )


def generate_answer(context_chunks: List[str], question: str) -> str:
    """
    Generate an answer using the local Ollama LLM (phi model).

    Args:
        context_chunks: Relevant text chunks from the vector store.
        question: The user's question.

    Returns:
        The generated answer string.
    """
    from langchain_ollama import OllamaLLM

    context = "\n\n".join(context_chunks)
    prompt = build_prompt(context, question)

    llm = OllamaLLM(model="phi")
    response = llm.invoke(prompt)
    return response.strip()


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------

def configure_page():
    """Set up the Streamlit page configuration."""
    st.set_page_config(
        page_title="Smart PDF Chatbot",
        page_icon="📄",
        layout="centered",
        initial_sidebar_state="expanded"
    )


def inject_custom_css():
    """Inject custom CSS for a modern light theme."""
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Caveat:wght@400;500;600;700&display=swap');

        /* ---------- Global ---------- */
        html, body, [class*="css"] {
            font-family: 'Caveat', cursive;
            color: #1e293b;
            font-size: 1.3rem;
            background-color: #fdfbf7;
        }
        
        .stApp {
            background-color: #fdfbf7;
            background-image: radial-gradient(#cbd5e1 1.5px, transparent 1.5px);
            background-size: 25px 25px;
        }

        /* ---------- Header & Logo ---------- */
        .header-container {
            background: #ffffff;
            border: 3px solid #1e293b;
            border-radius: 255px 15px 225px 15px / 15px 225px 15px 255px;
            padding: 2.5rem 2rem;
            margin-top: 1rem;
            margin-bottom: 2rem;
            text-align: center;
            box-shadow: 6px 6px 0px #1e293b;
            animation: fadeInDown 0.5s ease-out;
            transform: rotate(-1deg);
        }
        
        .animated-logo {
            margin: 0 auto 1rem auto;
            display: block;
        }
        
        .animated-logo path {
            stroke-dasharray: 300;
            stroke-dashoffset: 300;
            animation: drawLine 2s ease-in-out forwards;
        }
        
        @keyframes drawLine {
            to { stroke-dashoffset: 0; }
        }
        
        @keyframes fadeInDown {
            from { opacity: 0; transform: translateY(-10px) rotate(-1deg); }
            to { opacity: 1; transform: translateY(0) rotate(-1deg); }
        }

        .header-container h1 {
            color: #0f172a;
            font-size: 3.5rem;
            font-weight: 700;
            margin: 0 0 0.5rem 0;
            letter-spacing: 1px;
            line-height: 1.2;
        }
        .header-container p {
            color: #475569;
            font-size: 1.5rem;
            font-weight: 500;
            margin: 0;
        }

        /* ---------- Cards & Containers ---------- */
        .status-card {
            background: #ffffff;
            border: 3px solid #1e293b;
            border-left: 6px solid #10b981;
            border-radius: 15px 225px 15px 255px / 255px 15px 225px 15px;
            padding: 1rem 1.2rem;
            margin-bottom: 1rem;
            color: #1e293b;
            box-shadow: 4px 4px 0px #1e293b;
            animation: fadeIn 0.4s ease-out;
            font-weight: 600;
            font-size: 1.3rem;
        }
        .status-card.warning {
            border-left-color: #f59e0b;
        }
        
        @keyframes fadeIn {
            from { opacity: 0; }
            to { opacity: 1; }
        }

        /* ---------- Answer Box ---------- */
        .answer-box {
            background: #ffffff;
            border: 3px solid #1e293b;
            border-radius: 255px 15px 225px 15px / 15px 225px 15px 255px;
            padding: 1.5rem;
            margin-top: 1rem;
            line-height: 1.6;
            font-size: 1.4rem;
            color: #1e293b;
            box-shadow: 6px 6px 0px #1e293b;
            animation: slideUp 0.5s ease-out;
        }
        
        @keyframes slideUp {
            from { opacity: 0; transform: translateY(15px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        .answer-box strong {
            color: #4f46e5;
            font-size: 1.2rem;
            letter-spacing: 1px;
            text-transform: uppercase;
            display: block;
            margin-bottom: 0.5rem;
            font-weight: 700;
            text-decoration: underline;
            text-decoration-style: wavy;
        }

        /* ---------- Footer ---------- */
        .footer {
            text-align: center;
            color: #64748b;
            font-size: 1.1rem;
            font-weight: 600;
            margin-top: 3rem;
            padding-top: 1.5rem;
            border-top: 3px dashed #1e293b;
        }

        /* ---------- Streamlit Element Overrides ---------- */
        .stFileUploader > div {
            background: #ffffff !important;
            border: 3px dashed #1e293b !important;
            border-radius: 255px 15px 225px 15px / 15px 225px 15px 255px !important;
            transition: all 0.2s ease;
            box-shadow: 4px 4px 0px #1e293b !important;
        }
        .stFileUploader > div:hover {
            background: #fef08a !important;
        }
        
        div.stButton > button:first-child {
            background: #ffffff;
            color: #1e293b;
            border: 3px solid #1e293b;
            border-radius: 15px 225px 15px 255px / 255px 15px 225px 15px;
            padding: 0.5rem 1.2rem;
            font-weight: 700;
            font-size: 1.2rem;
            transition: all 0.2s ease;
            box-shadow: 4px 4px 0px #1e293b;
            font-family: 'Caveat', cursive !important;
        }
        div.stButton > button:first-child:hover {
            transform: translate(-2px, -2px);
            box-shadow: 6px 6px 0px #1e293b;
            background: #fef08a;
        }
        div.stButton > button:first-child:active {
            transform: translate(2px, 2px);
            box-shadow: 2px 2px 0px #1e293b;
        }
        
        /* Primary button specifically */
        div.stButton > button[kind="primary"] {
            background: #bae6fd;
        }
        div.stButton > button[kind="primary"]:hover {
            background: #7dd3fc;
        }
        
        div.stTextInput > div > div > input {
            background-color: #ffffff !important;
            border: 3px solid #1e293b !important;
            color: #1e293b !important;
            border-radius: 255px 15px 225px 15px / 15px 225px 15px 255px !important;
            padding: 0.7rem 1rem !important;
            font-size: 1.3rem !important;
            box-shadow: 4px 4px 0px #1e293b !important;
            transition: all 0.2s ease;
            font-family: 'Caveat', cursive !important;
        }
        div.stTextInput > div > div > input:focus {
            box-shadow: 6px 6px 0px #1e293b !important;
            background-color: #fef08a !important;
        }
        
        /* Expander */
        .streamlit-expanderHeader {
            background-color: #ffffff !important;
            border-radius: 15px 225px 15px 255px / 255px 15px 225px 15px !important;
            color: #1e293b !important;
            border: 3px solid #1e293b !important;
            font-weight: 700 !important;
            font-size: 1.2rem !important;
            font-family: 'Caveat', cursive !important;
        }
        
        /* Sidebar */
        [data-testid="stSidebar"] {
            background-color: #ffffff;
            border-right: 3px solid #1e293b;
        }
        
        [data-testid="stSidebar"] * {
            color: #1e293b;
            font-family: 'Caveat', cursive !important;
        }
        
        [data-testid="stHeader"] {
            background-color: transparent !important;
        }
        
        /* Responsive Mobile Adjustments */
        @media (max-width: 768px) {
            .header-container {
                padding: 1.5rem 1rem;
            }
            .header-container h1 {
                font-size: 2.5rem;
            }
            .header-container p {
                font-size: 1.2rem;
            }
            .answer-box {
                padding: 1.2rem;
                font-size: 1.2rem;
            }
            div.stButton > button:first-child {
                padding: 0.5rem 1rem;
                font-size: 1.1rem;
            }
        }
        
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header():
    """Render the top header banner with animated logo."""
    st.markdown(
        """
        <div class="header-container">
            <svg class="animated-logo" viewBox="0 0 100 100" width="80" height="80">
                <!-- A doodle-style open book / brain hybrid icon -->
                <path d="M 50 80 Q 20 80 10 50 Q 10 20 50 30 Q 90 20 90 50 Q 80 80 50 80 Z" fill="none" stroke="#1e293b" stroke-width="4" stroke-linecap="round" stroke-linejoin="round"/>
                <path d="M 50 30 L 50 80" fill="none" stroke="#1e293b" stroke-width="4" stroke-linecap="round"/>
                <path d="M 30 50 L 40 45 M 70 50 L 60 45" fill="none" stroke="#1e293b" stroke-width="4" stroke-linecap="round"/>
            </svg>
            <h1>Smart PDF Chatbot</h1>
            <p>Intelligence unleashed on your PDF documents</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_footer():
    """Render a subtle footer."""
    st.markdown(
        """
        <div class="footer">
            
        </div>
        """,
        unsafe_allow_html=True,
    )


def process_uploaded_pdf(uploaded_file) -> Optional[object]:
    """
    Handle the uploaded PDF: extract text, chunk it, and build a vector store.

    Returns:
        The FAISS vector store, or None if processing fails.
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    try:
        with st.spinner("Reading PDF..."):
            raw_text = extract_text_from_pdf(tmp_path)

        if not raw_text.strip():
            st.error("Could not extract any text from this PDF. It may be scanned or image-only.")
            return None

        with st.spinner("Splitting text into chunks..."):
            chunks = split_text_into_chunks(raw_text)

        st.markdown(
            f'<div class="status-card">Extracted <strong>{len(chunks)}</strong> text chunks from the PDF.</div>',
            unsafe_allow_html=True,
        )

        with st.spinner("Building vector store (this may take a moment the first time)..."):
            vector_store = build_vector_store(chunks)

        st.markdown(
            '<div class="status-card">Vector store ready — ask your questions below.</div>',
            unsafe_allow_html=True,
        )
        return vector_store

    finally:
        os.unlink(tmp_path)


def handle_question(vector_store, question: str):
    """Run similarity search + LLM and return chunks and answer."""
    with st.spinner("Searching for relevant passages..."):
        relevant_chunks = similarity_search(vector_store, question)

    if not relevant_chunks:
        return None, None

    with st.spinner("Generating answer with Ollama (phi)..."):
        answer = generate_answer(relevant_chunks, question)

    return relevant_chunks, answer


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    configure_page()
    inject_custom_css()
    render_header()



    # ---- File Upload ----
    st.markdown("### Upload your PDF")
    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=["pdf"],
        help="Upload a text-based PDF (scanned/image PDFs are not supported).",
    )

    if uploaded_file is None:
        st.markdown(
            '<div class="status-card warning">Waiting for a PDF upload...</div>',
            unsafe_allow_html=True,
        )
        render_footer()
        return

    # ---- Process PDF (cached per file) ----
    file_id = f"{uploaded_file.name}_{uploaded_file.size}"

    if "current_file_id" not in st.session_state or st.session_state.current_file_id != file_id:
        st.session_state.current_file_id = file_id
        st.session_state.vector_store = None
        st.session_state.pop("current_answer", None)
        st.session_state.pop("relevant_chunks", None)
        st.session_state.pop("current_question", None)

    if st.session_state.vector_store is None:
        vector_store = process_uploaded_pdf(uploaded_file)
        if vector_store is None:
            render_footer()
            return
        st.session_state.vector_store = vector_store
    else:
        st.markdown(
            '<div class="status-card">PDF already processed — ready for questions.</div>',
            unsafe_allow_html=True,
        )

    # ---- Question Input ----
    st.markdown("### Ask a Question")
    question = st.text_input(
        "Type your question about the PDF",
        placeholder="e.g. What is the main topic of this document?",
        label_visibility="collapsed",
    )

    col1, col2, _ = st.columns([1, 1, 3])
    ask_clicked = col1.button("Get Answer", type="primary", use_container_width=True)
    clear_clicked = col2.button("Clear", use_container_width=True)

    if clear_clicked:
        st.session_state.vector_store = None
        st.session_state.current_file_id = None
        st.session_state.pop("current_answer", None)
        st.session_state.pop("relevant_chunks", None)
        st.session_state.pop("current_question", None)
        st.rerun()

    if ask_clicked:
        if not question.strip():
            st.warning("Please enter a question before clicking Get Answer.")
        else:
            try:
                chunks, ans = handle_question(st.session_state.vector_store, question.strip())
                if chunks and ans:
                    st.session_state.relevant_chunks = chunks
                    st.session_state.current_answer = ans
                    st.session_state.current_question = question.strip()
                else:
                    st.warning("No relevant passages found in the document for your question.")
                    st.session_state.pop("current_answer", None)
            except Exception as e:
                st.error(f"An error occurred while generating the answer:\n\n`{e}`")
                st.info(
                    "Make sure Ollama is running (`ollama serve`) and the **phi** model "
                    "is available (`ollama pull phi`)."
                )

    # Display Answer and Export Options
    if st.session_state.get("current_answer"):
        with st.expander("Retrieved Context", expanded=False):
            for i, chunk in enumerate(st.session_state.relevant_chunks, 1):
                st.markdown(f"**Chunk {i}**")
                st.code(chunk, language=None)

        st.markdown(
            f'<div class="answer-box"><strong>Answer</strong><br>{st.session_state.current_answer}</div>',
            unsafe_allow_html=True,
        )

        st.markdown("<br>", unsafe_allow_html=True)
        col_dl, _ = st.columns([1, 3])
        
        export_text = f"Question: {st.session_state.current_question}\n\nAnswer: {st.session_state.current_answer}"
        
        with col_dl:
            st.download_button(
                label="Download Answer",
                data=export_text,
                file_name="chatbot_answer.txt",
                mime="text/plain",
                use_container_width=True
            )

    render_footer()


if __name__ == "__main__":
    main()
