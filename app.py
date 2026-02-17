import os
import shutil
import atexit
import logging
import warnings
import streamlit as st
from utils.loader import load_and_split_pdf
from utils.embeddings import create_or_load_vectorstore
from langchain_ollama import OllamaLLM



# Silence Python warnings
warnings.filterwarnings("ignore")

# Silence Torch internal logs
logging.getLogger("torch").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)

# Optional: reduce transformers verbosity
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

# =========================================================
# CLEANUP FUNCTION
# =========================================================

def cleanup_data():
    if os.path.exists("vectorstore"):
        shutil.rmtree("vectorstore", ignore_errors=True)
    if os.path.exists("data"):
        shutil.rmtree("data", ignore_errors=True)


if "app_initialized" not in st.session_state:
    cleanup_data()
    os.makedirs("vectorstore", exist_ok=True)
    os.makedirs("data", exist_ok=True)
    st.session_state.app_initialized = True

atexit.register(cleanup_data)


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="SmartDoc AI",
    layout="wide",
    page_icon="📄"
)

# =========================================================
# PROFESSIONAL UI 
# =========================================================

st.markdown("""
<style>

/* Remove Streamlit center constraint */
.block-container {
    padding-top: 2rem;
    padding-left: 4rem;
    padding-right: 4rem;
    max-width: 100%;
}

/* Background */
.stApp {
    background: linear-gradient(180deg, #0E1117 0%, #0B0F16 100%);
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #0F172A;
}

/* Chat bubbles */
div[data-testid="stChatMessage"] {
    border-radius: 18px;
    padding: 14px 20px;
    margin-bottom: 14px;
}

/* Chat input */
div[data-testid="stChatInput"] {
    border-radius: 18px;
}

/* Buttons */
.stButton>button {
    border-radius: 8px;
}

/* Divider */
hr {
    border: 0.5px solid #1F2937;
}
            
/* Remove Streamlit top header glow */
header[data-testid="stHeader"] {
    display: none;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# SESSION STATE
# =========================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "current_doc" not in st.session_state:
    st.session_state.current_doc = None


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:
    st.markdown("## 📄 SmartDoc AI")
    st.caption("Local RAG Knowledge Assistant")

    st.divider()

    uploaded_file = st.file_uploader("Upload PDF", type="pdf")

    if uploaded_file:
        st.success(f"Loaded: {uploaded_file.name}")

    st.divider()

    if st.button("🗑 Clear Chat"):
        st.session_state.messages = []
        st.rerun()


# =========================================================
# HEADER 
# =========================================================

st.markdown("""
<div style="margin-bottom:30px;">
    <h1 style="
        font-size:36px;
        margin-bottom:6px;
        font-weight:700;
        letter-spacing:-0.5px;
    ">
        SmartDoc AI
    </h1>
    <p style="
        color:#9CA3AF;
        font-size:15px;
        margin-top:0;
    ">
        Intelligent document assistant powered by local RAG
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)


# =========================================================
# DOCUMENT PROCESSING
# =========================================================

if uploaded_file is not None:

    if st.session_state.current_doc != uploaded_file.name:

        save_path = os.path.join("data", uploaded_file.name)

        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        if os.path.exists("vectorstore"):
            shutil.rmtree("vectorstore")

        with st.spinner("Processing document..."):
            chunks = load_and_split_pdf(save_path)
            vectorstore = create_or_load_vectorstore(chunks)

        st.session_state.vectorstore = vectorstore
        st.session_state.current_doc = uploaded_file.name
        st.session_state.messages = []

        st.toast(
            f"Document '{uploaded_file.name}' ready!",
            icon="✅"
        )


# =========================================================
# CHAT SECTION 
# =========================================================

if st.session_state.vectorstore is not None:

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    prompt = st.chat_input("Ask something about the document...")

    if prompt:

        st.session_state.messages.append(
            {"role": "user", "content": prompt}
        )

        with st.chat_message("user"):
            st.markdown(prompt)

        assistant_placeholder = st.empty()

        with assistant_placeholder.container():
            with st.chat_message("assistant"):
                with st.spinner("🤖 Thinking..."):

                    llm_classifier = OllamaLLM(
                        model="mistral",
                        temperature=0,
                        num_predict=10
                    )

                    classifier_prompt = f"""
Classify the user message into ONE of these categories:

GREETING
DOCUMENT
GENERAL

Respond with ONLY one word.

Message: {prompt}
"""

                    intent = llm_classifier.invoke(classifier_prompt).strip().upper()

                    if "GREETING" in intent:
                        response = "Hello! 👋 How can I help you with your document today?"

                    elif "GENERAL" in intent:
                        llm_general = OllamaLLM(
                            model="mistral",
                            temperature=0.2,
                            num_predict=1000
                        )
                        response = llm_general.invoke(prompt)

                    else:
                        results = st.session_state.vectorstore.similarity_search(
                            prompt,
                            k=5
                        )

                        context = "\n\n".join(
                            [doc.page_content for doc in results]
                        )

                        llm_rag = OllamaLLM(
                            model="mistral",
                            temperature=0.1,
                            num_predict=5000
                        )

                        full_prompt = f"""
Use ONLY the context below.
If answer not found, say "Not found in the document."

Context:
{context}

Question:
{prompt}

Answer:
"""

                        response = llm_rag.invoke(full_prompt)

                st.markdown(response)

        st.session_state.messages.append(
            {"role": "assistant", "content": response}
        )
