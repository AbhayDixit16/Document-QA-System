import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from utils.pdf_loader import load_pdf
from utils.text_splitter import split_documents
from utils.embeddings import get_embeddings
from utils.llm import get_llm

# Load environment variables
load_dotenv()

# Page Configuration
st.set_page_config(
    page_title="Document Q&A System",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom styling for premium look & feel
st.markdown("""
<style>
    .main {
        background-color: #0e1117;
    }
    .stApp {
        background: radial-gradient(circle at 10% 20%, rgba(14, 20, 36, 1) 0%, rgba(20, 26, 48, 1) 90.1%);
        color: #e0e6ed;
    }
    h1, h2, h3 {
        color: #ffffff !important;
        font-weight: 700;
        text-shadow: 0px 0px 8px rgba(0, 180, 255, 0.4);
    }
    .stCard {
        background-color: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        backdrop-filter: blur(10px);
    }
    .stChatMessage {
        background-color: rgba(255, 255, 255, 0.02) !important;
        border: 1px solid rgba(255, 255, 255, 0.05) !important;
        border-radius: 10px;
        padding: 12px;
        margin: 8px 0;
    }
    .stButton>button {
        background: linear-gradient(135deg, #0072ff 0%, #00c6ff 100%);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0px 5px 15px rgba(0, 198, 255, 0.4);
    }
</style>
""", unsafe_allow_html=True)

# Session State Initialization
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "pdf_processed" not in st.session_state:
    st.session_state.pdf_processed = False
if "current_pdf_name" not in st.session_state:
    st.session_state.current_pdf_name = ""

# App Header
st.title("📄 Premium Document Q&A System")
st.write("Upload a document, process its contents locally or in the cloud, and ask questions interactively using open-source models.")

# Sidebar Configuration
with st.sidebar:
    st.image("https://img.icons8.com/nolan/96/document.png", width=70)
    st.header("⚙️ Configuration & Models")

    # Embedding Configuration
    st.subheader("1. Embeddings")
    embed_provider = st.selectbox(
        "Embedding Provider",
        options=["huggingface", "google"],
        index=0,
        help="Hugging Face (local, free sentence-transformers) or Google Gemini Embeddings."
    )

    # LLM Configuration
    st.subheader("2. Large Language Model")
    llm_provider = st.selectbox(
        "LLM Provider",
        options=["groq", "google", "huggingface", "ollama"],
        index=0,
        help="Select the AI brain for answering questions."
    )

    # Model Names selection based on LLM Provider
    model_name = None
    api_key_input = None

    if llm_provider == "groq":
        model_name = st.selectbox(
            "Groq Model",
            options=["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "mixtral-8x7b-32768"],
            index=0
        )
        api_key_input = st.text_input("Groq API Key", type="password", value=os.getenv("GROQ_API_KEY", ""))

    elif llm_provider == "google":
        model_name = st.selectbox(
            "Gemini Model",
            options=["gemini-2.0-flash", "gemini-1.5-flash", "gemini-1.5-pro"],
            index=0
        )
        api_key_input = st.text_input("Google API Key", type="password", value=os.getenv("GOOGLE_API_KEY", ""))

    elif llm_provider == "huggingface":
        model_name = st.text_input("Hugging Face Model ID", value="meta-llama/Meta-Llama-3-8B-Instruct")
        api_key_input = st.text_input("HF API Token", type="password", value=os.getenv("HUGGINGFACEHUB_API_TOKEN", ""))

    elif llm_provider == "ollama":
        model_name = st.text_input("Ollama Model Name", value="llama3")
        st.info("Ensure Ollama is running locally (`ollama run " + model_name + "`)")

    # PDF Upload
    st.markdown("---")
    st.subheader("📂 Document Ingestion")
    uploaded_file = st.file_uploader(
        "Choose a PDF File",
        type=["pdf"],
        help="Upload the PDF you want to analyze."
    )

    process_button = st.button("Process & Index Document", use_container_width=True)

    if st.button("Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ----------------- Core Processing Logic -----------------

if process_button and uploaded_file:
    with st.spinner("Processing PDF and generating embeddings..."):
        try:
            # 1. Load document
            documents = load_pdf(uploaded_file)

            # 2. Split document into chunks
            chunks = split_documents(documents)

            # 3. Initialize Embeddings
            st.write("Initializing embedding model...")
            embedding_model = get_embeddings(provider=embed_provider, api_key=api_key_input if embed_provider == "google" else None)

            # 4. Ingest and create vector store
            st.write("Ingesting chunks into FAISS vector database...")
            vector_store = FAISS.from_documents(chunks, embedding_model)

            # Save into session state
            st.session_state.vector_store = vector_store
            st.session_state.pdf_processed = True
            st.session_state.current_pdf_name = uploaded_file.name
            st.session_state.messages = [] # Clear chat for the new document

            st.success(f"Successfully processed '{uploaded_file.name}'!")
            st.info(f"Loaded {len(documents)} pages, split into {len(chunks)} chunks.")

        except Exception as e:
            st.error(f"An error occurred during ingestion: {e}")

# ----------------- Chat Interface -----------------

# Display active document information
if st.session_state.pdf_processed:
    st.markdown(f"**Chatting with:** `{st.session_state.current_pdf_name}` | **LLM:** `{llm_provider}` | **Embeddings:** `{embed_provider}`")
else:
    st.info("👈 Please upload a PDF and click **Process & Index Document** in the sidebar to start.")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Query Processing
if question := st.chat_input("Ask a question about the document...", disabled=not st.session_state.pdf_processed):
    # Display user query
    with st.chat_message("user"):
        st.markdown(question)
    st.session_state.messages.append({"role": "user", "content": question})

    # Generate bot response
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        with st.spinner("Thinking..."):
            try:
                # 1. Retrieve context
                vector_store = st.session_state.vector_store
                retrieved_docs = vector_store.similarity_search(question, k=4)

                # Format retrieved chunks
                context = "\n\n".join([f"[Source: Page {doc.metadata.get('page', 0) + 1}]\n{doc.page_content}" for doc in retrieved_docs])

                # 2. Initialize LLM
                llm = get_llm(provider=llm_provider, api_key=api_key_input, model_name=model_name)

                # 3. Create context-aware query prompts
                from langchain_core.messages import SystemMessage, HumanMessage
                system_prompt = (
                    "You are an expert Document Q&A assistant. Use the following context extracted from the uploaded PDF to answer the question.\n"
                    "If you do not know the answer or if the text doesn't provide enough information, state that clearly.\n"
                    "Provide clear, concise, and structured answers, referencing specific page numbers from the context when available.\n\n"
                    f"Context:\n{context}"
                )

                messages = [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=question)
                ]

                # 4. Invoke LLM
                response = llm.invoke(messages)
                bot_response = response.content

                # Render response
                response_placeholder.markdown(bot_response)

                # Save message
                st.session_state.messages.append({"role": "assistant", "content": bot_response})

            except Exception as e:
                response_placeholder.error(f"Error generating answer: {e}")