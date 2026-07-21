# 📄 Document-QA-System

An AI-powered **Document Question Answering System** built using **Retrieval-Augmented Generation (RAG)**.  
This application allows users to upload PDF documents, process their content, and ask questions to get accurate AI-generated answers based on the document context.

The system uses **LangChain, FAISS Vector Database, Embeddings, and Large Language Models (LLMs)** to provide intelligent document understanding and retrieval.

---

## 🚀 Features

✅ Upload PDF documents  
✅ Extract text from PDF files  
✅ Intelligent text chunking  
✅ Generate vector embeddings  
✅ Store embeddings using FAISS Vector Database  
✅ Retrieve relevant document information  
✅ AI-powered question answering  
✅ Context-aware responses using RAG pipeline  
✅ User-friendly Streamlit interface  

---

## 🏗️ Project Architecture

User
|
| Upload PDF
↓
PDF Loader
|
↓
Text Extraction
|
↓
Text Chunking
|
↓
Embedding Generation
|
↓
FAISS Vector Database
|
↓
Retriever
|
↓
LLM Model
|
↓
AI Generated Answer


---

## 🛠️ Tech Stack

### Programming Language
- Python

### Framework
- Streamlit

### AI / ML Libraries
- LangChain
- FAISS
- Google Gemini API
- Sentence Transformers

### Other Tools
- PyPDF
- Python-dotenv
- Vector Database

---

## 📂 Project Structure

Document-QA-System
│
├── app.py # Main Streamlit application
├── requirements.txt # Required dependencies
├── README.md # Project documentation
├── .env.example # Environment variable example
├── test_api.py # API testing file
│
├── utils
│ ├── embeddings.py # Embedding generation
│ ├── llm.py # LLM configuration
│ ├── pdf_loader.py # PDF processing
│ ├── text_splitter.py # Text chunking logic
│ └── helper.py # Helper functions
│
├── pdfs # Uploaded PDF files
└── vectorstore # FAISS index storage

