import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings


def get_embeddings(provider="huggingface", api_key=None):
    """
    Create Embedding Model based on the provider.
    Supported providers: 'huggingface' (local), 'google' (Gemini).
    """
    if provider == "google":
        key = api_key or os.getenv("GOOGLE_API_KEY")
        if not key:
            raise ValueError("GOOGLE_API_KEY must be provided for Google embeddings.")
        return GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=key
        )
    else:  # Default to huggingface
        # This runs locally via sentence-transformers on CPU/GPU
        return HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
