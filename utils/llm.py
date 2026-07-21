import os


def get_llm(provider="groq", api_key=None, model_name=None):
    """
    Returns an LLM chat model client based on the selected provider.
    Supported providers:
    - 'groq': Using ChatGroq (requires GROQ_API_KEY)
    - 'google': Using ChatGoogleGenerativeAI (requires GOOGLE_API_KEY)
    - 'huggingface': Using ChatHuggingFace wrapper over HuggingFaceEndpoint (requires HUGGINGFACEHUB_API_TOKEN)
    - 'ollama': Using ChatOllama for local offline LLMs (no key required, but Ollama must be running)
    """

    if provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI
        key = api_key or os.getenv("GOOGLE_API_KEY")
        if not key:
            raise ValueError("GOOGLE_API_KEY is not set. Please provide it in the sidebar or your .env file.")

        return ChatGoogleGenerativeAI(
            model=model_name or "gemini-2.0-flash",
            google_api_key=key,
            temperature=0.1
        )

    elif provider == "groq":
        from langchain_groq import ChatGroq
        key = api_key or os.getenv("GROQ_API_KEY")
        if not key:
            raise ValueError("GROQ_API_KEY is not set. Please provide it in the sidebar or your .env file.")

        return ChatGroq(
            model=model_name or "llama-3.3-70b-versatile",
            api_key=key,
            temperature=0.1
        )

    elif provider == "huggingface":
        from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
        token = api_key or os.getenv("HUGGINGFACEHUB_API_TOKEN")
        if not token:
            raise ValueError("HUGGINGFACEHUB_API_TOKEN is not set. Please provide it in the sidebar or your .env file.")

        # Initialize standard HuggingFace endpoint
        llm = HuggingFaceEndpoint(
            repo_id=model_name or "meta-llama/Meta-Llama-3-8B-Instruct",
            huggingfacehub_api_token=token,
            temperature=0.1,
            max_new_tokens=512
        )
        return ChatHuggingFace(llm=llm)

    elif provider == "ollama":
        try:
            from langchain_ollama import ChatOllama
        except ImportError:
            from langchain_community.chat_models import ChatOllama

        return ChatOllama(
            model=model_name or "llama3",
            temperature=0.1
        )

    else:
        raise ValueError(f"Unknown LLM provider: {provider}")
