from langchain_community.document_loaders import PyPDFLoader
import tempfile
import os


def load_pdf(uploaded_file):
    """
    Reads an uploaded PDF and returns its content.
    """

    # Create a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(uploaded_file.read())
        temp_path = temp_file.name

    # Load PDF
    loader = PyPDFLoader(temp_path)
    documents = loader.load()

    # Delete temporary file
    os.remove(temp_path)

    return documents