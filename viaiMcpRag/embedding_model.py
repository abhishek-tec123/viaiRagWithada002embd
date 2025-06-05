from langchain_openai import OpenAIEmbeddings
import logging
import os

# Setup logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# Global variable to store the model
embedding_model = None

def initialize_model():
    global embedding_model
    log.info("Initializing OpenAI Ada-002 embedding model...")
    try:
        embedding_model = OpenAIEmbeddings(
            model="text-embedding-ada-002",
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        log.info("OpenAI Ada-002 embedding model initialized successfully")
        return embedding_model
    except Exception as e:
        log.error(f"Error initializing OpenAI embedding model: {str(e)}")
        raise e

def get_model():
    global embedding_model
    if embedding_model is None:
        return initialize_model()
    return embedding_model 