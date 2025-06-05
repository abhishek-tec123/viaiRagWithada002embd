import os
from vectorstore_utils import unzip_to_memory
import logging
from typing import Dict, Any
from context import generate_response_from_llm
from pymongo.errors import ConnectionFailure
from fastapi import HTTPException
from dotenv import load_dotenv
import asyncio
from functools import lru_cache
import time
from motor.motor_asyncio import AsyncIOMotorClient
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from langchain_community.vectorstores import FAISS
from embedding_model import get_model
# Load environment variables from .env
load_dotenv()

# -------------------------
# Configuration
# -------------------------
MONGO_URI = os.getenv("MONGODB_URI")
DB_NAME = os.getenv("DB_NAME")
COLLECTION_NAME = os.getenv("COLLECTION_NAME")
VECTOR_STORE_BASE_DIR = os.getenv("VECTOR_STORE_BASE_DIR", "./vectorstores")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
model_name = os.getenv("GROQ_MODEL_NAME")

# -------------------------
# Logging
# -------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("VectorPipeline")

# -------------------------
# Cache Configuration
# -------------------------
VECTOR_STORE_CACHE = {}
VECTOR_STORE_CACHE_TTL = 3600  # 1 hour in seconds
EMBEDDINGS_CACHE = None

# -------------------------
# MongoDB Connection Pool
# -------------------------
class MongoDBConnectionPool:
    _instance = None
    _client = None

    @classmethod
    async def get_client(cls):
        if cls._instance is None:
            cls._instance = cls()
            cls._client = AsyncIOMotorClient(MONGO_URI)
        return cls._client

    @classmethod
    async def close(cls):
        if cls._client:
            cls._client.close()
            cls._client = None
            cls._instance = None

# -------------------------
# Vector Store Cache
# -------------------------
class VectorStoreCache:
    @staticmethod
    def get_cache_key(user_id: str, folder_id: str) -> str:
        return f"{user_id}:{folder_id}"

    @staticmethod
    def get_vector_store(user_id: str, folder_id: str):
        cache_key = VectorStoreCache.get_cache_key(user_id, folder_id)
        if cache_key in VECTOR_STORE_CACHE:
            cache_entry = VECTOR_STORE_CACHE[cache_key]
            if time.time() - cache_entry['timestamp'] < VECTOR_STORE_CACHE_TTL:
                return cache_entry['retriever'], cache_entry['path']
        return None, None

    @staticmethod
    def set_vector_store(user_id: str, folder_id: str, retriever, path: str):
        cache_key = VectorStoreCache.get_cache_key(user_id, folder_id)
        VECTOR_STORE_CACHE[cache_key] = {
            'retriever': retriever,
            'path': path,
            'timestamp': time.time()
        }

# -------------------------
# Embeddings Cache
# -------------------------
def get_embeddings():
    global EMBEDDINGS_CACHE
    if EMBEDDINGS_CACHE is None:
        EMBEDDINGS_CACHE = get_model()
    return EMBEDDINGS_CACHE

# -------------------------
# MongoDB Fetch
# -------------------------
async def get_vector_store_by_user_and_folder(user_id: str, folder_id: str) -> Dict[str, Any]:
    log.info(f"Fetching vector store for user_id={user_id}, folder_id={folder_id}")
    try:
        client = await MongoDBConnectionPool.get_client()
        db = client[DB_NAME]
        collection = db[COLLECTION_NAME]
        document = await collection.find_one({"user_id": user_id, "folder_id": folder_id})
        if not document:
            raise ValueError("Vector store not found in MongoDB.")
        return document
    except ConnectionFailure as e:
        log.error(f"MongoDB connection error: {e}")
        raise HTTPException(status_code=500, detail="Database connection error")

# -------------------------
# Download and Save
# -------------------------
async def download_and_save_vector_store_by_folder(user_id: str, folder_id: str) -> tuple:
    try:
        # Check cache first
        cached_retriever, cached_path = VectorStoreCache.get_vector_store(user_id, folder_id)
        if cached_retriever and cached_path:
            log.info(f"Using cached vector store for user_id={user_id}, folder_id={folder_id}")
            return cached_retriever, cached_path

        # If not in cache, download from MongoDB
        log.info(f"Cache miss - downloading vector store for user_id={user_id}, folder_id={folder_id}")
        document = await get_vector_store_by_user_and_folder(user_id, folder_id)
        zip_data = document.get("vector_store")
        if not zip_data:
            raise ValueError("No zip data found in document")

        metadata = document.get("metadata", {})
        vectorstore_info = metadata.get("vectorstore_info", {})
        persist_dir = vectorstore_info.get("persist_dir", folder_id)
        extract_path = os.path.join(VECTOR_STORE_BASE_DIR, user_id, persist_dir)

        saved_path = await asyncio.to_thread(unzip_to_memory, zip_data, extract_path)
        
        # Load vector store and cache it
        embeddings = get_embeddings()
        vectorstore = FAISS.load_local(saved_path, embeddings=embeddings, allow_dangerous_deserialization=True)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        VectorStoreCache.set_vector_store(user_id, folder_id, retriever, saved_path)
        
        log.info(f"Successfully downloaded and cached vector store for user_id={user_id}, folder_id={folder_id}")
        return retriever, saved_path
    except Exception as e:
        log.error(f"Download error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# -------------------------
# Load Vector Store
# -------------------------
def load_vector_store(vector_store_path: str):
    log.info(f"Loading FAISS vector store from: {vector_store_path}")
    embeddings = get_embeddings()
    vectorstore = FAISS.load_local(vector_store_path, embeddings=embeddings, allow_dangerous_deserialization=True)
    return vectorstore.as_retriever(search_kwargs={"k": 3})

# -------------------------
# Combined Pipeline
# -------------------------
async def pipeline_query_with_llm(retriever, query: str, user_id: str, folder_id: str, llm_provider: str = "groq") -> Dict[str, Any]:
    log.info(f"Running query: '{query}'")
    docs = await asyncio.to_thread(retriever.invoke, query)
    if not docs:
        return {
            "status": "error",
            "message": "No relevant documents found for your query.",
            "data": None,
            "database_info": {
                "database_name": DB_NAME,
                "collection_name": COLLECTION_NAME,
                "user_id": user_id,
                "folder_id": folder_id
            }
        }

    combined_text = "\n\n".join(doc.page_content for doc in docs)
    log.info(f"Retrieved {len(docs)} documents for query processing")
    response = await generate_response_from_llm(input_text=combined_text, query=query, llm_provider=llm_provider)
    
    return {
        "status": "success",
        "message": "Query processed successfully",
        "data": {
            "query": query,
            "response": response,
        },
        "database_info": {
            "database_name": DB_NAME,
            "collection_name": COLLECTION_NAME,
            "user_id": user_id,
            "folder_id": folder_id
        }
    }

# -------------------------
# Batch Processing
# -------------------------
async def batch_process_queries(retriever, queries: list[str], user_id: str, folder_id: str, llm_provider: str = "groq") -> list[Dict[str, Any]]:
    tasks = [pipeline_query_with_llm(retriever, query, user_id, folder_id, llm_provider) for query in queries]
    return await asyncio.gather(*tasks)

# # -------------------------
# # Main Execution
# # -------------------------
# async def main():
#     user_id = "12345"
#     folder_id = "my1"
#     query = "write key points about this document"

#     try:
#         vector_store_path = await download_and_save_vector_store_by_folder(user_id, folder_id)
#         retriever = load_vector_store(vector_store_path)
#         result = await pipeline_query_with_groq(retriever, query, user_id, folder_id)

#         print("\n=== Final Structured Response ===\n")
#         print(json.dumps(result, indent=2))

#     except Exception as e:
#         log.error(f"Pipeline failed: {e}")
#         error_response = {
#             "status": "error",
#             "message": str(e),
#             "data": None,
#             "database_info": {
#                 "database_name": DB_NAME,
#                 "collection_name": COLLECTION_NAME,
#                 "user_id": user_id,
#                 "folder_id": folder_id
#             }
#         }
#         print(json.dumps(error_response, indent=2))
#     finally:
#         await MongoDBConnectionPool.close()

# if __name__ == "__main__":
#     asyncio.run(main())
