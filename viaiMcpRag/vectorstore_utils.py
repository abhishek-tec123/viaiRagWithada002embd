# vectorstore_utils.py

import io
import os
import zipfile
from datetime import datetime, timedelta
from typing import Dict, Any
from pymongo import MongoClient
from bson import ObjectId
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
import logging as log
from embedding_model import get_model
import shutil

load_dotenv()

MONGO_URI = os.getenv("MONGODB_URI")
DB_NAME = os.getenv("DB_NAME")
COLLECTION_NAME = os.getenv("COLLECTION_NAME")
VECTOR_STORE_BASE_DIR = os.getenv("VECTOR_STORE_BASE_DIR", "./vectorstores")

# Add these constants at the top with other constants
MAX_VECTOR_STORE_SIZE = 100 * 1024 * 1024  # 100MB per vector store
MAX_USER_STORAGE = 1024 * 1024 * 1024  # 1GB per user
STORAGE_TTL_DAYS = 1  # Vector stores older than this will be archived

class VectorStoreBuilder:
    def __init__(self, file_name: str, folder_id: str, chunk_size=100, chunk_overlap=25):
        print(f"\n[VectorStoreBuilder] Initializing with file: {file_name}, folder_id: {folder_id}")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.folder_id = folder_id
        base_name = os.path.splitext(os.path.basename(file_name))[0]
        self.persist_dir = f"VDB_{base_name}_{folder_id}"
        print("[VectorStoreBuilder] Using OpenAI Ada-002 embedding model")
        log.info("[VectorStoreBuilder] Using OpenAI Ada-002 embedding model")
        # Use the preloaded OpenAI model from embedding_model module
        self.embeddings = get_model()
        self._vectorstore = None

    def split_documents(self, documents):
        print(f"[VectorStoreBuilder] Splitting documents with chunk_size: {self.chunk_size}, overlap: {self.chunk_overlap}")
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        split_docs = text_splitter.split_documents(documents)
        print(f"[VectorStoreBuilder] Split documents into {len(split_docs)} chunks")
        log.info(f"[VectorStoreBuilder] Document split into {len(split_docs)} chunks with size {self.chunk_size} and overlap {self.chunk_overlap}")
        return split_docs

    def create_vectorstore(self, documents):
        print("[VectorStoreBuilder] Creating vector store")
        persist_path = os.path.join(os.getcwd(), self.persist_dir)
        print(f"[VectorStoreBuilder] Persist path: {persist_path}")
        log.info(f"[VectorStoreBuilder] Creating vector store using OpenAI Ada-002 embeddings")
        vectorstore = FAISS.from_documents(
            documents=documents,
            embedding=self.embeddings
        )
        vectorstore.save_local(folder_path=persist_path)
        print("[VectorStoreBuilder] Vector store created and saved successfully")
        log.info("[VectorStoreBuilder] Vector store created and saved successfully")
        self._vectorstore = vectorstore
        return vectorstore

    def get_vectorstore_info(self) -> Dict[str, Any]:
        print("[VectorStoreBuilder] Getting vector store info")
        if not self._vectorstore:
            print("[VectorStoreBuilder] Error: Vector store not initialized")
            return {"error": "Vector store not initialized"}

        info = {
            "persist_dir": self.persist_dir,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "total_documents": len(self._vectorstore.index_to_docstore_id),
            "embedding_model": "OpenAI Ada-002",
            "vectorstore_type": "FAISS"
        }
        print(f"[VectorStoreBuilder] Vector store info: process...")
        return info

    def create_zip_archive(self) -> bytes:
        print("[VectorStoreBuilder] Creating zip archive")
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            persist_path = os.path.join(os.getcwd(), self.persist_dir)
            if not os.path.exists(persist_path):
                print(f"[VectorStoreBuilder] Error: Directory {persist_path} does not exist")
                raise FileNotFoundError(f"Directory {persist_path} does not exist")
            for root, dirs, files in os.walk(persist_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, persist_path)
                    zip_file.write(file_path, arcname)
                    print(f"[VectorStoreBuilder] Added to zip: {arcname}")
        print("[VectorStoreBuilder] Zip archive created successfully")
        return zip_buffer.getvalue()

def check_storage_limits(user_id: str, new_store_size: int) -> bool:
    """Check if adding new vector store would exceed user's storage limit"""
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    collection = db[COLLECTION_NAME]
    
    # Get total storage for user
    user_stores = collection.find({"user_id": user_id})
    total_size = sum(store.get("size", 0) for store in user_stores)
    
    return total_size + new_store_size <= MAX_USER_STORAGE

def cleanup_old_stores():
    """Archive or delete vector stores older than TTL"""
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    collection = db[COLLECTION_NAME]
    
    cutoff_date = datetime.utcnow() - timedelta(days=STORAGE_TTL_DAYS)
    old_stores = collection.find({
        "created_at": {"$lt": cutoff_date},
        "last_accessed": {"$lt": cutoff_date}
    })
    
    for store in old_stores:
        # Delete the vector store files
        user_id = store.get("user_id")
        folder_id = store.get("folder_id")
        metadata = store.get("metadata", {})
        vectorstore_info = metadata.get("vectorstore_info", {})
        persist_dir = vectorstore_info.get("persist_dir", folder_id)
        
        # Delete the vector store directory
        vector_store_path = os.path.join(VECTOR_STORE_BASE_DIR, user_id, persist_dir)
        if os.path.exists(vector_store_path):
            try:
                shutil.rmtree(vector_store_path)
                print(f"[Cleanup] Deleted vector store directory: {vector_store_path}")
            except Exception as e:
                print(f"[Cleanup] Error deleting vector store directory {vector_store_path}: {str(e)}")
        
        # Update MongoDB document status
        collection.update_one(
            {"_id": store["_id"]},
            {"$set": {"status": "deleted"}}
        )
    
    client.close()

def cleanup_vector_store_files():
    """Clean up vector store files that are older than 24 hours"""
    try:
        cutoff_time = datetime.utcnow() - timedelta(hours=24)
        
        # Walk through the vector store base directory
        for root, dirs, files in os.walk(VECTOR_STORE_BASE_DIR):
            for dir_name in dirs:
                dir_path = os.path.join(root, dir_name)
                # Get directory creation time
                creation_time = datetime.fromtimestamp(os.path.getctime(dir_path))
                
                if creation_time < cutoff_time:
                    try:
                        shutil.rmtree(dir_path)
                        print(f"[Cleanup] Deleted old vector store directory: {dir_path}")
                    except Exception as e:
                        print(f"[Cleanup] Error deleting vector store directory {dir_path}: {str(e)}")
    except Exception as e:
        print(f"[Cleanup] Error during vector store cleanup: {str(e)}")

def update_store_access_time(mongo_id: str):
    """Update last accessed time for vector store"""
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    collection = db[COLLECTION_NAME]
    
    collection.update_one(
        {"_id": ObjectId(mongo_id)},
        {"$set": {"last_accessed": datetime.utcnow()}}
    )

def store_in_mongodb(zip_data: bytes, user_id: str, folder_id: str, metadata: Dict[str, Any]) -> str:
    if not check_storage_limits(user_id, len(zip_data)):
        raise ValueError("Storage limit exceeded for user")
        
    print(f"\n[StoreMongoDB] Storing vector store for user_id: {user_id}, folder_id: {folder_id}")
    client = MongoClient(MONGO_URI)
    print("[MONGO_Client]", MONGO_URI)
    db = client[DB_NAME]
    print("[DataBase Name]", DB_NAME)
    collection = db[COLLECTION_NAME]
    print("[COLLECTION_NAME]", COLLECTION_NAME)

    document = {
        "user_id": user_id,
        "folder_id": folder_id,
        "created_at": datetime.utcnow(),
        "last_accessed": datetime.utcnow(),
        "metadata": metadata,
        "vector_store": zip_data,
        "size": len(zip_data),
        "status": "active"
    }

    print("[StoreMongoDB] Updating/inserting document in MongoDB")
    result = collection.update_one(
        {"user_id": user_id, "folder_id": folder_id},
        {"$set": document},
        upsert=True
    )

    if result.upserted_id:
        mongo_id = str(result.upserted_id)
        print(f"[StoreMongoDB] New document created with ID: {mongo_id}")
    else:
        updated_doc = collection.find_one({"user_id": user_id, "folder_id": folder_id})
        mongo_id = str(updated_doc["_id"])
        print(f"[StoreMongoDB] Existing document updated with ID: {mongo_id}")

    client.close()
    return mongo_id

def get_vector_store_from_mongodb(mongo_id: str, user_id: str) -> Dict[str, Any]:
    print(f"\n[GetMongoDB] Fetching vector store with mongo_id: {mongo_id}, user_id: {user_id}")
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    collection = db[COLLECTION_NAME]
    object_id = ObjectId(mongo_id)
    document = collection.find_one({"_id": object_id, "user_id": user_id})
    client.close()
    if not document:
        print("[GetMongoDB] Error: Vector store not found")
        raise ValueError("Vector store not found")
    print("[GetMongoDB] Successfully retrieved vector store")
    return document

def unzip_to_memory(zip_data: bytes, extract_path: str) -> str:
    print(f"\n[UnzipMemory] Extracting to path: {extract_path}")
    if not os.path.exists(extract_path):
        print(f"[UnzipMemory] Creating directory: {extract_path}")
        os.makedirs(extract_path)
    with zipfile.ZipFile(io.BytesIO(zip_data), 'r') as zip_ref:
        print("[UnzipMemory] Extracting files")
        zip_ref.extractall(extract_path)
    print(f"[UnzipMemory] Successfully extracted to: {extract_path}")
    return extract_path
