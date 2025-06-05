from fastapi import FastAPI, UploadFile, File, Form, HTTPException,Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import os, shutil
from langchain.docstore.document import Document
from context import summarize_extracted_text
from fileUploader_Utils import FileUploader
from vectorstore_utils import (
    VectorStoreBuilder,
    store_in_mongodb,
    get_vector_store_from_mongodb,
    unzip_to_memory,
    cleanup_vector_store_files
)
from queryResponse import (
    download_and_save_vector_store_by_folder,
    load_vector_store,
    pipeline_query_with_llm,
    batch_process_queries
)
import logging
import tiktoken
from embedding_model import initialize_model
import asyncio
from datetime import datetime, timedelta

# Setup logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

app = FastAPI()

# Global variable to track last cleanup time
last_cleanup_time = None
CLEANUP_INTERVAL = timedelta(hours=1)  # Run cleanup every hour

async def periodic_cleanup():
    """Background task to periodically clean up old vector store files"""
    global last_cleanup_time
    
    while True:
        try:
            current_time = datetime.utcnow()
            
            # Check if it's time to run cleanup
            if last_cleanup_time is None or (current_time - last_cleanup_time) >= CLEANUP_INTERVAL:
                log.info("Running periodic vector store cleanup...")
                cleanup_vector_store_files()
                last_cleanup_time = current_time
                log.info("Vector store cleanup completed")
            
            # Sleep for 5 minutes before checking again
            await asyncio.sleep(300)
            
        except Exception as e:
            log.error(f"Error in periodic cleanup: {str(e)}")
            await asyncio.sleep(300)  # Sleep before retrying

@app.on_event("startup")
async def startup_event():
    log.info("Initializing application...")
    try:
        initialize_model()
        # Start the periodic cleanup task
        asyncio.create_task(periodic_cleanup())
        log.info("Application initialized successfully")
    except Exception as e:
        log.error(f"Error during application initialization: {str(e)}")
        raise e

def count_tokens(text: str) -> int:
    """Count tokens in text using tiktoken"""
    encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))

# ============================
# Upload Endpoint
# ============================
@app.post("/upload/")
async def upload_and_process(
    files: list[UploadFile] = File(None),
    urls: list[str] = Form(None),
    user_id: str = Form(...),
    folder_id: str = Form(...),
    llm_provider: str = Form("groq")  # Default to groq for backward compatibility
):
    print(f"\n[Upload Process] Starting upload process for user_id: {user_id}, folder_id: {folder_id}")
    if not files and not urls:
        print("[Upload Process] Error: Neither files nor URLs provided")
        raise HTTPException(status_code=400, detail="Either files or URLs must be provided.") 
    try:
        uploader = FileUploader()
        combined_text = ""
        source_names = []

        # Process files if provided
        if files:
            for file in files:
                print(f"[Upload Process] Processing file: {file.filename}")
                temp_path = file.filename
                with open(temp_path, "wb") as buffer:
                    shutil.copyfileobj(file.file, buffer)
                text = uploader.load_file(temp_path)
                os.remove(temp_path)
                combined_text += f"\n\n=== Content from {file.filename} ===\n{text}"
                source_names.append(file.filename)
                print(f"[Upload Process] File processed successfully: {file.filename}")

        # Process URLs if provided
        if urls:
            for url in urls:
                print(f"[Upload Process] Processing URL: {url}")
                text = uploader.extract_text_from_url(url)
                source_name = os.path.basename(url)
                combined_text += f"\n\n=== Content from {source_name} ===\n{text}"
                source_names.append(source_name)
                print(f"[Upload Process] URL processed successfully: {source_name}")
        
        # Log original text token count
        original_tokens = count_tokens(combined_text)
        log.info(f"[Upload Process] Original document token count: {original_tokens:,} tokens")
        
        # Summarize the combined text
        print(f"[Upload Process] Summarizing combined text")
        summary = await summarize_extracted_text(combined_text, llm_provider=llm_provider)
        print(f"[Upload Process] Summary: {summary}")

        # Log summary token count
        summary_tokens = count_tokens(summary)
        log.info(f"[Upload Process] Summary token count: {summary_tokens:,} tokens")
        log.info(f"[Upload Process] Token reduction ratio: {summary_tokens/original_tokens:.2%}")
        log.info(f"[Upload Process] Original vs Summary tokens: {original_tokens:,} → {summary_tokens:,} tokens")

        print("[Upload Process] Creating vector store")
        builder = VectorStoreBuilder(file_name="combined_documents", folder_id=folder_id)
        document = Document(page_content=summary, metadata={"source": ", ".join(source_names)})
        split_docs = builder.split_documents([document])
        print(f"[Upload Process] Documents split into {len(split_docs)} chunks")
        
        builder.create_vectorstore(split_docs)
        print("[Upload Process] Vector store created successfully")

        vectorstore_info = builder.get_vectorstore_info()
        print(f"[Upload Process] Vector store info: uploading...")
        
        zip_data = builder.create_zip_archive()
        print("[Upload Process] Created zip archive of vector store")

        print("[Upload Process] Storing in MongoDB")
        mongo_id = store_in_mongodb(
            zip_data=zip_data,
            user_id=user_id,
            folder_id=folder_id,
            metadata={
                "vectorstore_info": vectorstore_info,
                "document_info": {
                    "sources": source_names,
                    "chunks": len(split_docs),
                    "chunk_size": builder.chunk_size,
                    "chunk_overlap": builder.chunk_overlap
                }
            }
        )
        print(f"[Upload Process] Stored in MongoDB with ID: {mongo_id}")

        persist_path = os.path.join(os.getcwd(), builder.persist_dir)
        if os.path.exists(persist_path):
            shutil.rmtree(persist_path)
            print(f"[Upload Process] Cleaned up temporary directory: {persist_path}")

        return JSONResponse(content={
            "status": "success",
            "mongo_id": mongo_id,
            "user_id": user_id,
            "folder_id": folder_id,
            "vectorstore_info": vectorstore_info,
            "sources": source_names
        })

    except Exception as e:
        print(f"[Upload Process] Error occurred: {str(e)}")
        if 'builder' in locals() and hasattr(builder, 'persist_dir'):
            persist_path = os.path.join(os.getcwd(), builder.persist_dir)
            if os.path.exists(persist_path):
                shutil.rmtree(persist_path)
                print(f"[Upload Process] Cleaned up temporary directory after error: {persist_path}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================
# Download Vector Store Endpoint
# ============================
@app.get("/download-and-save/{mongo_id}")
async def download_and_save_vector_store(mongo_id: str, user_id: str):
    print(f"\n[Download Process] Starting download for mongo_id: {mongo_id}, user_id: {user_id}")
    try:
        print("[Download Process] Fetching vector store from MongoDB")
        document = get_vector_store_from_mongodb(mongo_id, user_id)
        zip_data = document.get("vector_store")
        if not zip_data:
            print("[Download Process] Error: No zip data found in DB")
            raise HTTPException(status_code=404, detail="No zip data found in DB")

        metadata = document.get("metadata", {})
        vectorstore_info = metadata.get("vectorstore_info", {})
        persist_dir = vectorstore_info.get("persist_dir", "vector_store")
        print(f"[Download Process] Found vector store with persist_dir: {persist_dir}")

        VECTOR_STORE_BASE_DIR = os.getenv("VECTOR_STORE_BASE_DIR", "./vectorstores")
        extract_path = os.path.join(VECTOR_STORE_BASE_DIR, user_id, persist_dir)
        print(f"[Download Process] Extracting to path: {extract_path}")
        
        saved_path = unzip_to_memory(zip_data, extract_path)
        print(f"[Download Process] Successfully extracted to: {saved_path}")

        return {
            "status": "success",
            "saved_path": saved_path,
            "metadata": metadata
        }
    except Exception as e:
        print(f"[Download Process] Error occurred: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================
# Get Vector Store Info Endpoint
# ============================
@app.get("/vector-store-info/{mongo_id}")
async def get_vector_store_info(mongo_id: str, user_id: str):
    print(f"\n[Info Process] Fetching info for mongo_id: {mongo_id}, user_id: {user_id}")
    try:
        print("[Info Process] Fetching vector store from MongoDB")
        document = get_vector_store_from_mongodb(mongo_id, user_id)
        print("[Info Process] Successfully retrieved vector store info")
        return {
            "status": "success",
            "user_id": document["user_id"],
            "folder_id": document["folder_id"],
            "created_at": document["created_at"],
            "metadata": document["metadata"]
        }
    except Exception as e:
        print(f"[Info Process] Error occurred: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================
# Query Vector Store (JSON Input)
# ============================
from fastapi.exceptions import RequestValidationError
from fastapi.exception_handlers import request_validation_exception_handler
import json
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    # Prepare a dict with just the URL
    url_dict = {"url": str(request.url)}
    print("\n[Validation Error] on request:")
    print(json.dumps(url_dict, indent=4))
    raw_body = await request.body()
    try:
        print(f"Raw request body (str): {raw_body.decode('utf-8')}")
    except Exception:
        print("Cannot decode request body as UTF-8")
    print("Validation errors:")
    print(exc.errors())
    
    return await request_validation_exception_handler(request, exc)

class QueryRequest(BaseModel):
    query: str
    user_id: str
    folder_id: str
    batch_mode: bool = False
    llm_provider: str = "groq"  # Default to groq for backward compatibility

@app.post("/query/")
async def query_documents(request: QueryRequest):
    # Print expected schema
    print("\n[Expected Schema]")
    print({
        "query": "string",
        "user_id": "string",
        "folder_id": "string",
        "batch_mode": "boolean",
        "llm_provider": "string"
    })

    # Print actual received values
    print("\n[Received Request Body]")
    print(request.dict())  # or use request.json() if it's a raw body

    # Proceed with original logic
    query = request.query
    user_id = request.user_id
    folder_id = request.folder_id
    batch_mode = request.batch_mode
    llm_provider = request.llm_provider

    print(f"\n[Query Process] Starting query process for user_id: {user_id}, folder_id: {folder_id}")
    try:
        print("[Query Process] Getting vector store")
        retriever, vector_store_path = await download_and_save_vector_store_by_folder(user_id, folder_id)
        
        if batch_mode:
            print("[Query Process] Processing batch queries")
            queries = query.split('\n')
            results = await batch_process_queries(retriever, queries, user_id, folder_id, llm_provider)
        else:
            print("[Query Process] Processing single query")
            results = await pipeline_query_with_llm(retriever, query, user_id, folder_id, llm_provider)
        
        print("[Query Process] Query completed successfully")
        return JSONResponse(content=results)
        
    except Exception as e:
        print(f"[Query Process] Error occurred: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


