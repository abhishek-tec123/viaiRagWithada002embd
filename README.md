VIAI MCP RAG System — Simple No-Code Guide

1. Upload URLs

You can upload content by providing links (URLs) to:
- Web pages (HTML)
- PDF files (.pdf)
- Word documents (.docx)
- Excel files (.xlsx, .xls)
- Blogs, news articles, and documentation pages

2. Upload Files

Supported file types you can upload:
- Plain text files: .txt
- PDF documents: .pdf
- Word documents: .docx
  (Note: .doc files are NOT supported — convert to .docx first)
- Excel spreadsheets: .xlsx, .xls

3. Query Content

You can ask questions in natural language about your uploaded URLs or files.
- Ask single or multiple questions
- Provide your user ID and folder ID
- Optionally choose the LLM provider (openai or groq)

That’s all! Contact us if you want step-by-step instructions.




# VIAI MCP RAG System - Query and URL Upload Guide

## URL Upload

### Supported URL Types
- Web pages (HTML content)
- Publicly accessible documents
- Text-based content pages
- Documentation pages
- Blog posts
- News articles

### URL Upload Endpoint (`POST /upload/`)

Upload and process URLs to create a searchable knowledge base.

**Request Body:**
```json
{
    "urls": ["https://example.com",]          // List of URLs to process
    "user_id": "string",                      // Required: Your user ID
    "folder_id": "string",                    // Required: Folder to store the processed content
    "llm_provider": "openai"                  // Optional: LLM provider (default: "openai")
}
```

**Response:**
```json
{
    "status": "success",
    "mongo_id": "string",                     // ID to reference this upload
    "user_id": "string",
    "folder_id": "string",
    "vectorstore_info": {
        // Vector store metadata
    },
    "sources": ["source_names"]               // Names of processed URLs
}
```

## Query System

### Query Endpoint (`POST /query/`)

Query the processed content using natural language.

**Request Body:**
```json
{
    "query": "string",                        // Your question or query
    "user_id": "string",                      // Required: Your user ID
    "folder_id": "string",                    // Required: Folder containing the processed content
    "batch_mode": false,                      // Optional: Process multiple queries (default: false)
    "llm_provider": "groq"                    // Optional: LLM provider (default: "groq")
}
```

### LLM Providers

The system supports multiple LLM providers for query processing:
- `groq` (default)
- `openai`

## Best Practices

1. **URL Upload**
   - Ensure URLs are publicly accessible
   - Prefer text-rich content pages
   - Avoid very large web pages
   - Use meaningful folder IDs for organization

2. **Querying**
   - Be specific in your questions
   - Use natural language
   - For batch queries, keep questions related to the same topic
   - Use appropriate LLM provider based on your needs


## Example Usage

1. **Uploading URLs:**
```python
import requests

urls = [
    "https://example.com/docs",
    "https://example.com/guide"
]

response = requests.post(
    "http://your-api/upload/",
    json={
        "urls": urls,
        "user_id": "user123",
        "folder_id": "docs_folder"
    }
)
```

2. **Querying Content:**
```python
import requests

# Single query
response = requests.post(
    "http://your-api/query/",
    json={
        "query": "What are the main features?",
        "user_id": "user123",
        "folder_id": "docs_folder"
    }
)

# Batch query
response = requests.post(
    "http://your-api/query/",
    json={
        "query": "What are the features?\nHow does it work?",
        "user_id": "user123",
        "folder_id": "docs_folder",
        "batch_mode": true
    }
)
```