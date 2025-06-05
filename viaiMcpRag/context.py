import os
from pydantic import BaseModel
from dotenv import load_dotenv
import logging
import re
from llm_provider import get_llm_provider

# Setup logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# Load environment variables from .env
load_dotenv()

# Define request model
class QueryRequest(BaseModel):
    query: str
    llm_provider: str = "groq"  # Default to groq for backward compatibility

def truncate_to_token_limit(text: str, max_tokens: int = 30000, buffer: int = 500) -> str:
    """
    Truncate the input text to fit within token limit.
    Approximate 1 token ≈ 4 characters.
    The buffer reserves tokens for prompt/query content.
    """
    max_chars = (max_tokens - buffer) * 4
    return text[:max_chars]

async def generate_response_from_llm(input_text: str, query: str = "", custom_prompt: str = None, llm_provider: str = "groq") -> str:
    # Check if the query is a greeting
    greeting_patterns = [
        r'^hi\b', r'^hello\b', r'^hey\b', r'^greetings\b',
        r'^good\s+(morning|afternoon|evening)\b',
        r'^howdy\b', r'^sup\b', r'^yo\b'
    ]
    
    if any(re.search(pattern, query.lower()) for pattern in greeting_patterns):
        return "Hello! I'm your AI assistant. How can I help you today?"

    base_prompt = custom_prompt or (
        "You are an expert query solver. If you cannot find any relevant information to answer the query, simply respond with 'No data found' without any explanation. "
        "Otherwise, provide a concise response in a short clear, informative paragraph. "
        "The paragraph must be maximum 50 words. "
        "Avoid introductions and focus strictly on essential facts and core ideas."
    )

    # Get the appropriate LLM provider
    provider = get_llm_provider(llm_provider)
    log.info(f"Using {llm_provider} provider for response generation")

    # Truncate input to avoid exceeding token limits
    truncated_input = truncate_to_token_limit(input_text)

    full_input = f"{base_prompt}\n\nUser Query: {query}\n\nJSON Data:\n{truncated_input}"

    # Generate response using the provider
    response = await provider.generate_response(full_input)

    # Clean the response: remove newlines, tabs, excessive whitespace, and unwanted characters
    cleaned_response = re.sub(r'\s+', ' ', response).strip()
    
    return cleaned_response

async def summarize_extracted_text(input_text: str, custom_prompt: str = None, llm_provider: str = "groq") -> str:
    summarization_prompt = custom_prompt or (
        "You are a highly skilled AI tasked with generating a comprehensive, structured summary of the following input text.\n\n"
        "Your goal is to extract and present all significant topics, subtopics, arguments, facts, and data points in detail. Do not omit key elements, even if the text is long.\n\n"
        "Instructions:\n"
        "- Cover every major topic and subtopic thoroughly, explaining the context and key points under each.\n"
        "- Preserve logical flow and structure, using bullet points or sections if appropriate.\n"
        "- Highlight factual data, technical information, definitions, and examples clearly.\n"
        "- Avoid vague generalizations — be precise and detailed.\n"
        "- Do not rewrite or paraphrase too abstractly; retain specific terminology where relevant.\n"
        "- Maintain objectivity and clarity, without editorializing or simplifying too much.\n"
        "- If there is structured content (tables, code, JSON, lists), describe their purpose and key elements accurately.\n"
        "- Do NOT include phrases like 'In conclusion' or 'The text discusses...'; simply present the extracted information.\n\n"
        "Output a detailed, accurate, and complete summary suitable for someone who needs a full understanding of the original content without reading it."
    )

    # Get the appropriate LLM provider
    provider = get_llm_provider(llm_provider)
    log.info(f"Using {llm_provider} provider for text summarization")

    # Truncate input to fit within token limits
    truncated_input = truncate_to_token_limit(input_text)

    full_input = f"{summarization_prompt}\n\nInput Text:\n{truncated_input}\n\nSummary:"

    # Generate summary using the provider
    return await provider.generate_response(full_input)