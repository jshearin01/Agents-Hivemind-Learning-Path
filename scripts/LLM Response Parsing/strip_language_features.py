import re
import logging
import json
import traceback
from typing import Optional

# Configure the basic logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Create a logger
logger = logging.getLogger(__name__)

def extract_content(text: str) -> Optional[str]:
    """
    Extract content from markdown code blocks with improved regex handling.
    """
    try:
        # Log start of markdown extraction process
        logger.info(f"Input text length: {len(text)}")
        
        # First attempt: Look for JSON content inside markdown code blocks
        # The regex pattern matches optional 'json' language identifier and captures content between ``` delimiters
        json_match = re.search(r'```(?:json)?\s*(.*?)\s*```', text, re.DOTALL)
        if json_match:
            content = json_match.group(1).strip()
            logger.info("Found backticked JSON content in code block")
            return content
        # Second attempt: Look for HTML content inside markdown code blocks
        # The regex pattern matches optional 'html' language identifier and captures content between ``` delimiters
        html_match = re.search(r'```(?:html)?\s*(.*?)\s*```', text, re.DOTALL)
        if html_match:
            content = html_match.group(1).strip()
            logger.info("Found backticked HTML content in code block")
            return content
            
        # Third attempt: Look for raw JSON objects in the text
        # The regex pattern matches anything between curly braces, including nested objects
        json_match = re.search(r'(\{.*\})', text, re.DOTALL)
        if json_match:
            content = json_match.group(1).strip()
            logger.info("Found direct JSON content")
            return content
            
        # Fallback: If no JSON structure is found, clean and return the original text
        # This handles cases where the content is plain text or has a different format
        logger.info("No JSON found, returning cleaned text")
        return clean_text_content(text)
        
    except Exception as e:
        logger.error(f"Error in markdown extraction: {e}")
        logger.error(traceback.format_exc())
        return None

def clean_text_content(content: str) -> str:
    """Clean and normalize text content."""
    try:
        content = re.sub(r'//.*?\n', '', content)
        content = re.sub(r',(\s*[}\]])', r'\1', content)
        content = ' '.join(content.split())
        content = ''.join(char for char in content if ord(char) >= 32)
        return content.strip()
    except Exception as e:
        logger.error(f"Error cleaning text content: {e}")
        return content

def validate_html_content(content: str) -> bool:
    """Validate HTML content structure."""
    try:
        if not content:
            return False
        
        content = content.strip()
        
        # Basic structure validation
        is_valid = (
            content.startswith('<div') and 
            content.endswith('</div>') and
            'class=' in content.lower()
        )
        
        return is_valid
        
    except Exception as e:
        logger.error(f"Error validating HTML: {e}")
        return False

def get_clean_content(response: str) -> Optional[str]:
    """Clean and validate response content."""
    if not response:
        return None
    
    try:
        # Handle LangChain response object
        if hasattr(response, 'content'):
            response = response.content
        
        # Extract content from markdown if needed
        content = extract_content(response)
        if not content:
            logger.error("No content extracted from response")
            return None
        
        # Log content length for debugging
        logger.info(f'Extracted content length: {len(content)}')
        
        try:
            # Replace problematic characters and normalize JSON
            content = (content.replace("'", '"')
                                .replace('\n', '')
                                .replace('\r', '')
                                .strip())
            
            # Parse JSON and extract content
            parsed_content = json.loads(content)
            if isinstance(parsed_content, dict) and 'content' in parsed_content:
                content = parsed_content['content']
                logger.info('Successfully parsed content')
            else:
                logger.error("Parsed content missing required 'content' field")
                return None
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error: {e}")
            # Try direct content extraction if JSON parsing fails
            match = re.search(r'"content":\s*"(.*?)"(?=\s*[,}])', content)
            if match:
                content = match.group(1)
                logger.info('Extracted content using regex fallback')
            else:
                logger.error("Failed to extract content using regex fallback")
                return None
        
        # Validate HTML structure
        if not validate_html_content(content):
            logger.error("Invalid HTML structure in content")
            return None
        
        return clean_text_content(content)
        
    except Exception as e:
        logger.error(f"Response parsing error: {e}")
        logger.debug(f"Raw response: {str(response)[:500]}...")  # Log truncated response
        return None