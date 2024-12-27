
"""
    LLMs often return the content within a formatted string such as
    "content": ```json {....}```
    or 
    "content": ```html <div>....</div>```
    and often include new line characters and double backslashes.
    This script parses and extracts the clean content string.
"""
import re

def extract_content(text: str) -> Optional[str]:
    """
    Extract content from markdown code blocks with improved regex handling.
    """
    try:
        # Log start of markdown extraction process
        print("=== START MARKDOWN EXTRACTION ===")
        print(f"Input text length: {len(text)}")
        
        # First attempt: Look for JSON content inside markdown code blocks
        # The regex pattern matches optional 'json' language identifier and captures content between ``` delimiters
        json_match = re.search(r'```(?:json)?\s*(.*?)\s*```', text, re.DOTALL)
        if json_match:
            content = json_match.group(1).strip()
            print("Found JSON content in code block")
            return content
        # Second attempt: Look for HTML content inside markdown code blocks
        # The regex pattern matches optional 'html' language identifier and captures content between ``` delimiters
        html_match = re.search(r'```(?:html)?\s*(.*?)\s*```', text, re.DOTALL)
        if html_match:
            content = html_match.group(1).strip()
            print("Found HTML content in code block")
            return content
            
        # Third attempt: Look for raw JSON objects in the text
        # The regex pattern matches anything between curly braces, including nested objects
        json_match = re.search(r'(\{.*\})', text, re.DOTALL)
        if json_match:
            content = json_match.group(1).strip()
            print("Found direct JSON content")
            return content
            
        # Fallback: If no JSON structure is found, clean and return the original text
        # This handles cases where the content is plain text or has a different format
        print("No JSON found, returning cleaned text")
        return clean_text_content(text)
    except:
        return "None"

def clean_text_content(content: str) -> str:
    """Clean and normalize text content."""
    content = re.sub(r'//.*?\n', '', content)
    content = re.sub(r',(\s*[}\]])', r'\1', content)
    content = ' '.join(content.split())
    content = ''.join(char for char in content if ord(char) >= 32)
    return content.strip()


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
        print(f"Error validating HTML: {e}")
        return False

def parse_response(response: str) -> Optional[str]:
    """Parse and validate response content."""
    if not response:
        return None
        
    try:
        # Try direct JSON parsing first
        if response.strip().startswith('{'):
            parsed = json.loads(response)
            if isinstance(parsed, dict) and 'content' in parsed:
                return parsed['content']
        
        # Extract from markdown if needed
        content = extract_content(response)
        if not content:
            print("No content")
            return None
        else:
            content = json.loads(content)['content']
            print('extracted content, json.loads:', content)
        
        # Validate HTML structure
        if not validate_html_content(content):
            print("No validate html content")
            return None
        else:
            print('Valid html')

        cleaned_content = clean_text_content(content)
            
        return content
        
    except Exception as e:
        print(f"Response parsing error: {e}")
        return None