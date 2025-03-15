"""
URL utilities for the LinkedIn job scraper.
"""

from urllib.parse import urlparse

def normalize_url(url):
    """
    Normalize URL by removing query parameters for deduplication.
    
    Args:
        url (str): URL to normalize
        
    Returns:
        str or None: Normalized URL or None if input is None
    """
    if url:
        parsed = urlparse(url)
        return f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
    return None 