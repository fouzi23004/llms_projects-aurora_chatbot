from bs4 import BeautifulSoup
from typing import List, Dict

from chunkers.html_chunker import chunk_by_sections
from chunkers.openapi_chunker import chunk_openapi_spec
from loaders.openapi_loader import load_openapi_spec


def smart_chunk_dispatcher(soup: BeautifulSoup, url: str, page_title: str,is_api_page : bool) -> List[Dict]:
    """
    Enhanced dispatcher that better detects API documentation pages
    """
  
    
    if is_api_page:
        spec_url = "https://documentation.ubuntu.com/lxd/latest/rest-api.yaml"
        spec = load_openapi_spec(spec_url)
        return chunk_openapi_spec(spec, url, page_title)
    # Otherwise, fall back to standard section chunking
    print("Using standard section chunking")
    return chunk_by_sections(soup, url, page_title)
