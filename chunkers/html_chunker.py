
from bs4 import BeautifulSoup
from typing import List, Dict

from chunkers.openapi_chunker import chunk_openapi_spec
from loaders.openapi_loader import load_openapi_spec
import os
from dotenv import load_dotenv

load_dotenv()  # load .env variables

def chunk_by_sections(soup: BeautifulSoup, url: str, page_title: str) -> List[Dict]:
    """Standard section-based chunking for regular documentation"""
    article = soup.find('article', id='furo-main-content')
    if not article:
        # Try alternative selectors
        article = soup.find('article') or soup.find('main') or soup.find('div', class_='content')
    
    if not article:
        return []

    chunks = []
    sections = article.find_all('section', recursive=True)

    for section in sections:
        section_id = section.get('id', '')
        heading = section.find(['h1', 'h2', 'h3'])
        section_title = heading.get_text(strip=True) if heading else f"Section {section_id or 'Untitled'}"

        section_text = section.get_text(separator=' ', strip=True)

        if len(section_text) < 30:
            continue

        chunks.append({
            'title': f"{page_title} - {section_title}",
            'section_id': section_id,
            'text': section_text,
            'source_url': f"{url}#{section_id}" if section_id else url
        })

    return chunks



def smart_chunk_dispatcher(soup: BeautifulSoup, url: str, page_title: str, is_api_page: bool) -> List[Dict]:
    """
    Enhanced dispatcher that better detects API documentation pages
    """

    if is_api_page:
        spec_filename = os.getenv("OPENAPI_SPEC_FILE", "rest-api.yaml")
        base_url = os.getenv("UBUNTU_DOC_URL", "https://documentation.ubuntu.com/lxd/latest/")
        spec_url = f"{base_url.rstrip('/')}/{spec_filename.lstrip('/')}"
        spec = load_openapi_spec(spec_url)
        return chunk_openapi_spec(spec, url, page_title)

    print("Using standard section chunking")
    return chunk_by_sections(soup, url, page_title)
