from bs4 import BeautifulSoup
from typing import List, Dict

def chunk_by_sections(soup: BeautifulSoup, url: str, page_title: str) -> List[Dict]:
    """Extract chunks from each top-level <section> inside <article>"""
    article = soup.find('article', id='furo-main-content')
    if not article:
        return []

    chunks = []

    # Find only *direct child* sections (top-level ones)
    top_sections = article.find_all('section', recursive=True)

    for section in top_sections:
        section_id = section.get('id', '')
        heading = section.find(['h1', 'h2', 'h3'])
        section_title = heading.get_text(strip=True) if heading else f"Section {section_id or 'Untitled'}"

        # Extract all visible text inside the section
        section_text = section.get_text(separator=' ', strip=True)

        # Sanity check
        if len(section_text) < 30:
            continue

        # Build chunk
        chunks.append({
            'title': f"{page_title} - {section_title}",
            'section_id': section_id,
            'text': section_text,
            'source_url': f"{url}#{section_id}" if section_id else url
        })

    return chunks
