#!/usr/bin/env python3
"""
Updated API Documentation Chunking Functions
Improved chunking for Redoc/Swagger documentation
"""

from bs4 import BeautifulSoup
from typing import List, Dict
from load_opeapi_spec import load_openapi_spec
def chunk_openapi_spec(spec: dict, base_url: str, page_title: str):
    chunks = []
    paths = spec.get("paths", {})

    for path, methods in paths.items():
        for method, details in methods.items():
            section_title = f"{method.upper()} {path}"
            lines = [f"### [{method.upper()}] {path}"]

            if 'summary' in details:
                lines.append(f"**Summary:** {details['summary']}")
            if 'description' in details:
                lines.append(f"**Description:** {details['description']}")
            if 'operationId' in details:
                lines.append(f"**Operation ID:** `{details['operationId']}`")
            if 'tags' in details:
                lines.append(f"**Tags:** {', '.join(details['tags'])}")
            if 'consumes' in details:
                lines.append(f"**Consumes:** {', '.join(details['consumes'])}")
            if 'produces' in details:
                lines.append(f"**Produces:** {', '.join(details['produces'])}")

            # Parameters
            parameters = details.get("parameters", [])
            if parameters:
                lines.append("**Parameters:**")
                for param in parameters:
                    location = param.get("in", "unknown")
                    name = param.get("name", "unnamed")
                    desc = param.get("description", "")
                    required = param.get("required", False)
                    lines.append(f"- `{name}` ({location}){' (required)' if required else ''}: {desc}")

            # Responses
            responses = details.get("responses", {})
            if responses:
                lines.append("**Responses:**")
                for status_code, resp in responses.items():
                    if "$ref" in resp:
                        ref = resp["$ref"]
                        lines.append(f"- `{status_code}`: See `{ref}`")
                    else:
                        desc = resp.get("description", "No description")
                        lines.append(f"- `{status_code}`: {desc}")

            chunks.append({
                "url": base_url,
                "page_title": page_title,
                "section_title": section_title,
                "content": "\n".join(lines)
            })

    return chunks



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

