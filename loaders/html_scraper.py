#!/usr/bin/env python3
"""
Simplified Ubuntu Documentation Scraper
Quick script to scrape and chunk Ubuntu documentation
"""

import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import time
import re
import json
import os
from typing import List, Dict

from ingest import smart_chunk_dispatcher




def scrape_ubuntu_doc_page(url: str) -> Dict:
    """Scrape a single Ubuntu documentation page with better content filtering"""
    headers = {
        'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36'
    }

    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')

        # Check if this is an API documentation page before removing elements
        is_api_page = ('/api/' in url or url.endswith('/api') or 
                      any(soup.select(indicator) for indicator in [
                          'div.redoc-container', 'div#redoc-container', 
                          '.redoc-wrap', 'redoc', 'div[data-redoc]'
                      ]))

        # Remove unwanted elements, but be more careful with API pages
        if is_api_page:
            # For API pages, only remove clearly non-content elements
            unwanted_selectors = [
                'script', 'style', 'header', 'footer',
                '.social-share', '.comments', '.feedback'
            ]
        else:
            # For regular pages, remove navigation elements more aggressively
            unwanted_selectors = [
                'script', 'style', 'nav', 'header', 'footer',
                '.sidebar', '.navigation', '.nav', '.menu',
                '.breadcrumb', '.breadcrumbs', '.toc', '.table-of-contents',
                '.page-nav', '.pagination', '.related-links',
                '.social-share', '.comments', '.feedback'
            ]

        # Extract title
        title_elem = soup.find('h1') or soup.find('title')
        title = title_elem.get_text().strip() if title_elem else "Untitled"

        # Try to find the actual article content with better selectors
        if is_api_page:
            # For API pages, look for Redoc containers first
            content_selectors = [
                'div.redoc-container',
                'div#redoc-container',
                '.redoc-wrap',
                'article',
                '.content',
                '.main-content',
                '[role="main"]',
                'main'
            ]
        else:
            # For regular pages, use standard selectors
            content_selectors = [
                'article',
                '.content',
                '.main-content',
                '.article-content',
                '.documentation-content',
                '[role="main"]',
                'main'
            ]


        main_content = None
        for selector in content_selectors:
            main_content = soup.select_one(selector)
            if main_content:
                break

        # If still no main content, try body but remove common non-content elements
        if not main_content:
            main_content = soup.find('body')
            if main_content:
                # Remove more navigation and sidebar elements from body
                for elem in main_content.select('.sidebar, .nav, .navigation, aside'):
                    elem.decompose()

        if main_content:
            # Get clean text
            content = main_content.get_text()

            # Clean up whitespace more efficiently
            content = re.sub(r'\n\s*\n\s*\n+', '\n\n', content)  # Multiple newlines to double
            content = re.sub(r'[ \t]+', ' ', content)  # Multiple spaces to single
            content = content.strip()

            # Check content length to prevent memory issues
            if len(content) > 500000:  # 500KB limit
                print(f"Warning: Large content ({len(content)} chars) from {url}, truncating...")
                content = content[:500000] + "...[TRUNCATED]"

            # Basic content validation
            if len(content) < 5:  # Too short, might be empty page
                return {'url': url, 'success': False, 'error': 'Content too short'}

            return {
                'url': url,
                'title': title,
                'content': content,
                'soup': soup,
                'success': True
}

    except MemoryError:
        print(f"Memory error scraping {url} - content too large")
        return {'url': url, 'success': False, 'error': 'Memory error'}
    except Exception as e:
        print(f"Error scraping {url}: {e}")
        return {'url': url, 'success': False, 'error': str(e)}

    return {'url': url, 'success': False}


def discover_doc_links(start_url: str, max_pages: int = 50) -> List[str]:
    """Discover documentation links from navigation"""
    discovered_urls = set([start_url])
    urls_to_check = [start_url]

    headers = {
        'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36'
    }

    base_path = urlparse(start_url).path.rstrip('/').rsplit('/', 1)[0]

    while urls_to_check and len(discovered_urls) < max_pages:
        current_url = urls_to_check.pop(0)

        try:
            response = requests.get(current_url, headers=headers, timeout=10)
            soup = BeautifulSoup(response.content, 'html.parser')

            # Look for navigation links, table of contents, etc.
            nav_selectors = ['nav a', '.toc a', '.navigation a', '.sidebar a', 'aside a']

            for selector in nav_selectors:
                links = soup.select(selector)
                for link in links:
                    href = link.get('href')
                    if href:
                        full_url = urljoin(current_url, href).split('#')[0]
                        parsed = urlparse(full_url)

                        # Check if it's within the same documentation section
                        if (parsed.path.startswith(base_path) and
                                full_url not in discovered_urls and
                                not href.startswith(('http://', 'https://')) or
                                'documentation.ubuntu.com' in parsed.netloc):

                            discovered_urls.add(full_url)
                            if len(urls_to_check) < 20:  # Limit queue size
                                urls_to_check.append(full_url)

            time.sleep(0.5)  # Be respectful

        except Exception as e:
            print(f"Error discovering from {current_url}: {e}")

    return list(discovered_urls)

  

def scrape_ubuntu_documentation(start_url: str, max_pages: int = 50) -> List[Dict]:
    """Complete scraping pipeline for Ubuntu documentation with memory management"""
    print(f"Starting scrape of {start_url}")

    # Discover all documentation URLs
    print("Discovering documentation pages...")
    urls = discover_doc_links(start_url, max_pages)
    print(f"Found {len(urls)} pages to scrape")

    # Limit the number of URLs to prevent memory issues
    if len(urls) > max_pages:
        urls = urls[:max_pages]
        print(f"Limited to first {max_pages} pages")

    # Scrape each page
    all_chunks = []
    successful_pages = 0
    failed_pages = 0
    
    for i, url in enumerate(urls, 1):
        print(f"Scraping page {i}/{len(urls)}: {url}")

        page_data = scrape_ubuntu_doc_page(url)
        if page_data['success'] and page_data.get('content'):
            try:
                # Chunk the content
                soup = page_data['soup']
                is_api_page = ('/api/' in url or url.endswith('/api') or 
                      any(soup.select(indicator) for indicator in [
                          'div.redoc-container', 'div#redoc-container', 
                          '.redoc-wrap', 'redoc', 'div[data-redoc]'
                      ]))

                section_chunks = smart_chunk_dispatcher(soup, url, page_data['title'], is_api_page)

                for j, chunk in enumerate(section_chunks):
                    chunk['chunk_index'] = j
                    chunk['total_chunks'] = len(section_chunks)
                    all_chunks.append(chunk)


                successful_pages += 1
                print(f"  ✓ Generated {len(section_chunks)} chunks")

            except Exception as e:
                print(f"  ✗ Error chunking content: {e}")
                failed_pages += 1
        else:
            failed_pages += 1
            error_msg = page_data.get('error', 'Unknown error')
            print(f"  ✗ Failed to scrape: {error_msg}")

        # Memory management: periodically clean up
        if i % 10 == 0:
            import gc
            gc.collect()

        time.sleep(1)  # Be respectful to the server

    print(f"\nScraping complete!")
    print(f"✓ Successful pages: {successful_pages}")
    print(f"✗ Failed pages: {failed_pages}")
    print(f"📄 Total chunks generated: {len(all_chunks)}")

    return all_chunks
