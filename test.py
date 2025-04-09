import requests
from bs4 import BeautifulSoup
import html2text
import json
import os
import re
from urllib.parse import urljoin, urlparse

def is_valid_url(url, base_domain):
    """Check if URL is valid and belongs to the same domain"""
    parsed = urlparse(url)
    return bool(parsed.netloc) and base_domain in parsed.netloc and parsed.scheme in ('http', 'https')

def scrape_page(url, visited_urls, base_domain, depth=0, max_depth=2):
    """Recursively scrape a page and its sublinks"""
    if url in visited_urls or depth > max_depth or not is_valid_url(url, base_domain):
        return {}
    
    print(f"Scraping: {url} (depth: {depth})")
    visited_urls.add(url)
    
    try:
        response = requests.get(url, timeout=10)
        if response.status_code != 200:
            print(f"Failed to retrieve {url}. Status code: {response.status_code}")
            return {}
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Initialize html2text
        text_maker = html2text.HTML2Text()
        text_maker.ignore_links = False
        text_maker.ignore_images = True
        text_maker.ignore_tables = False
        
        # Extract main content
        main_content = soup.find('main')
        if not main_content:
            main_content = soup.find('div', class_='c-wysiwyg')
            if not main_content:
                main_content = soup  # Use the entire page if specific tags not found
        
        # Convert to markdown text
        markdown_text = text_maker.handle(str(main_content))
        
        # Extract page title
        title = soup.find('title')
        title_text = title.text.strip() if title else url.split('/')[-1]
        
        # Extract page metadata
        meta_description = ""
        meta_desc_tag = soup.find('meta', attrs={'name': 'description'})
        if meta_desc_tag and 'content' in meta_desc_tag.attrs:
            meta_description = meta_desc_tag['content']
        
        # Create page data structure
        page_data = {
            "url": url,
            "title": title_text,
            "meta_description": meta_description,
            "content": markdown_text,
            "subpages": {}
        }
        
        # Find all links on the page
        if depth < max_depth:
            links = soup.find_all('a', href=True)
            for link in links:
                href = link['href']
                full_url = urljoin(url, href)
                
                # Skip non-HTML resources and external links
                if not is_valid_url(full_url, base_domain) or full_url == url:
                    continue
                
                # Skip if URL contains specific patterns (like PDF files, etc.)
                if any(ext in full_url.lower() for ext in ['.pdf', '.jpg', '.png', '.gif', '.zip']):
                    continue
                
                # Only follow links that seem relevant to the CS program
                if any(keyword in full_url.lower() for keyword in ['computer-science', 'compsci', 'comp-sci', 'cs-program', 'engineering', 'software']):
                    subpage_data = scrape_page(full_url, visited_urls, base_domain, depth + 1, max_depth)
                    if subpage_data:
                        page_data["subpages"][full_url] = subpage_data
        
        return page_data
    
    except Exception as e:
        print(f"Error scraping {url}: {str(e)}")
        return {}

def extract_structured_data(page_data):
    """Extract structured information from the scraped data"""
    structured_data = {
        "program_name": "Computer Science (BCompSc)",
        "faculty": "Gina Cody School of Engineering and Computer Science",
        "department": "Computer Science and Software Engineering",
        "degree": "Bachelor of Computer Science (BCompSc)",
        "urls": {},
        "admission_requirements": {},
        "program_structure": {},
        "courses": [],
        "specializations": [],
        "career_options": [],
        "faculty_info": [],
        "research_areas": []
    }
    
    # Process main page and subpages
    process_page_content(page_data, structured_data)
    
    return structured_data

def process_page_content(page_data, structured_data):
    """Process content from a page and extract structured information"""
    if not page_data:
        return
    
    url = page_data.get("url", "")
    content = page_data.get("content", "")
    title = page_data.get("title", "")
    
    # Store URL with its title
    structured_data["urls"][url] = title
    
    # Look for admission requirements
    if "admission" in url.lower() or "admission" in title.lower() or "requirements" in url.lower():
        structured_data["admission_requirements"][url] = {
            "title": title,
            "content": content[:1000]  # Store a preview of the content
        }
    
    # Look for program structure
    if "program" in url.lower() or "structure" in url.lower() or "curriculum" in url.lower():
        structured_data["program_structure"][url] = {
            "title": title,
            "content": content[:1000]
        }
    
    # Look for course information
    if "course" in url.lower() or "courses" in title.lower():
        course_info = {
            "url": url,
            "title": title,
            "description": content[:1000]
        }
        structured_data["courses"].append(course_info)
    
    # Look for specializations
    if "specialization" in url.lower() or "option" in url.lower() or "concentration" in url.lower():
        spec_info = {
            "url": url,
            "title": title,
            "description": content[:1000]
        }
        structured_data["specializations"].append(spec_info)
    
    # Look for career information
    if "career" in url.lower() or "job" in url.lower() or "after" in url.lower():
        career_info = {
            "url": url,
            "title": title,
            "description": content[:1000]
        }
        structured_data["career_options"].append(career_info)
    
    # Process subpages
    for subpage_url, subpage_data in page_data.get("subpages", {}).items():
        process_page_content(subpage_data, structured_data)

def scrape_concordia_cs_program():
    """Main function to scrape Concordia's Computer Science program"""
    # URL of the main page
    main_url = "https://www.concordia.ca/academics/undergraduate/computer-science.html"
    base_domain = "concordia.ca"
    
    # Set to keep track of visited URLs
    visited_urls = set()
    
    # Scrape the main page and its sublinks
    raw_data = scrape_page(main_url, visited_urls, base_domain)
    
    # Extract structured data
    structured_data = extract_structured_data(raw_data)
    
    # Save the raw data
    with open("concordia_cs_program_raw.json", "w", encoding="utf-8") as f:
        json.dump(raw_data, f, indent=4)
    
    # Save the structured data
    with open("concordia_cs_program_data.json", "w", encoding="utf-8") as f:
        json.dump(structured_data, f, indent=4)
    
    return {
        "raw_data_preview": {
            "title": raw_data.get("title", ""),
            "url": raw_data.get("url", ""),
            "content_preview": raw_data.get("content", "")[:500] + "..." if raw_data.get("content") else "",
            "subpages_count": len(raw_data.get("subpages", {}))
        },
        "structured_data": structured_data,
        "pages_scraped": len(visited_urls)
    }

# Execute the function
if __name__ == "__main__":
    result = scrape_concordia_cs_program()
    print("\nData extraction complete!")
    print(f"Pages scraped: {result['pages_scraped']}")
    print(f"Raw data preview: {result['raw_data_preview']['title']}")
    print(f"Subpages found: {result['raw_data_preview']['subpages_count']}")
    print(f"Structured data categories: {list(result['structured_data'].keys())}")
    
    # Print some statistics about the structured data
    print("\nStructured Data Statistics:")
    print(f"URLs collected: {len(result['structured_data']['urls'])}")
    print(f"Admission requirement pages: {len(result['structured_data']['admission_requirements'])}")
    print(f"Program structure pages: {len(result['structured_data']['program_structure'])}")
    print(f"Course information pages: {len(result['structured_data']['courses'])}")
    print(f"Specialization pages: {len(result['structured_data']['specializations'])}")
    print(f"Career information pages: {len(result['structured_data']['career_options'])}")
