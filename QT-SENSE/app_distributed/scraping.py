# scraping.py

import requests
from bs4 import BeautifulSoup
from constants import SERPER_API_KEY
import logging
import json
import pandas as pd

logger = logging.getLogger(__name__)

def scrape_landing_page(url):
    """
    Scrapes text content from a given URL.
    """
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        html = response.text

        soup = BeautifulSoup(html, 'html.parser')
        for script_or_style in soup(["script", "style"]):
            script_or_style.extract()
        text = soup.get_text(separator=' ')
        return text
    except Exception as e:
        logger.error(f"Error scraping {url}: {e}")
        return ""

def perform_google_search(query, num_results=3):
    """
    Performs a Google search using Serper API and returns a list of URLs.
    """
    try:
        url = "https://google.serper.dev/search"
        payload = {
            "q": query
        }
        headers = {
            'X-API-KEY': SERPER_API_KEY,
            'Content-Type': 'application/json'
        }
        response = requests.post(url, headers=headers, json=payload)
        if response.status_code == 200:
            data = response.json()
            urls = []
            for result in data.get('organic', []):
                if 'link' in result:
                    urls.append(result['link'])
                if len(urls) >= num_results:
                    break
            if not urls:
                logger.warning(f"No URLs found for '{query}'.")
            else:
                logger.info(f"Found {len(urls)} URLs for '{query}'.")
            return urls
        else:
            logger.error(f"Error during Google search for '{query}': {response.status_code} {response.text}")
            return []
    except Exception as e:
        logger.error(f"An error occurred during Google search for '{query}': {e}")
        return []

def read_pdf(file):
    """
    Reads a PDF file and returns the text content.
    """
    import PyPDF2
    try:
        pdf_reader = PyPDF2.PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    except Exception as e:
        logger.error(f"Error reading PDF file: {e}")
        return ""

def read_docx(file):
    """
    Reads a DOCX file and returns the text content.
    """
    from docx import Document
    try:
        doc = Document(file)
        text = "\n".join([para.text for para in doc.paragraphs])
        return text
    except Exception as e:
        logger.error(f"Error reading DOCX file: {e}")
        return ""

def extract_persons(leads_info_df):
    """
    Extracts unique person names associated with leads from the scraped data.
    """
    persons = []
    for _, row in leads_info_df.iterrows():
        contacts = row.get('Contacts', [])
        if isinstance(contacts, str):
            try:
                contacts = json.loads(contacts)
            except json.JSONDecodeError:
                logger.warning(f"Failed to decode Contacts for lead: {row.get('Company/Group Name', '')}")
                continue
        for contact in contacts:
            name = contact.get('Name')
            if name and name.strip().lower() != 'not available':
                persons.append((name.strip(), row.get('Company/Group Name', 'Unknown')))
    unique_persons = list(set(persons))
    return unique_persons
