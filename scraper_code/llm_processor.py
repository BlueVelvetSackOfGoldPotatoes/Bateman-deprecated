# llm_processor.py

import openai
import config
import logging
import os
from typing import Dict, Any, List
from pdfminer.high_level import extract_text
import json
import re
import requests
from bs4 import BeautifulSoup
import traceback

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)  # Set to INFO or DEBUG as needed

# Create console handler
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)  # Set to INFO or DEBUG as needed

# Create formatter and add to handler
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)

# Add handler to logger
logger.addHandler(ch)

def process_pdf_with_llm(pdf_path: str, doi: str) -> Dict[str, Any]:
    """Process the PDF using LLM and extract meaningful data."""
    try:
        # Extract text from PDF
        text = extract_text(pdf_path)
        logger.info(f"Extracted text from {pdf_path}")

        # Prepare prompt for LLM
        prompt = generate_prompt(text, config.PRODUCT_DESCRIPTION)

        # Prepare messages for the LLM
        api_messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]

        # Call the LLM
        response = ask(
            api_messages=api_messages,
            temperature=0.5,
            model=config.LLM_MODELS["oa"]
        )

        # Parse the response and extract data
        extracted_data = parse_llm_response(response)

        # Include reasoning (assuming the LLM provides it)
        reasoning = extracted_data.get("reasoning", "N/A")

        # Update the JSON metadata with extracted data
        update_metadata_with_llm(doi, extracted_data)

        return extracted_data

    except Exception as e:
        logger.error(f"Error processing PDF with LLM for DOI {doi}: {e}")
        return {}

def generate_prompt(text: str, product_description: str) -> str:
    """Generate a prompt for the LLM based on the PDF text and product description."""
    prompt = f"""
You are an assistant tasked with extracting meaningful information from academic papers to identify potential leads for a tech product.

**Product Description:**
{product_description}

**Paper Content:**
{text}

**Tasks:**
1. Identify if the paper's research aligns with the product's application.
2. Extract contact information such as email addresses and phone numbers.
3. Summarize how the product could be used by the lead.
4. Provide reasoning behind why this lead is a good fit.

**Output Format (JSON):**
{{
    "contact_emails": [""],
    "contact_phones": [""],
    "summary": "",
    "reasoning": ""
}}
"""
    return prompt

def ask(api_messages, temperature, model):
    """Interface to interact with different LLMs based on the model prefix."""
    origin, specific_model = model.split(':', 1) if ':' in model else ('oa', model)

    if origin == "oa":
        openai.api_key = config.OPENAI_API_KEY
        completion = client.chat.completions.create(
            model=specific_model,
            messages=api_messages,
            temperature=temperature
        )
        response = completion.choices[0].message.content

    elif origin == "hf":
        # Implement Hugging Face integration if needed
        # For simplicity, not implemented here
        response = "Hugging Face integration is not implemented."

    else:
        raise ValueError(f"Unknown origin: {origin}")

    response = response.strip()
    return response

def parse_llm_response(response: str) -> Dict[str, Any]:
    """Parse the JSON response from the LLM."""
    try:
        # Ensure the response is valid JSON
        parsed = json.loads(response)
        return parsed
    except json.JSONDecodeError:
        logger.error("Failed to parse LLM response as JSON.")
        return {}

def update_metadata_with_llm(doi: str, extracted_data: Dict[str, Any]):
    """Update the JSON metadata file with the extracted data."""
    try:
        json_path = os.path.join("./data/rsc/doi", f"{sanitize_filename(doi)}.json")
        if not os.path.exists(json_path):
            logger.warning(f"Metadata JSON not found for DOI {doi}. Skipping update.")
            return

        with open(json_path, "r+", encoding="utf-8") as json_file:
            metadata = json.load(json_file)
            metadata.update(extracted_data)
            json_file.seek(0)
            json.dump(metadata, json_file, indent=4)
            json_file.truncate()
        logger.info(f"Updated metadata JSON with LLM data for DOI {doi}")

    except Exception as e:
        logger.error(f"Error updating metadata with LLM data for DOI {doi}: {e}")
        traceback.print_exc()

def update_metadata_with_additional_info(doi: str, additional_info: Dict[str, Any]):
    """Update the JSON metadata file with additional scraped information."""
    try:
        json_path = os.path.join("./data/rsc/doi", f"{sanitize_filename(doi)}.json")
        if not os.path.exists(json_path):
            logger.warning(f"Metadata JSON not found for DOI {doi}. Skipping update.")
            return

        with open(json_path, "r+", encoding="utf-8") as json_file:
            metadata = json.load(json_file)
            metadata.update(additional_info)
            json_file.seek(0)
            json.dump(metadata, json_file, indent=4)
            json_file.truncate()
        logger.info(f"Updated metadata JSON with additional info for DOI {doi}")

    except Exception as e:
        logger.error(f"Error updating metadata with additional info for DOI {doi}: {e}")
        traceback.print_exc()

def sanitize_filename(name: str) -> str:
    """Sanitize the filename by removing or replacing invalid characters."""
    return re.sub(r'[\\/*?:"<>|]', "_", name)

def scrape_additional_info(extracted_data: Dict[str, Any]) -> Dict[str, Any]:
    """Scrape additional information about the researcher from online sources."""
    try:
        authors = extracted_data.get("Authors", "")
        if not authors:
            return {}

        # Assume the first and last authors are the leads
        author_list = [author.strip() for author in authors.split(',')]
        if len(author_list) < 2:
            leads = [author_list[0]]
        else:
            leads = [author_list[0], author_list[-1]]

        additional_info = {}

        for lead in leads:
            logger.info(f"Scraping additional info for Lead: {lead}")
            # Perform Google search for the lead
            search_results = google_search(lead, num_pages=3)
            if not search_results:
                logger.warning(f"No search results found for {lead}")
                continue

            # Scrape landing pages and extract text
            combined_text = ""
            for url in search_results:
                page_text = scrape_landing_page(url)
                combined_text += " " + page_text

            if combined_text:
                # Send the combined text to LLM for filtering
                report = generate_additional_report(combined_text, lead)
                additional_info[f"{sanitize_filename(lead)}_report"] = report

        return additional_info

    except Exception as e:
        logger.error(f"Error scraping additional info: {e}")
        traceback.print_exc()
        return {}

def google_search(query: str, num_pages: int = 3) -> List[str]:
    """Perform a Google search and return the top URLs from the first few pages."""
    try:
        from serpapi import GoogleSearch

        params = {
            "engine": "google",
            "q": query,
            "api_key": config.SERPAPI_API_KEY,
            "num": 10,
            "start": 0
        }

        urls = []
        for page in range(num_pages):
            params["start"] = page * 10
            search = GoogleSearch(params)
            results = search.get_dict()
            organic_results = results.get('organic_results', [])
            for result in organic_results:
                link = result.get('link')
                if link:
                    urls.append(link)
        return urls
    except Exception as e:
        logger.error(f"Error performing Google search: {e}")
        return []

def scrape_landing_page(url: str) -> str:
    """Download the landing page HTML, extract text, and return it."""
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'lxml')
            # Remove script and style elements
            for script_or_style in soup(["script", "style"]):
                script_or_style.decompose()
            # Get text
            text = soup.get_text(separator=' ', strip=True)
            return text
        else:
            logger.warning(f"Failed to access URL: {url}")
            return ""
    except Exception as e:
        logger.error(f"Error scraping landing page {url}: {e}")
        return ""

def generate_additional_report(text: str, lead_name: str) -> str:
    """Generate a detailed report using LLM based on the scraped text."""
    try:
        prompt = f"""
You are an assistant tasked with analyzing information about a researcher.

**Lead Name:** {lead_name}

**Scraped Information:**
{text}

**Tasks:**
1. Reason about if this lead is a good fit for Quantum Nuova.
2. Produce a detailed report of why this lead is a good fit.
3. Provide background information of the researcher, their team/lab.
4. Explain how they could use the Quantum Nuova (QT-Sense device).
5. Extract contact details and address.
6. Identify if there is a recent grant they have received.

**Output Format (JSON):**
{{
    "is_good_fit": true/false,
    "reasoning": "",
    "researcher_background": "",
    "team_background": "",
    "usage_of_quantum_nuova": "",
    "contact_details": "",
    "address": "",
    "recent_grants": ""
}}
"""
        api_messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]

        response = ask(
            api_messages=api_messages,
            temperature=0.5,
            model=config.LLM_MODELS["oa"]
        )

        parsed_response = parse_llm_response(response)
        return parsed_response  # Return as a dictionary

    except Exception as e:
        logger.error(f"Error generating additional report: {e}")
        return {}
