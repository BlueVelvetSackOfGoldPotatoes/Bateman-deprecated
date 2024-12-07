# utils.py

import streamlit as st
import json
import re
import subprocess
import os
import requests
import time
import random
from bs4 import BeautifulSoup
import tiktoken
from io import BytesIO
import logging
from dotenv import load_dotenv

# import from openai_api.py
from llm_center import llm_reasoning

from prompts import *

# ==========================
# Configure Logging
# ==========================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

load_dotenv()

def display_lead_information(leads, button_label, filename, height=400):
    """
    Displays the provided leads (list of dicts) and provides download options.
    
    :param leads: List of dictionaries containing lead information.
    :param button_label: The label for the download button.
    :param filename: The base filename for the downloaded file.
    :param height: The height of the displayed information.
    """
    # Display all leads in JSON format
    st.json(leads, expanded=False)
    
    # Convert to JSON string for download
    json_data = json.dumps(leads, indent=2)
    
    st.download_button(
        label=f"Download {button_label} as JSON",
        data=json_data,
        file_name=f"{filename}.json",
        mime='application/json'
    )
    
    # Separate companies and personnel
    companies = [lead for lead in leads if lead.get('Type') == "Company"]
    personnel = [lead for lead in leads if lead.get('Type') == "Personnel"]
    
    # Display Companies Information
    if companies:
        st.write("### Companies")
        for company in companies:
            st.json(company, expanded=False)
            st.markdown("---")
    
    # Display Personnel Information
    if personnel:
        st.write("### Personnel")
        for person in personnel:
            st.json(person, expanded=False)
            st.markdown("---")


def split_text_into_chunks(text, max_tokens=124000, overlap=100):
    """
    Splits text into chunks with a maximum number of tokens, allowing for overlap between chunks.
    
    :param text: The input text to split.
    :param max_tokens: The maximum number of tokens per chunk.
    :param overlap: The number of tokens to overlap between chunks.
    :return: A list of text chunks.
    """
    encoding = os.environ.get("ENCODING", "gpt2")  # Default to 'gpt2' if not set
    tokens = tiktoken.get_encoding(encoding).encode(text)
    
    chunks = []
    start = 0
    end = max_tokens
    
    while start < len(tokens):
        chunk_tokens = tokens[start:end]
        chunk_text = tiktoken.get_encoding(encoding).decode(chunk_tokens)
        chunks.append(chunk_text)
        start = end - overlap  # Move back by 'overlap' tokens for the next chunk
        end = start + max_tokens
    
    return chunks

def clean_text(text):
    """
    Cleans and normalizes text data.
    """
    return re.sub(r'\s+', ' ', text).strip()


def count_tokens(text):
    """
    Counts the number of tokens in a given text using the specified encoding.

    :param text: The text to count tokens for.
    :return: The number of tokens.
    """
    encoding = os.environ.get("ENCODING", "gpt2")  # Default to 'gpt2' if not set
    return len(tiktoken.get_encoding(encoding).encode(text))


def extract_persons(leads_info):
    """
    Extracts unique person names associated with leads from the scraped data.
    Processes both 'Contacts' and 'Researchers' fields.
    
    :param leads_info: List of dictionaries containing lead information.
    :return: List of tuples (person_name, associated_entity)
    """
    persons = []
    for lead in leads_info:
        lead_entity = lead.get('Entity', 'Unknown')
        
        # Process Contacts
        contacts = lead.get('Contacts', [])
        if isinstance(contacts, str):
            try:
                contacts = json.loads(contacts)
            except json.JSONDecodeError:
                logger.warning(f"Failed to decode Contacts for lead: {lead_entity}")
                contacts = []
        for contact in contacts:
            name = contact.get('Name')
            if name and name.strip().lower() != 'not available':
                persons.append((name.strip(), lead_entity))
        
        # Process Researchers
        researchers = lead.get('Researchers', [])
        if isinstance(researchers, str):
            try:
                researchers = json.loads(researchers)
            except json.JSONDecodeError:
                logger.warning(f"Failed to decode Researchers for lead: {lead_entity}")
                researchers = []
        for researcher in researchers:
            if isinstance(researcher, dict):
                name = researcher.get('name') or researcher.get('Name')
            else:
                name = researcher
            if name and name.strip().lower() != 'not available':
                persons.append((name.strip(), lead_entity))
    
    # Remove duplicates
    unique_persons = list(set(persons))
    return unique_persons


def download_leads(leads, json_filename):
    """
    Facilitates the download of leads as a JSON file.
    
    :param leads: List of dictionaries containing lead information.
    :param json_filename: The base filename for the downloaded JSON file.
    """
    try:
        json_data = json.dumps(leads, indent=2)
        st.download_button(
            label=f"Download {json_filename} as JSON",
            data=json_data,
            file_name=f"{json_filename}.json",
            mime='application/json'
        )
    except Exception as e:
        logger.error(f"Error while downloading JSON file: {e}")


def clean_html(html, url):
    try:
        soup = BeautifulSoup(html, 'html.parser')

        # Remove non-visible elements
        for element in soup(["script", "style", "head", "meta", "noscript", "iframe", "header", "footer", "nav", "aside"]):
            element.extract()

        # Define patterns for common layout components
        layout_patterns = [
            r'\bmenu\b', r'\bfooter\b', r'\bsidebar\b', r'\bheader\b',
            r'\bnav\b', r'\bnavigation\b', r'\bnavbar\b', r'\btopbar\b',
            r'\btoc\b', r'\bbreadcrumb\b', r'\bsitemap\b', r'\bwidget\b',
            r'\bpromo\b', r'\bcopyright\b', r'\ball rights reserved\b'
        ]

        # Compile regex for performance
        layout_regex = re.compile('|'.join(layout_patterns), re.IGNORECASE)

        # Remove elements with class or id matching layout patterns
        for tag in soup.find_all(True):  # True matches all tags
            class_attr = tag.get('class', [])
            id_attr = tag.get('id', '')
            
            # Combine class and id attributes for matching
            attributes = ' '.join(class_attr) + ' ' + id_attr
            if layout_regex.search(attributes):
                tag.extract()

        # Get visible text
        text = soup.get_text(separator=' ')

        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        # Define non-informational patterns to remove from text
        non_informational_patterns = [
            r'accept cookies', r'terms of service', r'privacy policy', r'navigation',
            r'back to top', r'© \d{4}', r'all rights reserved',
            r'subscribe', r'contact us', r'follow us', r'login', r'sign up',
            r'powered by', r'careers', r'blog', r'faq', r'sitemap', r'ads?'
        ]

        # Compile regex for performance
        non_info_regex = re.compile('|'.join(non_informational_patterns), re.IGNORECASE)

        # Remove non-informational patterns from text
        cleaned_text = non_info_regex.sub('', text)

        # Optionally, remove extra spaces introduced by pattern removal
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()

        return cleaned_text

    except Exception as parse_error:
        logger.error(f"Error parsing HTML for {url}: {parse_error}")
        return ""


def scrape_landing_page(url):
    """
    Scrapes text content from a given URL, falling back to `curl` if `requests` fails or returns no content.
    Cleans the text to ensure it contains only meaningful content.

    Args:
        url (str): The URL to scrape.

    Returns:
        str: The cleaned text content from the page, or an empty string if scraping fails.
    """
    html = ""

    # Attempt to fetch using `requests`
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=5)
        response.raise_for_status()
        html = response.text
        if not html.strip():  # Check for empty content
            logger.warning(f"No content received for {url} with requests. Trying curl...")
    except requests.exceptions.RequestException as e:
        logger.warning(f"Error with requests for {url}: {e}. Trying curl...")

    # Fallback to `curl` if necessary
    if not html.strip():
        try:
            result = subprocess.run(
                ['curl', '-L', '-s', url, '-A', 'Mozilla/5.0'],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            if result.returncode == 0:
                html = result.stdout
            else:
                logger.error(f"Error using curl: {result.stderr}")
                return ""
        except Exception as curl_error:
            logger.error(f"Error running curl for {url}: {curl_error}")
            return ""

    return clean_html(html, url)


def run_query(query, num_results, url):
    urls = []
    payload = {
        "q": query
    }
    headers = {
        'X-API-KEY': os.environ.get("SERPER_API_KEY"),
        'Content-Type': 'application/json'
    }
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=10)
        response.raise_for_status()  # Raises HTTPError for bad responses (4xx or 5xx)
        data = response.json()
        
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
    except requests.exceptions.HTTPError as http_err:
        logger.error(f"HTTP error occurred during Google search for '{query}': {http_err}")
    except requests.exceptions.ConnectionError as conn_err:
        logger.error(f"Connection error occurred during Google search for '{query}': {conn_err}")
    except requests.exceptions.Timeout as timeout_err:
        logger.error(f"Timeout error occurred during Google search for '{query}': {timeout_err}")
    except requests.exceptions.RequestException as req_err:
        logger.error(f"Request exception occurred during Google search for '{query}': {req_err}")
    except ValueError as json_err:
        logger.error(f"JSON decode error for query '{query}': {json_err}")
    except Exception as e:
        logger.error(f"Unexpected error during Google search for '{query}': {e}")
    return []


def perform_google_search(base_query, num_results=3, keywords=None, lead_name=None):
    """
    Performs Google searches using Serper API.

    :param base_query: The base search query string (e.g., person's name).
    :param num_results: Number of top results to retrieve per keyword. 
    :param keywords: List of keywords to append to the base query. If None, perform a single search.
    :param lead_name: (Optional) Lead name to include in the search.
    :return: 
        - If keywords is None, returns a list of URLs.
        - If keywords are provided, returns a dict mapping each keyword to its list of URLs.
    """
    try:
        api_url = "https://google.serper.dev/search"

        if not keywords:
            # Perform single search
            query_parts = [base_query]
            if lead_name:
                query_parts.append(lead_name)
            final_query = " + ".join(query_parts)

            logger.debug(f"Final search query: {final_query}")

            # Execute the query
            query_urls = run_query(final_query, num_results, api_url)
            if not query_urls:
                logger.warning(f"No URLs found for query '{final_query}'.")
                return []
            else:
                # To prevent hitting rate limits, introduce a delay
                time.sleep(random.uniform(1, 3))
                return query_urls
        else:
            # Perform per-keyword searches
            results = {}
            for keyword in keywords:
                # Build query: base_query + lead_name + keyword
                query_parts = [base_query]
                if lead_name:
                    query_parts.append(lead_name)
                if keyword:
                    query_parts.append(keyword)
                final_query = " + ".join(query_parts)

                logger.debug(f"Final search query for keyword '{keyword}': {final_query}")

                # Execute the query
                query_urls = run_query(final_query, num_results, api_url)
                if not query_urls:
                    logger.warning(f"No URLs found for query '{final_query}'.")
                    results[keyword] = []
                else:
                    results[keyword] = query_urls

                # To prevent hitting rate limits, introduce a delay
                time.sleep(random.uniform(1, 3))  # Random delay between 1 and 3 seconds

            return results

    except Exception as e:
        logger.error(f"An error occurred during Google search: {e}")
        return {} if keywords else []


def clean_leads(leads):
    """
    Cleans and normalizes leads data.
    
    :param leads: List of dictionaries containing lead information.
    :return: Cleaned list of dictionaries.
    """
    cleaned_leads = []
    for lead in leads:
        cleaned_lead = {}
        for key, value in lead.items():
            if isinstance(value, list):
                cleaned_lead[key] = ', '.join(map(str, value))
            elif isinstance(value, dict):
                # Flatten nested dictionaries if necessary
                for sub_key, sub_value in value.items():
                    cleaned_lead[f"{key} {sub_key}"] = sub_value if sub_value else "Not Available"
            else:
                cleaned_lead[key] = value if value else "Not Available"
        cleaned_lead['Source URLs'] = lead.get('Source URLs', [])
        cleaned_lead['Type'] = lead.get('Type', "Company" if 'CEO/PI' in lead else "Personnel")
        cleaned_leads.append(cleaned_lead)
    return cleaned_leads


def extract_lead_info_with_llm_per_field(lead_name, columns_to_retrieve):
    """
    Extracts lead information using LLM based on per-field scraped data.
    
    :param lead_name: Name of the lead company.
    :param columns_to_retrieve: List of fields to extract.
    :return: Dictionary with extracted lead information.
    """
    # Perform Google searches with keywords
    search_results = perform_google_search(base_query=lead_name, num_results=3, keywords=columns_to_retrieve)
    
    logger.debug(f"Search Results: {search_results}")
    
    total_text = ""
    if isinstance(search_results, dict):
        for urls in search_results.values():
            for url in urls:
                scraped_content = scrape_landing_page(url)
                if scraped_content:
                    total_text += scraped_content + " "
    else:
        for url in search_results:
            scraped_content = scrape_landing_page(url)
            if scraped_content:
                total_text += scraped_content + " "
    
    # Define the prompt template
    prompt = f"""
{prompt_extract_lead_info_with_llm_per_field}

**Lead Entity:** {lead_name}

**Fields to Extract:**
{columns_to_retrieve}

THE TEXT TO ANALYSE:
{total_text}
"""
    response = llm_reasoning(prompt, "gpt-4o-mini")
    try:
        lead_info = json.loads(response)
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error in LLM response for lead '{lead_name}': {e}")
        lead_info = {"Entity": lead_name}
    
    # Initialize 'Employees' as an empty list
    lead_info['Employees'] = []
    lead_info['Source URLs'] = search_results if isinstance(search_results, list) else [url for sublist in search_results.values() for url in sublist]
    lead_info['Type'] = "Company" if 'CEO/PI' in lead_info else "Personnel"
    
    return lead_info


def extract_person_info_with_llm(text, person_name, source_urls, key_words):
    """
    Sends scraped person information to llm for detailed extraction.
    
    :param text: Scraped text content about the person.
    :param person_name: Name of the person.
    :param source_urls: List of URLs where the information was found.
    :param key_words: List of keywords to guide the extraction.
    :return: Dictionary containing detailed person information.
    """
    prompt = f"""
{prompt_extract_person_info_with_llm}

**Person Name:** {person_name}

**Text:**
{text}

**Fields to Extract:**
{key_words}

**Source URLs:**
{source_urls}
"""
    
    try:
        person_info_text = llm_reasoning(prompt, "gpt-4o-mini")
        logger.info(f"Raw LLM Response for Person Information Extraction ('{person_name}'):")
        logger.info(person_info_text)

        # Parse the JSON response
        person_info = json.loads(person_info_text)

        # Ensure that 'Source URLs' is correctly set
        if 'Source URLs' not in person_info or not person_info['Source URLs']:
            person_info['Source URLs'] = source_urls
        else:
            # Merge source URLs from the LLM output and provided source_urls
            person_info['Source URLs'] = list(set(person_info['Source URLs'] + source_urls))

    except json.JSONDecodeError:
        logger.error("Failed to parse the response from LLM while extracting person info.")
        logger.debug(f"Received Response: {person_info_text}")
        person_info = {key: "Not Available" for key in key_words}
        person_info["Source URLs"] = source_urls if source_urls else []
    except Exception as e:
        logger.error(f"An error occurred while extracting person info for '{person_name}': {e}")
        person_info = {key: "Not Available" for key in key_words}
        person_info["Source URLs"] = source_urls if source_urls else []

    # Ensure all required fields are present
    for field in key_words:
        if field not in person_info:
            person_info[field] = "Not Available"

    # Ensure 'Source URLs' is included
    if 'Source URLs' not in person_info:
        person_info['Source URLs'] = source_urls if source_urls else []

    return person_info


def generate_leads_with_llm(context, num_leads):
    """
    Generates a list of leads using LLM based on provided context and lead types.
    
    :param context: Contextual information to guide lead generation.
    :param num_leads: Number of leads to generate.
    :return: List of dictionaries containing generated leads.
    """
    leads = []
    try:
        prompt = f"""
{prompt_generate_leads_with_llm}

Generate exactly {num_leads} unique leads based on the following context:
{context}
There are no more or fewer than {num_leads} leads.
"""
        response = llm_reasoning(prompt, temperature=1.5)
        generated_leads = json.loads(response)

        for lead in generated_leads:
            if "Entity" in lead:
                leads.append({"Entity": lead["Entity"].strip()})
    except Exception as e:
        logger.error(f"Error in generate_leads_with_llm: {e}")

    return leads


def search_url_attendees_exhibitors(url_name, context):
    """
    Performs a Google search to find exhibitors or attendees of the url, using context.
    Returns a list of company names.
    
    :param url_name: Name of the URL/event.
    :param context: Contextual information to guide the extraction.
    :return: List of company names as strings.
    """
    leads_list = []
    try:
        queries = [
            f"{url_name} exhibitor list",
            f"{url_name} floor plan",
            # Additional queries can be uncommented as needed
            # f"{url_name} participants",
            # f"{url_name} sponsors",
            # f"{url_name} partners",
        ]

        for query in queries:
            urls = perform_google_search(query, num_results=3)
            for url in urls:
                logger.info(f"Scraping URL: {url}")
                text = scrape_landing_page(url)
                if text:
                    # Use LLM to extract company names from the text, possibly using context
                    extracted_leads = extract_companies_from_text(text, context)
                    leads_list.extend(extracted_leads)
    except Exception as e:
        logger.error(f"Error searching URL attendees/exhibitors: {e}")
    # Remove duplicates
    leads_list = list(set(leads_list))
    return leads_list


def scrape_url_website(url_url, context):
    """
    Attempts to scrape the URL website to extract exhibitors or attendees, using context.
    Returns a list of dictionaries with the key 'Entity'.
    
    :param url_url: URL of the URL website.
    :param context: Contextual information to guide the extraction (e.g., URL theme).
    :return: List of dictionaries containing company/attendee names under the key 'Entity'.
    """
    leads_list = []
    try:
        logger.info(f"Starting to scrape URL website: {url_url} with context: {context}")
        
        # Fetch the URL website
        response = requests.get(url_url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=10)
        response.raise_for_status()
        html = response.text
        soup = BeautifulSoup(html, 'html.parser')

        # Define keywords to identify relevant sections
        keywords = [
            'exhibitor', 'exhibitors', 'attendee', 'attendees',
            'participant', 'participants', 'sponsor', 'sponsors', 'partner', 'partners'
        ]
        context_keywords = context.lower().split()

        # Find all links containing the keywords and context
        relevant_links = []
        for link in soup.find_all('a', href=True):
            link_text = link.get_text(strip=True).lower()
            link_href = link['href'].lower()
            if any(keyword in link_text or keyword in link_href for keyword in keywords) and \
               any(c_keyword in link_text for c_keyword in context_keywords):
                full_url = requests.compat.urljoin(url_url, link['href'])
                relevant_links.append(full_url)

        # Remove duplicate URLs
        relevant_links = list(set(relevant_links))
        logger.info(f"Found {len(relevant_links)} relevant sub-pages to scrape.")

        # Scrape each relevant link to extract company/attendee names
        for link in relevant_links:
            logger.info(f"Scraping URL sub-page: {link}")
            try:
                page_response = requests.get(link, headers={'User-Agent': 'Mozilla/5.0'}, timeout=10)
                page_response.raise_for_status()
                page_html = page_response.text
                page_soup = BeautifulSoup(page_html, 'html.parser')

                # Extract text from the page
                text = page_soup.get_text(separator=' ', strip=True)

                # Use LLM to extract company names from the text, passing the context
                extracted_leads = extract_companies_from_text(text, context)
                
                # Ensure extracted_leads is a list of strings
                if isinstance(extracted_leads, list):
                    for lead in extracted_leads:
                        if isinstance(lead, str) and lead.strip():
                            leads_list.append({"Entity": lead.strip()})
                else:
                    logger.warning(f"extract_companies_from_text returned non-list type: {type(extracted_leads)}")
            except requests.exceptions.RequestException as e:
                logger.error(f"Error fetching sub-page {link}: {e}")
            except Exception as e:
                logger.error(f"Unexpected error while scraping {link}: {e}")

        # Remove duplicate leads based on 'Entity'
        unique_leads = {lead["Entity"] for lead in leads_list if "Entity" in lead and isinstance(lead["Entity"], str)}
        leads_list = [{"Entity": entity} for entity in unique_leads]
        logger.info(f"Extracted {len(leads_list)} unique leads from URL website.")
    
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching URL website {url_url}: {e}")
    except Exception as e:
        logger.error(f"Unexpected error in scrape_url_website: {e}")
    
    return leads_list


def search_leads_via_url(url_input, context):
    """
    Searches for leads based on the URL name or URL, using context.
    Returns a list of dictionaries with the key 'Entity'.
    
    :param url_input: URL name or URL.
    :param context: Contextual information to guide the search.
    :return: List of dictionaries containing company/attendee names under the key 'Entity'.
    """
    leads_list = []
    try:
        # Check if the input is a URL
        if re.match(r'https?://', url_input):
            # It's a URL, try to scrape it
            leads = scrape_url_website(url_input, context)
        else:
            # It's a URL name, perform a search
            leads = search_url_attendees_exhibitors(url_input, context)
        
        # Ensure all leads are dictionaries with 'Entity' key
        for lead in leads:
            if isinstance(lead, dict) and 'Entity' in lead:
                entity = lead['Entity']
                if isinstance(entity, str) and entity.strip():
                    leads_list.append({"Entity": entity.strip()})
            elif isinstance(lead, str) and lead.strip():
                # If lead is a string, wrap it in a dictionary
                leads_list.append({"Entity": lead.strip()})
            else:
                logger.warning(f"Invalid lead format: {lead}")
        
        # Deduplicate leads based on 'Entity'
        unique_entities = set()
        unique_leads = []
        for lead in leads_list:
            entity = lead.get("Entity")
            if entity and entity not in unique_entities:
                unique_leads.append({"Entity": entity})
                unique_entities.add(entity)
        
        logger.info(f"Total unique leads found via URL: {len(unique_leads)}")
        return unique_leads

    except Exception as e:
        logger.error(f"Error searching leads via URL: {e}")
        return []


def extract_companies_from_text(text, context):
    """
    Uses LLM to extract company names from the provided text, using context.
    Returns a list of company names as strings.
    
    :param text: Text content to analyze.
    :param context: Contextual information to guide the extraction.
    :return: List of company names as strings.
    """
    prompt = f"""
{prompt_extract_companies_from_text}

**Context:**
{context}

**Text:**
{text}
"""
    try:
        response = llm_reasoning(prompt, "gpt-4o-mini")
        companies = json.loads(response)
        if isinstance(companies, list):
            return [company.strip() for company in companies if isinstance(company, str) and company.strip()]
        else:
            logger.error(f"LLM response is not a list: {companies}")
            return []
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON from LLM response: {e}")
        return []
    except Exception as e:
        logger.error(f"Error in extract_companies_from_text: {e}")
        return []


def aggregate_employees_under_companies(leads_info, person_leads):
    """
    Aggregates employee information under their respective companies.

    :param leads_info: List of dictionaries containing lead information.
    :param person_leads: List of dictionaries containing person leads information.
    :return: List of companies with nested employee information.
    """
    companies = {lead['Entity']: lead for lead in leads_info}

    for person in person_leads:
        associated_lead = person.get('Associated Lead')
        if associated_lead and associated_lead in companies:
            if 'Employees' not in companies[associated_lead]:
                companies[associated_lead]['Employees'] = []
            companies[associated_lead]['Employees'].append(person)

    # Convert back to list
    aggregated_leads = list(companies.values())
    return aggregated_leads