import streamlit as st
import pandas as pd
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

def display_lead_information(df, button_label, filename, height=400):
    """
    Displays the provided DataFrame and provides download options.
    
    :param df: The DataFrame to display.
    :param button_label: The label for the download button.
    :param filename: The base filename for the downloaded file.
    :param height: The height of the displayed DataFrame.
    """
    df = clean_dataframe(df)  # Ensure consistent data types
    st.dataframe(df, height=height)
    
    csv = df.to_csv(index=False).encode('utf-8')
    
    excel_buffer = BytesIO()
    try:
        with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
            df.to_excel(writer, index=False)
        excel_data = excel_buffer.getvalue()
    except Exception as e:
        logger.error(f"Error converting DataFrame to Excel: {e}")
        excel_data = None
    
    st.download_button(
        label=f"Download {button_label} as CSV",
        data=csv,
        file_name=f"{filename}.csv",
        mime='text/csv'
    )
    
    if excel_data:
        st.download_button(
            label=f"Download {button_label} as Excel",
            data=excel_data,
            file_name=f"{filename}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    
    # Separate DataFrames for Companies and Personnel
    companies_df = df[df['Type'] == "Company"]
    personnel_df = df[df['Type'] == "Personnel"]
    
    # Display Companies Information
    if not companies_df.empty:
        st.write("### Companies")
        for idx, row in companies_df.iterrows():
            company_name = row.get('Entity', 'Unknown')
            st.info(f"**Entity:** {company_name}")
            st.info(f"**Category:** {row.get('Category', 'Not Available')}")
            st.info(f"**CEO/PI:** {row.get('CEO/PI', 'Not Available')}")
            st.info(f"**Country:** {row.get('Country', 'Not Available')}")
            st.info(f"**University:** {row.get('University', 'Not Available')}")
            st.info(f"**Summary:** {row.get('Summary', 'Not Available')}")
            st.info(f"**Recommendations:** {row.get('Recommendations', 'Not Available')}")
            
            # Handle Source URLs
            sources = row.get('Source URLs', '')
            if sources and sources != "Not Available":
                if isinstance(sources, list):
                    formatted_sources = '<br>'.join([f'<a href="{url}" target="_blank">{url}</a>' for url in sources])
                else:
                    formatted_sources = '<br>'.join([f'<a href="{url.strip()}" target="_blank">{url.strip()}</a>' for url in re.split(r';|,', sources) if url.strip()])
                st.markdown(f"**Sources for {company_name}:**<br>{formatted_sources}", unsafe_allow_html=True)
            else:
                st.info(f"No sources available for {company_name}.")
            
            st.markdown("---")
    
    # Display Personnel Information
    if not personnel_df.empty:
        st.write("### Personnel")
        for idx, row in personnel_df.iterrows():
            personnel_name = row.get('Personnel Name', 'Unknown')
            st.info(f"**Name:** {personnel_name}")
            st.info(f"**Title:** {row.get('Personnel Title', 'Not Available')}")
            st.info(f"**Email:** {row.get('Personnel Email', 'Not Available')}")
            st.info(f"**Phone:** {row.get('Personnel Phone', 'Not Available')}")
            
            # Link to Associated Company
            company = row.get('Entity', 'Unknown')
            st.info(f"**Associated Company:** {company}")
            
            # Handle Source URLs
            sources = row.get('Source URLs', '')
            if sources and sources != "Not Available":
                if isinstance(sources, list):
                    formatted_sources = '<br>'.join([f'<a href="{url}" target="_blank">{url}</a>' for url in sources])
                else:
                    formatted_sources = '<br>'.join([f'<a href="{url.strip()}" target="_blank">{url.strip()}</a>' for url in re.split(r';|,', sources) if url.strip()])
                st.markdown(f"**Sources for {personnel_name}:**<br>{formatted_sources}", unsafe_allow_html=True)
            else:
                st.info(f"No sources available for {personnel_name}.")
            
            st.markdown("---")

def split_text_into_chunks(text, max_tokens=124000, overlap=100):
    """
    Splits text into chunks with a maximum number of tokens, allowing for overlap between chunks.
    
    :param text: The input text to split.
    :param max_tokens: The maximum number of tokens per chunk.
    :param overlap: The number of tokens to overlap between chunks.
    :return: A list of text chunks.
    """
    tokens = tiktoken.get_encoding(os.environ.get("ENCODING")).encode(text)
    
    chunks = []
    start = 0
    end = max_tokens
    
    while start < len(tokens):
        chunk_tokens = tokens[start:end]
        chunk_text = tiktoken.get_encoding(os.environ.get("ENCODING")).decode(chunk_tokens)
        chunks.append(chunk_text)
        start = end - overlap  # Move back by 'overlap' tokens for the next chunk
        end = start + max_tokens
    
    return chunks

def summarize_paper(paper_title, paper_abstract):
    """
    Generates a summary for a given paper abstract using llm.
    """
    try:
        prompt = f"Provide a summary for the following paper. Together with a summary of the overall paper, be sure to inlude the methods used, what was the experiment, what technologies were used, what was the result, and what were the hypotheses as well as the relevant field. \n\nTitle: {paper_title}\n\nAbstract: {paper_abstract}\n\nSummary:"

        summary = llm_reasoning(prompt, "gpt-4o-mini")
        return summary
    except Exception as e:
        logger.error(f"Error summarizing paper '{paper_title}': {e}")
        return "Summary not available."


def clean_text(text):
    """
    Cleans and normalizes text data.
    """
    return re.sub(r'\s+', ' ', text).strip()


def count_tokens(text):
    """
    Counts the number of tokens in a given text using the specified encoding.

    :param text: The text to count tokens for.
    :param encoding_name: The name of the encoding to use.
    :return: The number of tokens.
    """
    return len(tiktoken.get_encoding(os.environ.get("ENCODING")).encode(text))


def extract_persons(leads_info_df):
    """
    Extracts unique person names associated with leads from the scraped data.
    Processes both 'Contacts' and 'Researchers' fields.
    """
    persons = []
    for _, row in leads_info_df.iterrows():
        lead_entity = row.get('Entity', 'Unknown')
        
        # Process Contacts
        contacts = row.get('Contacts', [])
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
        researchers = row.get('Researchers', [])
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


def download_leads(leads_df, excel_filename):
    """
    Facilitates the download of DataFrames as Excel files.
    """
    try:
        leads_df = leads_df.fillna('').astype(str)
        excel_buffer = BytesIO()
        with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
            leads_df.to_excel(writer, index=False)
        excel_buffer.seek(0)
        unique_key = f"download_final_{excel_filename}"
        st.download_button(
            label="Download Excel File",
            data=excel_buffer,
            file_name=f"{excel_filename}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key=unique_key
        )
    except Exception as e:
        logger.error(f"Error while downloading Excel file: {e}")


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

    # Parse HTML and clean text content
    try:
        soup = BeautifulSoup(html, 'html.parser')

        # Remove non-visible elements
        for element in soup(["script", "style", "head", "meta", "noscript", "iframe"]):
            element.extract()

        # Get visible text
        text = soup.get_text(separator=' ')

        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        # Filter out non-informational patterns
        non_informational_patterns = [
            r'accept cookies', r'terms of service', r'privacy policy', r'navigation',
            r'back to top', r'© [\d]{4}', r'all rights reserved'
        ]
        text = re.sub('|'.join(non_informational_patterns), '', text, flags=re.IGNORECASE)

        return text
    except Exception as parse_error:
        logger.error(f"Error parsing HTML for {url}: {parse_error}")
        return ""


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
                if keyword.strip() == "":
                    # Handle empty keyword as general search
                    query = base_query
                else:
                    # Build query: base_query + keyword
                    query = f"{base_query} + {keyword}"

                logger.debug(f"Final search query for keyword '{keyword}': {query}")

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


def convert_lists_to_strings(df):
    """
    Converts all list entries in the DataFrame to comma-separated strings.
    
    :param df: The pandas DataFrame to process.
    :return: The processed DataFrame with lists converted to strings.
    """
    for column in df.columns:
        df[column] = df[column].apply(
            lambda x: ', '.join(map(str, x)) if isinstance(x, list) else json.dumps(x) if isinstance(x, dict) else x
        )
        df[column] = df[column].fillna("Not Available").astype(str)
    return df

def clean_dataframe(df):
    """
    Expands columns with dictionary (struct) types into separate columns.
    Also ensures consistent data types across all columns by converting lists to strings.
    """
    # Expand Phone Number
    if 'Phone Number' in df.columns:
        phone_df = df['Phone Number'].apply(
            lambda x: x if isinstance(x, dict) else 
                      {"number": x, "purpose": "Not Available"} if isinstance(x, str) else 
                      {"number": "Not Available", "purpose": "Not Available"}
        )
        phone_expanded = pd.json_normalize(phone_df)
        phone_expanded = phone_expanded.add_prefix('Phone Number ')
        df = pd.concat([df.drop(columns=['Phone Number']), phone_expanded], axis=1)
    
    # Expand Email
    if 'Email' in df.columns:
        email_df = df['Email'].apply(
            lambda x: x if isinstance(x, dict) else 
                      {"address": x, "purpose": "Not Available"} if isinstance(x, str) else 
                      {"address": "Not Available", "purpose": "Not Available"}
        )
        email_expanded = pd.json_normalize(email_df)
        email_expanded = email_expanded.add_prefix('Email ')
        df = pd.concat([df.drop(columns=['Email']), email_expanded], axis=1)
    
    # Convert all lists in the DataFrame to comma-separated strings
    df = convert_lists_to_strings(df)
    
    return df


def extract_lead_info_with_llm_per_field(lead_name, columns_to_retrieve):
    """
    Extracts lead information using llm based on per-field scraped data.
    Splits the data into chunks if the prompt exceeds the token limit.
    
    :param lead_name: Name of the lead company.
    :return: Dictionary with extracted lead information.
    """

    # Define the keywords for different searches
    search_keywords = ["about", "team"]  # "" for general homepage, "team" for personnel

    # Perform Google searches with keywords
    search_results = perform_google_search(base_query=lead_name, num_results=2, keywords=search_keywords)

    logger.debug(f"Search Results: {search_results}")

    # Extract URLs from search results
    homepage_urls = search_results.get("about", [])  # URLs from the general search

    personnel_urls = search_results.get("team", [])  # URLs from the "team" search

    # Safely access the first URL if available - this is wrong and should actually visit all urls and concatenate them - but this needs chunking and what not...
    if isinstance(homepage_urls, list) and len(homepage_urls) > 0:
        homepage_text = scrape_landing_page(homepage_urls[0])
    else:
        logger.warning(f"No homepage URLs found for lead '{lead_name}'.")
        homepage_text = ""

    if isinstance(personnel_urls, list) and len(personnel_urls) > 0:
        personnel_text = scrape_landing_page(personnel_urls[0])
    else:
        logger.warning(f"No personnel URLs found for lead '{lead_name}'.")
        personnel_text = ""
    
    total_text = homepage_text + "\n" + personnel_text

    # Define the prompt template
    prompt = f"""
You are an AI assistant specialized in extracting business information. Based on the provided scraped data, extract the following details about the company in JSON format only. Do not include any additional text, explanations, or markdown.

**Lead Entity:** {lead_name}

**Fields to Extract:**
{columns_to_retrieve}

**Instructions:**
- Populate each field with the most relevant information extracted from the text.
- If a field's information is not available, set it to "Not Available".
- Ensure the JSON is valid, properly formatted, and includes all specified fields.

**Output Structure Example (fields may differ):**
{{
    "Entity": "Genentech",
    "Category": "Biotechnology Company",
    "CEO/PI": "Ashley Magargee",
    "Researchers": [
        "Aaron Wecksler",
        "Adeyemi Adedeji",
        ...
    ],
    "Grants": [
        {{
            "Name": "Cancer Research Initiative",
            "Amount": "$500,000",
            "Period": "2 years",
            "Start Date": "2022-01-15"
        }},
        ...
    ],
    "Phone Number": {{
        "number": "(650) 225-1000",
        "purpose": "Corporate Office"
    }},
    "Email": {{
        "address": "patientinfo@gene.com",
        "purpose": "Patient Resource Center"
    }},
    "Country": "USA",
    "University": "Stanford University",
    "Summary": "Genentech is a leading biotechnology company focused on developing medicines for serious medical conditions.",
    "Contacts": [
        {{
            "Name": "Ashley Magargee",
            "Title": "Chief Executive Officer"
        }},
        ...
    ]
}}

===========
THE TEXT TO ANALYSE:
{total_text}
"""
    response = llm_reasoning(prompt, "gpt-4o-mini")
    lead_info = json.loads(response)

    return lead_info


def extract_person_info_with_llm(text, person_name, source_urls, key_words):
    """
    Sends scraped person information to llm for detailed extraction.
    """
    prompt = f"""
    You are an AI assistant specialized in extracting detailed person profiles from text. Extract the following information about the person and provide it in the specified JSON format only. Do not include any additional text, explanations, or markdown.

    
    **Person Name:** {person_name}
    
    **Text:**
    {text}
    
    **Fields to Extract:**
    {key_words}
    
    ** Source URLs **
    {source_urls}

    **Instructions:**
    - Populate each field with the most relevant information extracted from the text.
    - If a field's information is not available, set it to "Not Available".
    - Ensure the JSON is valid, properly formatted, and includes all specified keyword fields.
    
    **Output:**
    {{
        "keyword 1": "...",
        "keyword 2": "...",
    }}
    """
    try:
        person_info_text = llm_reasoning(prompt, "gpt-4o-mini")
        logger.info(f"Raw llm Response for Person Information Extraction ('{person_name}'):")
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
        logger.error("Failed to parse the response from llm while extracting person info.")
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
    Generates a list of leads using llm based on provided context and lead types.
    """
    leads = []
    try:
        prompt = f"""
    Generate exactly {num_leads} unique leads based on the following context:
    {context}

    Output the leads as a JSON array with each lead in the following format:
    [
        {{"Entity": "Entity Name 1"}},
        {{"Entity": "Entity Name 2"}},
        ...
    ]
    Ensure that:
    - There are no more or fewer than {num_leads} leads.
    - The JSON syntax is correct with proper commas and brackets.
    - Each lead object contains only the "Entity" key with a string value.

    Output as a JSON array:
    [
        {{"Entity": "Entity Name 1"}},
        {{"Entity": "Entity Name 2"}}
    ]
    """
        response = llm_reasoning(prompt, t = 1.5)
        generated_leads = json.loads(response)

        for lead in generated_leads:
            if "Entity" in lead:
                leads.append({"Entity": lead["Entity"].strip()})
    except Exception as e:
        logger.error(f"Error in generate_leads_with_llm: {e}")

    return leads


def search_conference_attendees_exhibitors(conference_name, context):
    """
    Performs a Google search to find exhibitors or attendees of the conference, using context.
    Returns a list of tuples (name, type).
    """
    leads_list = []
    try:
        queries = [
            f"{conference_name} exhibitor list",
            f"{conference_name} floor plan",
            # f"{conference_name} participants",
            # f"{conference_name} sponsors",
            # f"{conference_name} partners",
        ]

        for query in queries:
            urls = perform_google_search(query, num_results=3)
            for url in urls:
                logger.info(f"Scraping URL: {url}")
                text = scrape_landing_page(url)
                if text:
                    # Use llm to extract company names from the text, possibly using context
                    extracted_leads = extract_companies_from_text(text, context)
                    leads_list.extend(extracted_leads)
    except Exception as e:
        logger.error(f"Error searching conference attendees/exhibitors: {e}")
    # Remove duplicates
    leads_list = list(set(leads_list))
    return leads_list


def scrape_conference_website(conference_url, context):
    """
    Attempts to scrape the conference website to extract exhibitors or attendees, using context.
    Returns a list of dictionaries with the key 'Entity'.
    
    :param conference_url: URL of the conference website.
    :param context: Contextual information to guide the extraction (e.g., conference theme).
    :return: List of dictionaries containing company/attendee names under the key 'Entity'.
    """
    leads_list = []
    try:
        logger.info(f"Starting to scrape conference website: {conference_url} with context: {context}")
        
        # Fetch the conference website
        response = requests.get(conference_url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=10)
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
                full_url = requests.compat.urljoin(conference_url, link['href'])
                relevant_links.append(full_url)

        # Remove duplicate URLs
        relevant_links = list(set(relevant_links))
        logger.info(f"Found {len(relevant_links)} relevant sub-pages to scrape.")

        # Scrape each relevant link to extract company/attendee names
        for link in relevant_links:
            logger.info(f"Scraping conference sub-page: {link}")
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
        logger.info(f"Extracted {len(leads_list)} unique leads from conference website.")
    
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching conference website {conference_url}: {e}")
    except Exception as e:
        logger.error(f"Unexpected error in scrape_conference_website: {e}")
    
    return leads_list


def search_leads_via_conference(conference_input, context):
    """
    Searches for leads based on the conference name or URL, using context.
    Returns a list of dictionaries with the key 'Entity'.
    
    :param conference_input: Conference name or URL.
    :param context: Contextual information to guide the search.
    :return: List of dictionaries containing company/attendee names under the key 'Entity'.
    """
    leads_list = []
    try:
        # Check if the input is a URL
        if re.match(r'https?://', conference_input):
            # It's a URL, try to scrape it
            leads = scrape_conference_website(conference_input, context)
        else:
            # It's a conference name, perform a search
            leads = search_conference_attendees_exhibitors(conference_input, context)
        
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
        
        logger.info(f"Total unique leads found via conference: {len(unique_leads)}")
        return unique_leads

    except Exception as e:
        logger.error(f"Error searching leads via conference: {e}")
        return []


def extract_companies_from_text(text, context):
    """
    Uses llm to extract company names from the provided text, using context.
    Returns a list of tuples (name, type).
    """
    leads_list = []
    try:
        prompt = f"""
You are an AI assistant that extracts company names from the following text. The companies are likely to be exhibitors, attendees, sponsors, or partners of a conference related to the context provided.

**Context:**
{context}

**Text:**
{text}

Please provide the extracted information in JSON format only. Ensure the JSON is valid and properly formatted. Do not include any markdown or additional text.

Example:
["Company A", "Company B", "Company C"]
"""
        response = llm_reasoning(prompt, "gpt-4o-mini")
        companies = json.loads(response)
        return list(set(company.strip() for company in companies if company.strip()))
    except Exception as e:
        logger.error(f"Error in extract_companies_from_text: {e}")
        return []