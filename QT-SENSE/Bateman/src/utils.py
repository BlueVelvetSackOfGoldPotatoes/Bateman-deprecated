import streamlit as st
import pandas as pd
import json
import re
import subprocess
import os
import requests
import random
import time
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
    
    # Display info boxes for each lead
    for idx, row in df.iterrows():
        if 'Entity' in row and 'Name' in row:
            company_name = row.get('Entity', 'Unknown')
            person_name = row.get('Name', 'Unknown')
            st.info(f"**Entity:** {company_name}")
            st.info(f"**Person Name:** {person_name}")
            # Display other fields as needed
            # Example:
            st.info(f"**Current Position:** {row.get('Current Position', 'Not Available')}")
            st.info(f"**Budget Score:** {row.get('Budget Score', 'Not Available')}")
            st.info(f"**Authority Score:** {row.get('Authority Score', 'Not Available')}")
            st.info(f"**Need Score:** {row.get('Need Score', 'Not Available')}")
            st.info(f"**Timeline Score:** {row.get('Timeline Score', 'Not Available')}")
            st.info(f"**Overall BANT Score:** {row.get('Overall BANT Score', 'Not Available')}")
            st.info(f"**Recommendations:** {row.get('Recommendations', 'Not Available')}")
            # Handle Source URLs
            sources = row.get('Source URLs', '')
            if sources and sources != "Not Available":
                if isinstance(sources, list):
                    formatted_sources = '<br>'.join([f'<a href="{url}" target="_blank">{url}</a>' for url in sources])
                else:
                    formatted_sources = '<br>'.join([f'<a href="{url.strip()}" target="_blank">{url.strip()}</a>' for url in re.split(r';|,', sources) if url.strip()])
                st.markdown(f"**Sources for {person_name}:**<br>{formatted_sources}", unsafe_allow_html=True)
            else:
                st.info(f"No sources available for {person_name}.")
        
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

        summary = llm_reasoning(prompt)
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

def perform_google_search(query, num_results=3, additional_keywords=None):
    """
    Performs a Google search using Serper API and returns a list of URLs.

    :param query: The search query string.
    :param num_results: Number of top results to retrieve.
    :param additional_keywords: List of additional keywords to refine the search.
    :return: List of URLs from the search results.
    """
    try:
        url = "https://google.serper.dev/search"

        if additional_keywords:
            query = f"{query} {' '.join(additional_keywords)}"

        payload = {
            "q": query
        }
        headers = {
            'X-API-KEY': os.environ.get("SERPER_API_KEY"),
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

def scrape_information_field(company_name, field, num_search_results=1):
    """
    Scrapes information for a specific field of a company by performing targeted Google searches.

    :param company_name: Name of the company.
    :param field: Information field to search for (e.g., "Email", "Phone Number").
    :param num_search_results: Number of top Google search results to scrape.
    :return: Dictionary containing aggregated text and source URLs for the field.
    """
    additional_keywords = []
    field_lower = field.lower()
    if field_lower == "email":
        additional_keywords = ["email"] # , "support", "info", "email", "reach us", "get in touch"
    elif field_lower == "phone number":
        additional_keywords = ["contact"] # , "support", "customer service", "call us", "reach us"
    elif field_lower == "researchers":
        additional_keywords = ["scientists"] # , "staff", "employees", "faculty", "members"
    elif field_lower == "grants":
        additional_keywords = ["funding"] # , "projects", "awards", "grants", "sponsorships"
    elif field_lower == "country":
        additional_keywords = ["country"] # , "address", "headquarters", "base", "where we are"
    elif field_lower == "university":
        additional_keywords = ["affiliated university"] # , "institution", "academic affiliation", "campus"
    elif field_lower == "contacts":
        additional_keywords = ["team"] # , "staff", "members", "profiles", "contacts", "directory"
    elif field_lower == "ceo/pi":
        additional_keywords = ["director", "principal investigator", "head of department"] # "leadership", "executive team", 
    elif field_lower == "address":
        additional_keywords = ["headquarters"] # "location", "office address", "base", "where we are"
    elif field_lower == "website":
        additional_keywords = ["homepage"] # "official site", "corporate website", "about us"
    elif field_lower == "social media":
        additional_keywords = ["LinkedIn"] # , "Facebook", "Twitter", "Instagram", "social profiles"
    elif field_lower == "services":
        additional_keywords = ["what we do"] # , "offerings", "services", "solutions", "capabilities"
    elif field_lower == "products":
        additional_keywords = ["our products"] # , "product line", "offerings", "goods", "merchandise"
    elif field_lower == "mission":
        additional_keywords = ["about us"] # , "mission statement", "our mission", "purpose", "vision"
    elif field_lower == "values":
        additional_keywords = ["core values"] # , "our values", "principles", "beliefs"
    # elif field_lower == "careers":
    #     additional_keywords = ["jobs"] # , "employment", "career opportunities", "join us", "work with us"
    else:
        # Default keywords if field is unrecognized
        additional_keywords = ["information", "details", "about", "overview"]


    search_query = f"{company_name} {field}"
    logger.info(f"Performing Google search for: '{search_query}' with keywords: {additional_keywords}")
    urls = perform_google_search(search_query, num_results=num_search_results, additional_keywords=additional_keywords)

    aggregated_text = ""
    source_urls = []

    for url in urls:
        logger.info(f"Scraping URL for field '{field}': {url}")
        text = scrape_landing_page(url)
        if text:
            aggregated_text += text + " "
            source_urls.append(url)
        else:
            logger.warning(f"No text scraped from URL: {url}")

    return {
        "field": field,
        "text": aggregated_text,
        "source_urls": source_urls
    }

def process_chunk(prompt):
    """
    Sends a prompt chunk to OpenAI and retrieves the JSON response.

    :param prompt: The prompt string to send to OpenAI.
    :return: The JSON response as a string, or None if an error occurs.
    """
    try:
        
        lead_info_text = llm_reasoning(prompt)
        logger.info(f"llm Response for Chunk:\n{lead_info_text}")
        return lead_info_text
    except Exception as e:
        logger.error(f"Error processing chunk: {e}")
        return None


def extract_lead_info_with_llm_per_field(lead_name, lead_category, field_data_list, max_tokens=124000, overlap=100):
    """
    Extracts lead information using llm based on per-field scraped data.
    Splits the data into chunks if the prompt exceeds the token limit.
    
    :param lead_name: Name of the lead company.
    :param lead_category: Category of the lead.
    :param field_data_list: List of dictionaries containing field, text, and source URLs.
    :param max_tokens: Maximum number of tokens per OpenAI request.
    :param overlap: Number of tokens to overlap between chunks.
    :return: Dictionary with extracted lead information.
    """
    # Initialize the lead_info dictionary
    lead_info = {
        "Entity": lead_name,
        "CEO/PI": "Not Available",
        "Researchers": [],
        "Grants": [],
        "Phone Number": {"number": "Not Available", "purpose": "Not Available"},
        "Email": {"address": "Not Available", "purpose": "Not Available"},
        "Country": "Not Available",
        "University": "Not Available",
        "Summary": "Not Available",
        "Contacts": [],
        "Category": lead_category,
        "Source URLs": ""
    }

    # Define the prompt template
    prompt_template = f"""
You are an AI assistant specialized in extracting business information. Based on the provided scraped data, extract the following details about the company in JSON format only. Do not include any additional text, explanations, or markdown.

**Lead Entity:** {lead_name}
**Lead Category:** {lead_category}

**Fields to Extract:**
- CEO/PI
- Researchers
- Grants (with Name, Amount, Period, Start Date)
- Phone Number
- Email
- Country
- University
- Summary
- Contacts

**Instructions:**
- Populate each field with the most relevant information extracted from the text.
- If a field's information is not available, set it to "Not Available".
- Ensure the JSON is valid, properly formatted, and includes all specified fields.

**Output Structure:**
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
    ],
    "Source URLs": "https://example.com/contact, https://example.com/about"
}}
"""

    # Aggregate all field data into a single text
    aggregated_field_data = ""
    for field_data in field_data_list:
        field = field_data['field']
        text = field_data['text']
        sources = field_data['source_urls']
        field_prompt = f"\n**Field:** {field}\n**Text:** {text}\n**Source URLs:** {', '.join(sources)}\n"
        aggregated_field_data += field_prompt


    # Split the aggregated_field_data into manageable chunks
    chunks = split_text_into_chunks(aggregated_field_data, max_tokens=max_tokens, overlap=overlap)
    print(chunks)

    responses = []
    for idx, chunk in enumerate(chunks):
        full_prompt = prompt_template + chunk + "\n\nPlease provide the extracted information in JSON format only. Ensure the JSON is valid and properly formatted. Do not include any markdown or additional text."
        response = process_chunk(full_prompt)
        if response:
            responses.append(response)
        else:
            logger.error(f"Chunk {idx+1} processing failed.")

    # Combine all responses into the lead_info dictionary
    for response in responses:
        try:
            partial_info = json.loads(response)  # Expecting JSON format
            for key, value in partial_info.items():
                if key in ["Entity", "Type"]:
                    continue  # Already set
                elif key in ["CEO/PI", "Country", "University", "Summary"]:
                    if lead_info[key] == "Not Available":
                        lead_info[key] = value
                elif key in ["Phone Number", "Email"]:
                    if key == "Email" and lead_info[key]["address"] == "Not Available":
                        lead_info[key]["address"] = value.get("address", "Not Available")
                        lead_info[key]["purpose"] = value.get("purpose", "Not Available")
                    elif key == "Phone Number" and lead_info[key]["number"] == "Not Available":
                        lead_info[key]["number"] = value.get("number", "Not Available")
                        lead_info[key]["purpose"] = value.get("purpose", "Not Available")
                elif key == "Grants":
                    if isinstance(value, list):
                        for grant in value:
                            if isinstance(grant, dict):
                                lead_info["Grants"].append(grant)
                elif key in ["Researchers", "Contacts"]:
                    if isinstance(value, list):
                        lead_info[key].extend([item for item in value if item not in lead_info[key]])
                elif key == "Source URLs":
                    if isinstance(value, list):
                        lead_info["Source URLs"] += "; ".join(value) + "; "
                    elif isinstance(value, str):
                        lead_info["Source URLs"] += value + "; "
        except json.JSONDecodeError:
            logger.error("Failed to parse a partial response from llm.")
            continue

    # Clean up the 'Source URLs' field
    lead_info["Source URLs"] = lead_info["Source URLs"].rstrip("; ")

    # Ensure all required fields are present
    required_fields = ["Entity", "CEO/PI", "Researchers", "Grants",
                       "Phone Number", "Email", "Country", "University",
                       "Summary", "Contacts", "Category", "Source URLs"]
    for field in required_fields:
        if field not in lead_info:
            lead_info[field] = "Not Available"

    return lead_info

def extract_person_info_with_llm(text, person_name, source_urls):
    """
    Sends scraped person information to llm for detailed extraction.
    """
    prompt = f"""
    You are an AI assistant specialized in extracting detailed person profiles from text. Extract the following information about the person and provide it in the specified JSON format only. Do not include any additional text, explanations, or markdown.
    
    **Person Name:** {person_name}
    
    **Text:**
    {text}
    
    **Fields to Extract:**
    - "Name"
    - "Education"
    - "Current Position"
    - "Expertise"
    - "Email"
    - "Phone Number"
    - "Faculty"
    - "University"
    - "Bio"
    - "Academic/Work Website Profile Link"
    - "LinkedIn/Profile Link"
    - "Facebook/Profile Link"
    - "Grant"
    
    **Instructions:**
    - Populate each field with the most relevant information extracted from the text.
    - If a field's information is not available, set it to "Not Available".
    - Ensure the JSON is valid, properly formatted, and includes all specified fields.
    
    **Output:**
    {{
        "Name": "...",
        "Education": "...",
        "Current Position": "...",
        "Expertise": "...",
        "Email": "...",
        "Phone Number": "...",
        "Faculty": "...",
        "University": "...",
        "Bio": "...",
        "Academic/Work Website Profile Link": "...",
        "LinkedIn/Profile Link": "...",
        "Facebook/Profile Link": "...",
        "Grant": "..."
    }}
    """
    sources = []
    try:
        person_info_text = llm_reasoning(prompt)

        logger.info(f"Raw llm Response for Person Information Extraction ('{person_name}'):")
        logger.info(person_info_text)

        # Parse the JSON response
        person_info = json.loads(person_info_text)
        
        # Split the source_urls string into a list if it's a string
        if isinstance(source_urls, str):
            sources = [url.strip() for url in re.split(r';|,', source_urls) if url.strip()]
        elif isinstance(source_urls, list):
            sources = source_urls
        else:
            sources = []
        
        person_info['Source URLs'] = sources
    except json.JSONDecodeError:
        logger.error("Failed to parse the response from llm while extracting person info.")
        logger.debug(f"Received Response: {person_info_text}")
        person_info = {
            "Name": person_name,  # Correctly set to person_name
            "Education": "Not Available",
            "Current Position": "Not Available",
            "Expertise": "Not Available",
            "Email": "Not Available",
            "Phone Number": "Not Available",
            "Faculty": "Not Available",
            "University": "Not Available",
            "Bio": "Not Available",
            "Academic/Work Website Profile Link": "Not Available",
            "LinkedIn/Profile Link": "Not Available",
            "Facebook/Profile Link": "Not Available",
            "Grant": "Not Available",
            "Source URLs": sources
        }
    except Exception as e:
        logger.error(f"An error occurred while extracting person info for '{person_name}': {e}")
        person_info = {
            "Name": person_name,  # Correctly set to person_name
            "Education": "Not Available",
            "Current Position": "Not Available",
            "Expertise": "Not Available",
            "Email": "Not Available",
            "Phone Number": "Not Available",
            "Faculty": "Not Available",
            "University": "Not Available",
            "Bio": "Not Available",
            "Academic/Work Website Profile Link": "Not Available",
            "LinkedIn/Profile Link": "Not Available",
            "Facebook/Profile Link": "Not Available",
            "Grant": "Not Available",
            "Source URLs": sources
        }

    # Ensure all required fields are present
    required_fields = [
        "Name", "Education", "Current Position", "Expertise", "Email",
        "Phone Number", "Faculty", "University", "Bio",
        "Academic/Work Website Profile Link", "LinkedIn/Profile Link",
        "Facebook/Profile Link", "Grant", "Source URLs"
    ]

    for field in required_fields:
        if field not in person_info:
            person_info[field] = "Not Available"

    return person_info


def generate_leads_with_llm(context, num_leads, lead_types):
    """
    Generates a list of leads using llm based on provided context and lead types.
    """
    leads = []
    for lead_type in lead_types:
        prompt = f"""
You are an AI assistant tasked with generating a list of {num_leads} unique and realistic {lead_type.lower()}s that are relevant to the following context:

{context}

Please provide the names, the affiliated University, Country, and City, along with their type in a JSON array. Ensure that the examples exist in the real-world {lead_type.lower()}s. Each {lead_type.lower()} should be distinct and real.

Output:
[
    {{"entity": "Entity Name 1", "university": "University name 1", "city": "city name 1", "country": "country name 1", "type": "{lead_type}"}},
    {{"entity": "Entity Name 2", "university": "University name 2", "city": "city name 2", "country": "country name 2", "type": "{lead_type}"}},
    ...
]
"""
        try:
            names_text = llm_reasoning(prompt)

            logger.info(f"Raw llm Response for Lead Generation ({lead_type}):")
            logger.info(names_text)

            lead_names = json.loads(names_text)
            for lead in lead_names[:num_leads]:
                leads.append({
                    "Entity": lead.get("entity", "Not Available"),
                    "Type": lead.get("type", "Not Available"),
                    "University": lead.get("university", "Not Available"),
                    "City": lead.get("city", "Not Available"),
                    "Country": lead.get("country", "Not Available")
                })
        except json.JSONDecodeError:
            logger.error(f"Failed to parse the response from llm for {lead_type}. Please try again.")
        except Exception as e:
            logger.error(f"An error occurred while generating leads for {lead_type}: {e}")


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
            urls = perform_google_search(query, num_results=1)
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
    Returns a list of tuples (name, type).
    
    :param conference_url: URL of the conference website.
    :param context: Contextual information to guide the extraction (e.g., conference theme).
    :return: List of tuples containing company/attendee names and their types.
    """
    leads_list = []
    try:
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

                # Use llm to extract company names from the text, passing the context
                extracted_leads = extract_companies_from_text(text, context)
                leads_list.extend(extracted_leads)
            except requests.exceptions.RequestException as e:
                logger.error(f"Error fetching sub-page {link}: {e}")
            except Exception as e:
                logger.error(f"Unexpected error while scraping {link}: {e}")

        # Remove duplicate leads
        leads_list = list(set(leads_list))
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching conference website {conference_url}: {e}")
    except Exception as e:
        logger.error(f"Unexpected error in scrape_conference_website: {e}")

    return leads_list

def search_leads_via_conference(conference_input, context):
    """
    Searches for leads based on the conference name or URL, using context.
    Returns a list of tuples (name, type).
    """
    leads_list = []
    # Check if the input is a URL
    if re.match(r'https?://', conference_input):
        # It's a URL, try to scrape it
        leads_list = scrape_conference_website(conference_input, context)
    else:
        # It's a conference name, perform a search
        leads_list = search_conference_attendees_exhibitors(conference_input, context)
    return leads_list

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
        companies_text = llm_reasoning(prompt)
        logger.info(f"Extracted companies: {companies_text}")

        companies = json.loads(companies_text)
        leads_list = [(company.strip(), 'Company') for company in companies if company and len(company.strip()) > 1]
    except Exception as e:
        logger.error(f"Error extracting companies from text: {e}")
    return leads_list