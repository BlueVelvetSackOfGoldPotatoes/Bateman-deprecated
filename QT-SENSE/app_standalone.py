import streamlit as st
import pandas as pd
import json
import uuid
import re
import time
import os
import requests
import random
import numpy as np
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import ElementNotInteractableException, TimeoutException, NoSuchElementException
from webdriver_manager.chrome import ChromeDriverManager
from fake_useragent import UserAgent
from bs4 import BeautifulSoup
import openai
from openai import OpenAI
from streamlit_tags import st_tags
from selenium.webdriver.chrome.service import Service
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import tiktoken
from io import BytesIO
import logging
from dotenv import load_dotenv

# Add these imports:
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

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

client = OpenAI(api_key=os.environ.get("OPENAI_KEY"))

# ==========================
# Define Utility Functions
# ==========================

def count_tokens(text, encoding_name='cl100k_base'):
    """
    Counts the number of tokens in a given text using the specified encoding.
    """
    encoding = tiktoken.get_encoding(encoding_name)
    return len(encoding.encode(text))

def clean_dataframe(df):
    """
    Converts list-type columns into JSON strings for better display and download.
    """
    for column in df.columns:
        if df[column].apply(lambda x: isinstance(x, list)).any():
            df[column] = df[column].apply(lambda x: json.dumps(x) if isinstance(x, list) else x)
    return df

def display_and_download(df, button_label, filename, height=400):
    """
    Displays a DataFrame and provides options to download it as CSV or Excel.
    """
    df = clean_dataframe(df)
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

def clean_response(response_text):
    """
    Cleans the GPT response to extract valid JSON.
    """
    response_text = response_text.strip()
    response_text = re.sub(r"^'''json", "", response_text, flags=re.IGNORECASE)
    response_text = re.sub(r"'''$", "", response_text, flags=re.IGNORECASE)
    if response_text.lower().startswith('json'):
        response_text = response_text[4:].strip()
    match = re.search(r'(\[.*\]|\{.*\})', response_text, re.DOTALL)
    return match.group(1) if match else response_text

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

def extract_lead_info_with_gpt(text, columns_to_retrieve, lead_name, lead_category, source_urls):
    """
    Sends scraped lead information to GPT for extraction based on user-specified fields.
    """
    encoding = 'cl100k_base'
    model_max_tokens = 128000

    initial_prompt = f"""
You are an AI assistant that extracts the following information from the given text:

[{', '.join([f'"{col}"' for col in columns_to_retrieve])}]

Lead Name: {lead_name}
Lead Category: {lead_category}

Text:
"""

    tokens_initial = count_tokens(initial_prompt, encoding)
    tokens_text = count_tokens(text, encoding)

    available_tokens = model_max_tokens - tokens_initial - 1000

    avg_words_per_token = 0.75
    max_words = int(available_tokens * avg_words_per_token)

    words = text.split()
    if len(words) > max_words:
        text = ' '.join(words[:max_words])
        logger.info(f"Truncated text for '{lead_name}' to {max_words} words.")

    # Dynamically construct the output format
    output_example = "{\n"
    for col in columns_to_retrieve:
        if col in ['Researchers', 'University', 'Grant Received']:  # List fields
            output_example += f'    "{col}": ["..."],\n'
        elif col == 'Contacts':
            output_example += '''    "Contacts": [
        {
            "Name": "...",
            "Email": "...",
            "Phone Number": "..."
        },
        ...
    ],\n'''
        else:
            output_example += f'    "{col}": "...",\n'
    output_example += '    "Source URLs": "..."'  # Include Source URLs
    output_example += '\n}'

    prompt = f"""
{initial_prompt}
{text}

Please provide the extracted information in JSON format only. Ensure that all list fields contain only strings and that contacts are connected to specific individuals. If certain information is not available, use "Not Available" or an empty list.

Output:
{output_example}
"""

    total_tokens = count_tokens(prompt, encoding)
    if total_tokens > model_max_tokens:
        logger.error(f"Lead info for '{lead_name}' exceeds the maximum token limit even after truncation. Skipping this lead.")
        lead_info = {col: "Not Available" for col in columns_to_retrieve}
        lead_info["Company/Group Name"] = lead_name
        lead_info["Category"] = lead_category
        lead_info["Source URLs"] = source_urls
        return lead_info

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )

        lead_info_text = clean_response(response.choices[0].message.content)

        logger.info(f"Raw GPT-4o Response for Lead Information Extraction ('{lead_name}'):")
        logger.info(lead_info_text)

        lead_info = json.loads(lead_info_text)
        lead_info['Source URLs'] = source_urls
        lead_info['Category'] = lead_category  # Ensure 'Category' is included
    except json.JSONDecodeError:
        logger.error("Failed to parse the response from GPT while extracting lead info.")
        logger.debug(f"Received Response: {lead_info_text}")
        lead_info = {col: "Not Available" for col in columns_to_retrieve}
        lead_info["Company/Group Name"] = lead_name
        lead_info["Category"] = lead_category
        lead_info["Source URLs"] = source_urls
    except Exception as e:
        logger.error(f"An error occurred while extracting lead info for '{lead_name}': {e}")
        lead_info = {col: "Not Available" for col in columns_to_retrieve}
        lead_info["Company/Group Name"] = lead_name
        lead_info["Category"] = lead_category
        lead_info["Source URLs"] = source_urls

    # Ensure all required columns are present
    for col in columns_to_retrieve:
        if col not in lead_info:
            lead_info[col] = "Not Available"

    return lead_info

def extract_person_info_with_gpt(text, person_name, source_urls):
    """
    Sends scraped person information to GPT for detailed extraction.
    """
    encoding = 'cl100k_base'
    model_max_tokens = 128000

    prompt = f"""
You are an AI assistant that extracts detailed information about a person from the given text.

Person Name: {person_name}

Text:
{text}

Please provide the extracted information in JSON format only. Include the following fields:
- "Name"
- "Education"
- "Current Position"
- "Work Area"
- "Hobbies"
- "Email"
- "Phone Number"
- "Address"
- "Faculty"
- "University"
- "Bio"
- "Profile Link"

Ensure that all fields are filled accurately based on the provided text. If certain information is not available, use "Not Available".

Output:
{{
    "Name": "",
    "Education": "",
    "Current Position": "",
    "Work Area": "",
    "Hobbies": "",
    "Email": "",
    "Phone Number": "",
    "Address": "",
    "Faculty": "",
    "University": "",
    "Bio": "",
    "Profile Link": ""
}}
"""

    total_tokens = count_tokens(prompt, encoding)
    if total_tokens > model_max_tokens:
        logger.error(f"Person info for '{person_name}' exceeds the maximum token limit. Skipping this person.")
        return {
            "Name": person_name,
            "Education": "Not Available",
            "Current Position": "Not Available",
            "Work Area": "Not Available",
            "Hobbies": "Not Available",
            "Email": "Not Available",
            "Phone Number": "Not Available",
            "Address": "Not Available",
            "Faculty": "Not Available",
            "University": "Not Available",
            "Bio": "Not Available",
            "Profile Link": "Not Available",
            "Source URLs": source_urls
        }

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )

        person_info_text = clean_response(response.choices[0].message.content)

        logger.info(f"Raw GPT-4o Response for Person Information Extraction ('{person_name}'):")
        logger.info(person_info_text)

        person_info = json.loads(person_info_text)
        person_info['Source URLs'] = source_urls
    except json.JSONDecodeError:
        logger.error("Failed to parse the response from GPT while extracting person info.")
        logger.debug(f"Received Response: {person_info_text}")
        person_info = {
            "Name": person_name,
            "Education": "Not Available",
            "Current Position": "Not Available",
            "Work Area": "Not Available",
            "Hobbies": "Not Available",
            "Email": "Not Available",
            "Phone Number": "Not Available",
            "Address": "Not Available",
            "Faculty": "Not Available",
            "University": "Not Available",
            "Bio": "Not Available",
            "Profile Link": "Not Available",
            "Source URLs": source_urls
        }
    except Exception as e:
        logger.error(f"An error occurred while extracting person info for '{person_name}': {e}")
        person_info = {
            "Name": person_name,
            "Education": "Not Available",
            "Current Position": "Not Available",
            "Work Area": "Not Available",
            "Hobbies": "Not Available",
            "Email": "Not Available",
            "Phone Number": "Not Available",
            "Address": "Not Available",
            "Faculty": "Not Available",
            "University": "Not Available",
            "Bio": "Not Available",
            "Profile Link": "Not Available",
            "Source URLs": source_urls
        }

    # Ensure all required fields are present
    required_fields = ["Name", "Education", "Current Position", "Work Area", "Hobbies",
                       "Email", "Phone Number", "Address", "Faculty", "University", "Bio", "Profile Link"]
    for field in required_fields:
        if field not in person_info:
            person_info[field] = "Not Available"

    return person_info

def generate_leads_with_gpt(context, num_leads, lead_types):
    """
    Generates a list of leads using GPT based on provided context and lead types.
    """
    leads = []
    for lead_type in lead_types:
        prompt = f"""
You are an AI assistant tasked with generating a list of {num_leads} unique and realistic {lead_type.lower()}s that are relevant to the following context:

{context}

Please provide the names along with their type in a JSON array. Ensure that the names are real-world {lead_type.lower()}s and avoid generic terms or scientific concepts. Each {lead_type.lower()} should be distinct and plausible.

Output:
[
    {{"name": "Name 1", "type": "{lead_type}"}},
    {{"name": "Name 2", "type": "{lead_type}"}},
    ...
]
"""
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2
            )

            names_text = clean_response(response.choices[0].message.content)

            logger.info(f"Raw GPT-4o Response for Lead Generation ({lead_type}):")
            logger.info(names_text)

            lead_names = json.loads(names_text)
            leads.extend([(lead['name'], lead['type']) for lead in lead_names[:num_leads]])
        except json.JSONDecodeError:
            logger.error(f"Failed to parse the response from GPT for {lead_type}. Please try again.")
        except Exception as e:
            logger.error(f"An error occurred while generating leads for {lead_type}: {e}")

    return leads

def rank_leads(leads_df, weights):
    """
    Ranks leads based on configurable weights for different criteria.
    """
    current_year = time.localtime().tm_year

    def calculate_score(row):
        score = 0
        # Parse Contacts
        contacts = row.get('Contacts')
        if pd.notnull(contacts):
            if isinstance(contacts, str):
                try:
                    contacts = json.loads(contacts)
                except json.JSONDecodeError:
                    contacts = []
            elif isinstance(contacts, list):
                pass
            else:
                contacts = []
        else:
            contacts = []

        # Email
        has_email = False
        for contact in contacts:
            email = contact.get('Email')
            if email and email.strip().lower() not in ['not provided', 'not available', '']:
                has_email = True
                break
        if has_email:
            score += weights['email']

        # Phone Number
        has_phone = False
        for contact in contacts:
            phone = contact.get('Phone Number')
            if phone and phone.strip().lower() not in ['not provided', 'not available', '']:
                has_phone = True
                break
        if has_phone:
            score += weights['phone']

        # Grants Received
        grant_received = row.get('Grant Received')
        grant_count = 0
        if pd.notnull(grant_received):
            if isinstance(grant_received, list):
                grant_count = len(grant_received)
            elif isinstance(grant_received, str) and grant_received.strip():
                grant_count = 1
        score += weights['grants'] * grant_count

        # Date
        date_value = row.get('Date')
        if pd.notnull(date_value) and str(date_value).strip():
            try:
                years = re.findall(r'\b(19|20)\d{2}\b', str(date_value))
                if years:
                    year = int(years[0])
                    year_score = (year - 2000) / (current_year - 2000)
                    year_score = max(0, min(year_score, 1))
                    score += weights['date'] * year_score
            except:
                pass  # Ignore invalid date formats

        return score

    try:
        leads_df['Ranking'] = leads_df.apply(calculate_score, axis=1)
        # Sort the DataFrame by the 'Ranking' column in descending order
        leads_df = leads_df.sort_values(by='Ranking', ascending=False).reset_index(drop=True)
    except Exception as e:
        logger.error(f"An error occurred while ranking leads: {e}")
        # If ranking fails, return the original DataFrame without ranking
    return leads_df

def clean_text(text):
    """
    Cleans and normalizes text data.
    """
    return re.sub(r'\s+', ' ', text).strip()

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

# ==========================
# Initialize Session State
# ==========================

if 'step' not in st.session_state:
    st.session_state['step'] = 1
if 'leads' not in st.session_state:
    st.session_state['leads'] = []
if 'leads_df' not in st.session_state:
    st.session_state['leads_df'] = pd.DataFrame()
if 'leads_info_df' not in st.session_state:
    st.session_state['leads_info_df'] = pd.DataFrame()
if 'ranked_leads_df' not in st.session_state:
    st.session_state['ranked_leads_df'] = pd.DataFrame()
if 'person_leads_df' not in st.session_state:
    st.session_state['person_leads_df'] = pd.DataFrame()
if 'excel_filename' not in st.session_state:
    st.session_state['excel_filename'] = "leads"

# ==========================
# Define Section Functions
# ==========================

def input_leads():
    """
    Section for Input Leads: Generate Leads or Add Leads Manually.
    """
    st.subheader("Step 1: Input Leads")

    option = st.radio("Choose how to input leads:", ["Generate Leads", "Add Leads Manually"], key='lead_input_option')

    if option == "Generate Leads":
        context = st.text_area("Context used for search and filtering:", height=150, key='context_input')
        num_leads_total = st.number_input("Number of leads per type:", min_value=1, max_value=100, value=10, step=1, key='num_leads_total')
        default_lead_types = ["Companies", "Research Groups"]
        lead_types = st_tags(
            label='',
            text='Add or remove lead types:',
            value=default_lead_types,
            suggestions=default_lead_types,
            maxtags=10,
            key='lead_types'
        )
        generate_btn = st.button("Generate Leads", key='generate_leads_btn')
        if generate_btn:
            if not context.strip():
                logger.error("Lead generation attempted with empty context.")
                st.warning("Please provide a context for lead generation.")
            elif not lead_types:
                logger.error("Lead generation attempted without specifying lead types.")
                st.warning("Please specify at least one lead type.")
            else:
                with st.spinner('Generating leads...'):
                    leads = generate_leads_with_gpt(context, num_leads_total, lead_types)
                if leads:
                    # Append to the existing leads list
                    st.session_state['leads'].extend(leads)
                    
                    # Create a DataFrame from the new leads
                    new_leads_df = pd.DataFrame(leads, columns=["Name", "Type"])
                    
                    # Append to the existing leads_df DataFrame
                    st.session_state['leads_df'] = pd.concat([st.session_state['leads_df'], new_leads_df], ignore_index=True)
                    
                    # Update the 'step' if necessary
                    st.session_state['step'] = max(st.session_state['step'], 2)
                    
                    st.success(f"Leads generated successfully! Total leads: {len(st.session_state['leads'])}")
                    logger.info(f"Generated {len(leads)} leads.")
                else:
                    logger.error("Lead generation failed.")
                    st.error("Failed to generate leads. Please check the logs for more details.")

    if option == "Add Leads Manually":
        st.write("Enter your leads below:")

        # Remove the 'value' parameter and rely on 'key' to manage state
        leads_input = st.text_area(
            "Enter one lead per line, in the format 'Name,Type':",
            height=150,
            key='manual_leads_input'
        )

        add_leads_btn = st.button("Add Leads", key='add_leads_btn')
        if add_leads_btn:
            leads_list = []
            for line in leads_input.strip().split('\n'):
                parts = line.strip().split(',')
                if len(parts) == 2:
                    name = parts[0].strip()
                    lead_type = parts[1].strip()
                    leads_list.append((name, lead_type))
                else:
                    logger.warning(f"Invalid format in line: {line}")
                    st.warning(f"Invalid format in line: {line}. Please use 'Name,Type'.")
            if leads_list:
                # Append to the existing leads list
                st.session_state['leads'].extend(leads_list)
                
                # Append to the existing leads_df DataFrame
                new_leads_df = pd.DataFrame(leads_list, columns=["Name", "Type"])
                st.session_state['leads_df'] = pd.concat([st.session_state['leads_df'], new_leads_df], ignore_index=True)
                
                # Update the 'step' if necessary
                st.session_state['step'] = max(st.session_state['step'], 2)
                
                st.success(f"Leads added successfully! Total leads: {len(st.session_state['leads'])}")
                logger.info(f"Added {len(leads_list)} leads manually.")
            else:
                logger.error("No valid leads entered.")
                st.error("No valid leads entered. Please ensure each line is in the format 'Name,Type'.")

    # Display the Leads DataFrame if it exists
    if not st.session_state['leads_df'].empty:
        st.write("### Leads")
        st.write("You can edit the leads below:")

        try:
            # Create a temporary DataFrame to work with
            temp_leads_df = st.session_state['leads_df'].copy()

            # Use the data editor on the temporary DataFrame
            edited_leads_df = st.data_editor(
                temp_leads_df,
                num_rows="dynamic",
                key='leads_editor'
            )

            # Add a 'Save Changes' button to commit the edits
            if st.button('Save Changes', key='save_leads_changes'):
                # Update the session state with the edited leads
                st.session_state['leads_df'] = edited_leads_df

                # Synchronize the 'leads' list with the updated DataFrame
                st.session_state['leads'] = list(zip(edited_leads_df['Name'], edited_leads_df['Type']))

                st.success("Leads updated successfully!")
                logger.info("Leads edited via data editor.")

            # Display the edited DataFrame
            display_and_download(
                df=edited_leads_df,
                button_label="Leads",
                filename="leads"
            )
        except Exception as e:
            logger.error(f"Error editing leads: {e}")
            st.error(f"Error editing leads: {e}")

def scrape_lead_information():
    """
    Section for Scrape Lead Information: Search and Scrape Lead Information.
    """
    st.subheader("Step 2: Search and Scrape Lead Information")
    default_columns = ["Company/Group Name", "CEO/PI", "Researchers", "Grant Received", "Date", "Country", "University", "Email", "Phone Number", "Summary", "Contacts"]
    columns_to_retrieve = st_tags(
        label='',
        text='Add or remove information fields:',
        value=default_columns,
        suggestions=default_columns,
        maxtags=20,
        key='columns_to_retrieve'
    )
    search_btn = st.button("Search and Scrape Leads", key='search_leads_btn')
    if search_btn:
        if not st.session_state['leads']:
            logger.error("Lead scraping attempted without any generated or added leads.")
            st.error("No leads available. Please add or generate leads first.")
        elif not columns_to_retrieve:
            logger.error("Lead scraping attempted without specifying information fields.")
            st.error("Please select at least one information field to retrieve.")
        else:
            with st.spinner('Searching and scraping lead information...'):
                leads_info = []
                # Placeholder to display current lead being processed
                lead_placeholder = st.empty()
                for idx, (lead_name, lead_category) in enumerate(st.session_state['leads']):
                    lead_placeholder.text(f"Processing Lead {idx+1}/{len(st.session_state['leads'])}: {lead_name}")
                    urls = perform_google_search(lead_name)
                    if not urls:
                        logger.warning(f"No URLs found for '{lead_name}'.")
                        continue
                    scraped_text = ""
                    sources = []
                    for url in urls[:3]:
                        # Display the URL being scraped
                        logger.info(f"Scraping URL for '{lead_name}': {url}")
                        with st.spinner(f"Scraping URL for '{lead_name}': {url}"):
                            text = scrape_landing_page(url)
                            if text:
                                scraped_text += text + " "
                                sources.append(url)
                    if not scraped_text.strip():
                        logger.warning(f"No text scraped from URLs for '{lead_name}'.")
                        continue
                    cleaned_text = clean_text(scraped_text)
                    source_urls = ', '.join(sources)
                    lead_info = extract_lead_info_with_gpt(cleaned_text, columns_to_retrieve, lead_name, lead_category, source_urls)
                    if lead_info:
                        leads_info.append(lead_info)
            if leads_info:
                try:
                    st.session_state['leads_info_df'] = pd.DataFrame(leads_info)
                    # Ensure 'Company/Group Name' is present
                    if 'Company/Group Name' not in st.session_state['leads_info_df'].columns:
                        st.session_state['leads_info_df']['Company/Group Name'] = st.session_state['leads_info_df'].get('Name', 'Unknown')
                        logger.warning("'Company/Group Name' was missing and has been set to 'Name' or 'Unknown'.")
                    # Ensure all required columns are present with default values
                    required_columns = ["Company/Group Name", "CEO/PI", "Researchers", "Grant Received", "Date", "Country", "University", "Email", "Phone Number", "Summary", "Contacts", "Category", "Source URLs"]
                    for col in required_columns:
                        if col not in st.session_state['leads_info_df'].columns:
                            st.session_state['leads_info_df'][col] = "Not Available"
                            logger.warning(f"Column '{col}' was missing in leads_info_df and has been set to 'Not Available'.")
                    st.session_state['step'] = max(st.session_state['step'], 3)
                    st.success("Lead information scraped successfully!")
                    logger.info("Lead information scraping completed successfully.")
                except Exception as e:
                    logger.error(f"Error creating leads_info_df: {e}")
                    st.error(f"Error processing scraped lead information: {e}")
            else:
                logger.error("Lead information scraping failed.")
                st.error("Failed to scrape lead information. Please check the logs for more details.")

    # Display the Leads Information DataFrame if it exists
    if not st.session_state['leads_info_df'].empty:
        st.write("### Leads Information")
        display_and_download(
            df=st.session_state['leads_info_df'],
            button_label="Leads Information",
            filename="leads_info"
        )

def rank_leads_section():
    """
    Section for Rank Leads: Rank the Leads based on configurable weights.
    """
    st.subheader("Step 3: Rank the Leads")
    st.markdown("**Configure Ranking Weights:**")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        weight_email = st.slider("Email Weight", min_value=0.0, max_value=10.0, value=5.0, step=0.1, key='weight_email')
    with col2:
        weight_phone = st.slider("Phone Weight", min_value=0.0, max_value=10.0, value=5.0, step=0.1, key='weight_phone')
    with col3:
        weight_grants = st.slider("Grants Weight", min_value=0.0, max_value=10.0, value=3.0, step=0.1, key='weight_grants')
    with col4:
        weight_date = st.slider("Date Weight", min_value=0.0, max_value=10.0, value=2.0, step=0.1, key='weight_date')

    rank_btn = st.button("Rank Leads", key='rank_leads_btn')
    if rank_btn:
        if st.session_state['leads_info_df'].empty:
            logger.error("Lead ranking attempted without lead information.")
            st.error("No lead information available. Please scrape lead information first.")
        else:
            with st.spinner('Ranking leads...'):
                weights = {
                    'email': weight_email,
                    'phone': weight_phone,
                    'grants': weight_grants,
                    'date': weight_date
                }
                ranked_leads_df = rank_leads(st.session_state['leads_info_df'], weights)
                st.session_state['ranked_leads_df'] = ranked_leads_df
                st.session_state['step'] = max(st.session_state['step'], 4)
                st.success("Leads ranked successfully!")
                logger.info("Leads ranked successfully.")

    # Display the Ranked Leads DataFrame if it exists
    if not st.session_state['ranked_leads_df'].empty:
        st.write("### Ranked Leads")
        try:
            display_and_download(
                df=st.session_state['ranked_leads_df'],
                button_label="Ranked Leads",
                filename="ranked_leads"
            )
        except Exception as e:
            logger.error(f"Error displaying ranked_leads_df: {e}")
            st.error(f"Error displaying ranked leads: {e}")

def analytics_section():
    """
    Section for Analytics: Provide various analytics and visualizations.
    """
    st.subheader("Step 4: Analytics")
    analytics_expander = st.expander("View Analytics")
    with analytics_expander:
        # Number of Grants per Lead
        if 'Grant Received' in st.session_state['ranked_leads_df'].columns:
            st.markdown("### Number of Grants per Lead")
            def count_grants(grant_entry):
                if isinstance(grant_entry, list):
                    return len(grant_entry)
                elif isinstance(grant_entry, str) and grant_entry.strip():
                    return 1
                else:
                    return 0
            st.session_state['ranked_leads_df']['Grant Count'] = st.session_state['ranked_leads_df']['Grant Received'].apply(count_grants)
            grant_counts = st.session_state['ranked_leads_df']['Grant Count']

            # Plotting Number of Grants per Lead
            fig, ax = plt.subplots(figsize=(10, 6))
            x = np.arange(len(st.session_state['ranked_leads_df']))
            ax.bar(x, grant_counts)
            ax.set_xlabel('Lead Name')
            ax.set_ylabel('Number of Grants')
            ax.set_title('Number of Grants per Lead')
            ax.set_xticks(x)
            try:
                ax.set_xticklabels(st.session_state['ranked_leads_df']['Company/Group Name'], rotation=90)
            except KeyError:
                logger.error("'Company/Group Name' column missing in ranked_leads_df when setting x-tick labels.")
                ax.set_xticklabels(['Unknown'] * len(x), rotation=90)
            ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
            plt.tight_layout()
            st.pyplot(fig)
        else:
            logger.warning("No 'Grant Received' column available in ranked_leads_df.")
            st.warning("No 'Grant Received' data available for grants analysis.")

        # Grant Years Distribution
        if 'Grant Received' in st.session_state['ranked_leads_df'].columns:
            st.markdown("### Grant Years Distribution")
            def extract_years(grant_entry):
                years = []
                if isinstance(grant_entry, list):
                    for grant in grant_entry:
                        matches = re.findall(r'\b(19|20)\d{2}\b', grant)
                        years.extend(matches)
                elif isinstance(grant_entry, str):
                    matches = re.findall(r'\b(19|20)\d{2}\b', grant_entry)
                    years.extend(matches)
                return [int(year) for year in years]

            st.session_state['ranked_leads_df']['Grant Years'] = st.session_state['ranked_leads_df']['Grant Received'].apply(extract_years)
            all_years = [year for sublist in st.session_state['ranked_leads_df']['Grant Years'] for year in sublist]
            if all_years:
                years_series = pd.Series(all_years)
                years_counts = years_series.value_counts().sort_index()
                # Plot the grant years distribution
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.bar(years_counts.index.astype(int), years_counts.values)
                ax.set_xlabel('Year')
                ax.set_ylabel('Number of Grants')
                ax.set_title('Distribution of Grants by Year')
                plt.xticks(rotation=90)
                plt.tight_layout()
                st.pyplot(fig)
            else:
                logger.warning("No grant years data available.")
                st.warning("No grant years data available for distribution analysis.")
        else:
            logger.warning("No 'Grant Received' column available in ranked_leads_df.")
            st.warning("No 'Grant Received' data available for grant years distribution.")

        # Number of Phone Numbers and Emails per Company
        st.markdown("### Number of Phone Numbers and Emails per Company")
        if 'Contacts' in st.session_state['ranked_leads_df'].columns:
            def count_emails_and_phones(contacts_entry):
                num_emails = 0
                num_phones = 0
                if isinstance(contacts_entry, str):
                    try:
                        contacts_list = json.loads(contacts_entry)
                    except json.JSONDecodeError:
                        contacts_list = []
                elif isinstance(contacts_entry, list):
                    contacts_list = contacts_entry
                else:
                    contacts_list = []
                for contact in contacts_list:
                    email = contact.get('Email')
                    if email and email.strip().lower() not in ['not provided', 'not available', '']:
                        num_emails += 1
                    phone = contact.get('Phone Number')
                    if phone and phone.strip().lower() not in ['not provided', 'not available', '']:
                        num_phones += 1
                return pd.Series({'Num Emails': num_emails, 'Num Phones': num_phones})

            contacts_counts = st.session_state['ranked_leads_df']['Contacts'].apply(count_emails_and_phones)
            st.session_state['ranked_leads_df'] = st.session_state['ranked_leads_df'].join(contacts_counts)
            # Display the counts in a table
            try:
                st.write("#### Emails and Phone Numbers per Company")
                st.write(st.session_state['ranked_leads_df'][['Company/Group Name', 'Num Emails', 'Num Phones']])
            except KeyError:
                logger.error("'Company/Group Name', 'Num Emails', or 'Num Phones' columns missing in ranked_leads_df.")
                st.error("Required columns for contact analysis are missing.")

            # Plot number of emails and phone numbers per company
            fig, ax = plt.subplots(figsize=(10, 6))
            x = np.arange(len(st.session_state['ranked_leads_df']))
            width = 0.35
            try:
                ax.bar(x - width/2, st.session_state['ranked_leads_df']['Num Emails'], width, label='Emails')
                ax.bar(x + width/2, st.session_state['ranked_leads_df']['Num Phones'], width, label='Phone Numbers')
                ax.set_xlabel('Company/Group Name')
                ax.set_ylabel('Count')
                ax.set_title('Number of Emails and Phone Numbers per Company')
                ax.set_xticks(x)
                try:
                    ax.set_xticklabels(st.session_state['ranked_leads_df']['Company/Group Name'], rotation=90)
                except KeyError:
                    logger.error("'Company/Group Name' column missing in ranked_leads_df when setting x-tick labels.")
                    ax.set_xticklabels(['Unknown'] * len(x), rotation=90)
                ax.legend()
                plt.tight_layout()
                st.pyplot(fig)
            except KeyError:
                logger.error("One of the required columns for plotting is missing in ranked_leads_df.")
                st.error("Required columns for contact plotting are missing.")
        else:
            logger.warning("No 'Contacts' data available in ranked_leads_df.")
            st.warning("No 'Contacts' data available for contact analysis.")

        # Number of Individual Names per Company
        st.markdown("### Number of Individual Names per Company")
        def count_individual_names(row):
            names = set()
            # From CEO/PI
            ceo_pi = row.get('CEO/PI')
            if isinstance(ceo_pi, str) and ceo_pi.strip() and ceo_pi.strip().lower() != 'not available':
                names.add(ceo_pi.strip())
            # From Researchers
            researchers = row.get('Researchers')
            if isinstance(researchers, list):
                names.update([r for r in researchers if r.strip().lower() != 'not available'])
            elif isinstance(researchers, str) and researchers.strip() and researchers.strip().lower() != 'not available':
                names.add(researchers.strip())
            # From Contacts
            contacts = row.get('Contacts')
            if isinstance(contacts, str):
                try:
                    contacts_list = json.loads(contacts)
                except json.JSONDecodeError:
                    contacts_list = []
            elif isinstance(contacts, list):
                contacts_list = contacts
            else:
                contacts_list = []
            for contact in contacts_list:
                name = contact.get('Name')
                if name and name.strip().lower() != 'not available':
                    names.add(name.strip())
            return len(names)

        try:
            st.session_state['ranked_leads_df']['Num Individuals'] = st.session_state['ranked_leads_df'].apply(count_individual_names, axis=1)
            # Display the counts in a table
            try:
                st.write("#### Number of Individual Names per Company")
                st.write(st.session_state['ranked_leads_df'][['Company/Group Name', 'Num Individuals']])
            except KeyError:
                logger.error("'Company/Group Name' or 'Num Individuals' columns missing in ranked_leads_df.")
                st.error("Required columns for individual names count are missing.")

            # Plot number of individuals per company
            try:
                fig, ax = plt.subplots(figsize=(10, 6))
                x = np.arange(len(st.session_state['ranked_leads_df']))
                ax.bar(x, st.session_state['ranked_leads_df']['Num Individuals'])
                ax.set_xlabel('Company/Group Name')
                ax.set_ylabel('Number of Individuals')
                ax.set_title('Number of Individual Names per Company')
                ax.set_xticks(x)
                try:
                    ax.set_xticklabels(st.session_state['ranked_leads_df']['Company/Group Name'], rotation=90)
                except KeyError:
                    logger.error("'Company/Group Name' column missing in ranked_leads_df when setting x-tick labels.")
                    ax.set_xticklabels(['Unknown'] * len(x), rotation=90)
                plt.tight_layout()
                st.pyplot(fig)
            except KeyError:
                logger.error("'Company/Group Name' or 'Num Individuals' columns missing in ranked_leads_df.")
                st.error("Required columns for individual names plotting are missing.")
            except Exception as e:
                logger.error(f"Error plotting individual names: {e}")
                st.error(f"Error plotting individual names: {e}")
        except Exception as e:
            logger.error(f"Error counting individual names: {e}")
            st.error(f"Error counting individual names: {e}")

        # Summaries
        if 'Summary' in st.session_state['ranked_leads_df'].columns:
            st.markdown("### Summaries")
            for idx, row in st.session_state['ranked_leads_df'].iterrows():
                company_name = row['Company/Group Name']
                summary = row['Summary']
                if pd.notnull(summary) and summary.strip():
                    st.write(f"**{company_name}:** {summary}")
                else:
                    st.write(f"**{company_name}:** No summary available.")

            # Word Cloud for Summaries
            summaries = st.session_state['ranked_leads_df']['Summary'].dropna().tolist()
            combined_text = ' '.join(summaries)
            if combined_text:
                st.markdown("### Word Cloud for Summaries")
                wordcloud = WordCloud(width=800, height=400, background_color='white').generate(combined_text)
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.imshow(wordcloud, interpolation='bilinear')
                ax.axis('off')
                plt.tight_layout()
                st.pyplot(fig)
            else:
                logger.warning("No summaries available for word cloud.")
                st.warning("No summaries available to generate a word cloud.")
        else:
            logger.warning("No 'Summary' column available in ranked_leads_df.")
            st.warning("No 'Summary' data available for summaries and word cloud.")

        # ==========================
        # Enhanced Analytics: Person Profiles
        # ==========================
        st.markdown("### Detailed Person Profiles")

        if 'person_leads_df' in st.session_state and not st.session_state['person_leads_df'].empty:
            person_leads_df = st.session_state['person_leads_df']
            # Ensure 'Company/Group Name' exists in person_leads_df
            if 'Company/Group Name' not in person_leads_df.columns:
                if 'Associated Company' in person_leads_df.columns:
                    person_leads_df = person_leads_df.rename(columns={'Associated Company': 'Company/Group Name'})
                    logger.info("Renamed 'Associated Company' to 'Company/Group Name' in person_leads_df.")
                else:
                    logger.warning("Company association for persons not found.")
                    st.warning("Company association for persons not found. Assigning 'Unknown' to persons without a company.")
                    person_leads_df['Company/Group Name'] = 'Unknown'
            else:
                logger.info("'Company/Group Name' exists in person_leads_df.")

            # Check again after renaming or assigning
            if 'Company/Group Name' in person_leads_df.columns:
                try:
                    # Group persons by company
                    grouped = person_leads_df.groupby('Company/Group Name')

                    for company, group in grouped:
                        st.markdown(f"#### {company}")
                        # Display a table for each company
                        display_and_download(
                            df=group.drop(columns=['Company/Group Name']),
                            button_label=f"Persons at {company}",
                            filename=f"persons_at_{company.replace(' ', '_')}"
                        )
                except Exception as e:
                    logger.error(f"Error displaying person profiles: {e}")
                    st.error(f"Error displaying person profiles: {e}")
            else:
                logger.error("'Company/Group Name' column missing in person_leads_df after renaming.")
                st.error("'Company/Group Name' column missing in person_leads_df after renaming.")
        else:
            logger.warning("No person leads data available.")
            st.warning("No person leads data available. Please extract and scrape persons first.")

def extract_persons_section():
    """
    Section for Extract Persons: Extract and Scrape Persons Associated with Leads.
    """
    st.subheader("Step 5: Extract and Scrape Persons Associated with Leads")
    extract_persons_btn = st.button("Extract and Scrape Persons", key='extract_persons_btn')
    if extract_persons_btn:
        if st.session_state['leads_info_df'].empty:
            logger.error("Person extraction attempted without lead information.")
            st.error("No lead information available. Please scrape lead information first.")
        else:
            with st.spinner('Extracting persons associated with leads...'):
                persons = extract_persons(st.session_state['leads_info_df'])
                if not persons:
                    logger.warning("No persons found associated with the leads.")
                    st.warning("No persons found associated with the leads.")
                else:
                    logger.info(f"Found {len(persons)} unique persons associated with the leads.")

        if 'persons' in locals() and persons:
            with st.spinner('Generating detailed person information...'):
                person_leads = []
                # Placeholder to display current person being processed
                person_placeholder = st.empty()
                for idx, (person_name, associated_lead) in enumerate(persons):
                    person_placeholder.text(f"Processing Person {idx+1}/{len(persons)}: {person_name} (Company: {associated_lead})")
                    person_urls = perform_google_search(person_name)
                    if not person_urls:
                        logger.warning(f"No URLs found for '{person_name}'.")
                        continue
                    scraped_text = ""
                    sources = []
                    # Display and scrape the first three URLs for the person
                    for url in person_urls[:3]:
                        logger.info(f"Scraping URL for '{person_name}': {url}")
                        with st.spinner(f"Scraping URL for '{person_name}': {url}"):
                            scraped_text += scrape_landing_page(url) + " "
                            sources.append(url)
                    if not scraped_text.strip():
                        logger.warning(f"No text scraped from URLs for '{person_name}'.")
                        continue
                    cleaned_text = clean_text(scraped_text)
                    source_urls = ', '.join(sources)
                    person_info = extract_person_info_with_gpt(cleaned_text, person_name, source_urls)
                    if person_info:
                        # Ensure 'Company/Group Name' is included
                        person_info['Company/Group Name'] = associated_lead
                        person_leads.append(person_info)
            if person_leads:
                try:
                    person_leads_df = pd.DataFrame(person_leads)
                    st.session_state['person_leads_df'] = person_leads_df
                    st.success("Person information scraped successfully!")
                    logger.info("Person information scraped successfully.")
                except Exception as e:
                    logger.error(f"Error creating person_leads_df: {e}")
                    st.error(f"Error processing scraped person information: {e}")
            else:
                logger.error("Person information scraping failed.")
                st.error("Failed to scrape person information. Please check the logs for more details.")

    # Display the Person Leads DataFrame if it exists
    if not st.session_state['person_leads_df'].empty:
        st.write("### Person Leads Information")
        display_and_download(
            df=st.session_state['person_leads_df'],
            button_label="Person Leads Information",
            filename="person_leads_info"
        )

def download_data_section():
    """
    Section for Download Data: Download different datasets.
    """
    st.subheader("Step 6: Download Data")

    # Option to download Leads Information
    if not st.session_state['leads_info_df'].empty:
        st.markdown("### Download Leads Information")
        download_leads_btn = st.button("Download Leads Information", key='download_leads_info_btn')
        if download_leads_btn:
            try:
                download_leads(st.session_state['leads_info_df'], "leads_info")
                st.success("Leads information downloaded successfully!")
                logger.info("Leads information downloaded successfully.")
            except Exception as e:
                logger.error(f"Error downloading leads_info: {e}")
                st.error(f"Error downloading leads information: {e}")

    # Option to download Ranked Leads
    if not st.session_state['ranked_leads_df'].empty:
        st.markdown("### Download Ranked Leads")
        ranked_download_btn = st.button("Download Ranked Leads", key='download_ranked_leads_btn')
        if ranked_download_btn:
            try:
                download_leads(st.session_state['ranked_leads_df'], "ranked_leads")
                st.success("Ranked leads downloaded successfully!")
                logger.info("Ranked leads downloaded successfully.")
            except Exception as e:
                logger.error(f"Error downloading ranked_leads: {e}")
                st.error(f"Error downloading ranked leads: {e}")

    # Option to download Person Leads
    if not st.session_state['person_leads_df'].empty:
        st.markdown("### Download Person Leads")
        person_excel_filename = st.text_input(
            "Enter a name for the Persons Excel file (without extension):",
            value="person_leads",
            key='person_excel_input'
        )
        person_download_btn = st.button("Download Person Leads Data", key='download_person_excel_btn')
        if person_download_btn:
            if not person_excel_filename.strip():
                st.warning("Please enter a valid filename for the Excel file.")
            else:
                try:
                    download_leads(st.session_state['person_leads_df'], person_excel_filename)
                    st.success(f"Person leads data downloaded as {person_excel_filename}.xlsx")
                    logger.info(f"Person leads data downloaded as {person_excel_filename}.xlsx")
                except Exception as e:
                    logger.error(f"Error downloading person_leads: {e}")
                    st.error(f"Error downloading person leads: {e}")

    # Option to download all data together (ZIP functionality can be implemented as needed)
    st.markdown("### Download All Data")
    all_download_btn = st.button("Download All Data as ZIP", key='download_all_data_btn')
    if all_download_btn:
        logger.warning("ZIP download functionality is not implemented yet.")
        st.warning("ZIP download functionality is not implemented yet.")

# ==========================
# Main Application Code with Navigation
# ==========================

# Remove the main function and place the code at the top level
st.set_page_config(layout="wide")
st.title("Sequential Lead Generation Tool")

# Navigation Menu
menu = ["Input Leads", "Scrape Lead Information", "Rank Leads", "Analytics", "Extract Persons", "Download Data"]
choice = st.sidebar.selectbox("Navigate", menu)

if choice == "Input Leads":
    input_leads()
elif choice == "Scrape Lead Information":
    scrape_lead_information()
elif choice == "Rank Leads":
    rank_leads_section()
elif choice == "Analytics":
    analytics_section()
elif choice == "Extract Persons":
    extract_persons_section()
elif choice == "Download Data":
    download_data_section()

# ==========================
# Additional Information
# ==========================
st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.info(
    """
    **Sequential Lead Generation Tool**

    This application helps in generating, scraping, ranking, and analyzing leads. Navigate through the menu to perform different tasks.

    - **Input Leads:** Generate or manually add leads.
    - **Scrape Lead Information:** Fetch detailed information about each lead.
    - **Rank Leads:** Prioritize leads based on configurable weights.
    - **Analytics:** Visualize and analyze lead data.
    - **Extract Persons:** Extract and gather information about individuals associated with leads.
    - **Download Data:** Download the processed data in various formats.
    """
)