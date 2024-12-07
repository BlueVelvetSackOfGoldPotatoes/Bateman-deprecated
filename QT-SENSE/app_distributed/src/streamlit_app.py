# app.py

import streamlit as st
from streamlit_tags import st_tags
import time
import json
import os
import random
import logging
from dotenv import load_dotenv
from streamlit_option_menu import option_menu

# Import utility functions from utils.py
from utils import (
    clean_text,
    extract_persons,
    download_leads,
    scrape_landing_page,
    perform_google_search,
    extract_person_info_with_llm,
    generate_leads_with_llm,
    extract_lead_info_with_llm_per_field,
    search_leads_via_url,
    display_lead_information
)

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

# Load environment variables
load_dotenv()

# ==========================
# Streamlit App Configuration
# ==========================

def run_missing_dependencies(unmet):
    """
    Executes the missing prerequisite sections in order.
    """
    for dep in unmet:
        dep_key = dep.replace(" ", "_").lower()
        if dep_key == "input_leads":
            input_leads()
        elif dep_key == "scrape_lead_information":
            scrape_lead_information()
        elif dep_key == "download_data":
            download_data_section()
        else:
            logger.warning(f"No function defined for dependency: {dep}")

# Set the page configuration as the very first Streamlit command
st.set_page_config(layout="wide")
st.title("BATEMAN")

# Initialize session state variables
session_vars = [
    'processed_leads', 'leads_info', 'is_scraping',
    'person_leads', 'leads', 'leads_list',
    'ranked_leads', 'json_filename', 'menu_selection',
    'step', 'context',
    'selected_url_leads',
    'url_leads_selection'
]

for var in session_vars:
    if var not in st.session_state:
        if var.endswith('_leads') or var.endswith('_info') or var.endswith('_list') or var == 'ranked_leads':
            st.session_state[var] = []
        elif var == 'json_filename':
            st.session_state[var] = "leads"
        elif var == 'menu_selection':
            st.session_state[var] = "Input Leads"
        elif var == 'step':
            st.session_state[var] = 0
        elif var == 'context':
            st.session_state[var] = ""
        elif var in ['selected_url_leads', 'url_leads_selection']:
            st.session_state[var] = {}
        else:
            st.session_state[var] = []

# Define menu options
menu_options = [
    "Input Leads", "Analyse Lead Information", "Download Data"
]

# Create the option menu
selected = option_menu(
    menu_title=None,
    options=menu_options,
    icons=["pencil", "search", "bar-chart", "activity", "people", "book", "download"],
    menu_icon="cast",
    orientation="horizontal",
    key='menu_selection'
)

# Update menu selection in session state
if selected != st.session_state['menu_selection']:
    st.session_state['menu_selection'] = selected

# ==========================
# Define Section Functions
# ==========================

def input_leads():
    """
    Section for Input Leads: Generate Leads, Add Leads Manually, or Search Leads via URL.
    """
    leads_list = []
    # Initialize session state if not already
    if 'leads' not in st.session_state:
        st.session_state['leads'] = []

    if 'leads_info' not in st.session_state:
        st.session_state['leads_info'] = []

    if 'person_leads' not in st.session_state:
        st.session_state['person_leads'] = []

    if 'context' not in st.session_state:
        st.session_state['context'] = ""

    if 'step' not in st.session_state:
        st.session_state['step'] = 1

    # Input Leads Section
    st.subheader("Input Leads")

    # Context Input
    context = st.text_area("Context:", value=st.session_state['context'])
    st.session_state['context'] = context

    # Lead Input Options
    option = st.radio(
        "Choose how to input leads:",
        ["Generate Leads", "Add Leads Manually", "Search Leads via URL"],
        key='lead_input_option'
    )

    # Generate Leads Section
    if option == "Generate Leads":
        num_leads_total = st.number_input(
            "Number of leads per type:",
            min_value=1,
            max_value=100,
            value=10,
            step=1,
            key='num_leads_total'
        )
        default_lead_types = ["Research Groups"]
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
                st.error("Please provide a context for lead generation.")
                st.stop()
            elif not lead_types:
                logger.error("Lead generation attempted without specifying lead types.")
                st.warning("Please specify at least one lead type.")
            else:
                with st.spinner('Generating leads...'):
                    leads = generate_leads_with_llm(context, num_leads_total)
                if leads:
                    # Append to the existing leads list
                    st.session_state['leads'].extend(leads)

                    st.session_state['step'] = max(st.session_state['step'], 2)

                    st.success(f"Leads generated successfully! Total leads: {len(st.session_state['leads'])}")
                    logger.info(f"Generated {len(leads)} leads.")
                else:
                    logger.error("Lead generation failed.")
                    st.error("Failed to generate leads. Please check the logs for more details.")
                    st.stop()

    # Add Leads Manually Section
    if option == "Add Leads Manually":
        st.write("Enter your leads below:")

        leads_input = st.text_area(
            "Enter one lead per line, in the format 'Entity':",
            height=150,
            key='manual_leads_input'
        )

        add_leads_btn = st.button("Add Leads", key='add_leads_btn')
        if add_leads_btn:
            if not context.strip():
                logger.error("Lead addition attempted with empty context.")
                st.warning("Please provide a context for lead addition.")
            for line in leads_input.strip().split('\n'):
                parts = line.strip().split(',')
                if len(parts) == 1:
                    name = parts[0].strip()

                    leads_list.append({
                        "Entity": name
                    })
                else:
                    logger.warning(f"Invalid format in line: {line}")
                    st.warning(f"Invalid format in line: {line}. Please use 'Entity'.")

            if leads_list:
                # Convert existing leads to a set for faster lookup
                existing_entities = set(lead["Entity"] for lead in st.session_state['leads'])

                # Filter out any leads that already exist
                new_unique_leads = [lead for lead in leads_list if lead["Entity"] not in existing_entities]

                if new_unique_leads:
                    # Append to the existing leads list
                    st.session_state['leads'].extend(new_unique_leads)
                    logger.debug(f"new_unique_leads: {new_unique_leads[:5]}") 
                    
                    st.session_state['step'] = max(st.session_state['step'], 2)

                    st.success(f"Added {len(new_unique_leads)} new leads successfully! Total leads: {len(st.session_state['leads'])}")
                    logger.info(f"Added {len(new_unique_leads)} leads manually.")
                else:
                    st.info("No new unique leads to add.")
            else:
                logger.error("No valid leads entered.")
                st.error("No valid leads entered. Please ensure each line contains 'Entity'.")
                st.stop()

    # Search Leads via URL Section
    elif option == "Search Leads via URL":
        st.write("### Search Leads via URL")
        st.write("Enter the URL details below:")
        url_input = st.text_input("Enter the URL name or URL:", key='url_input')

        search_btn = st.button("Search Leads", key='search_url_leads_btn')
        if search_btn:
            if not context.strip():
                logger.error("Lead search via URL attempted with empty context.")
                st.error("Please provide a context for lead search.")
                st.stop()
            if not url_input.strip():
                st.warning("Please enter a URL name or URL.")
            else:
                with st.spinner('Searching for leads associated with the URL...'):
                    leads_found = search_leads_via_url(url_input, context)
                    if leads_found:
                        st.success(f"Found {len(leads_found)} leads related to the URL.")
                        
                        # Ensure leads_found is a list of dictionaries with 'Entity' key
                        new_url_leads = [{"Entity": lead["Entity"]} for lead in leads_found if "Entity" in lead]

                        # Remove duplicates based on 'Entity'
                        existing_entities = set(lead["Entity"] for lead in st.session_state['leads'])
                        unique_url_leads = [lead for lead in new_url_leads if lead["Entity"] not in existing_entities]

                        if unique_url_leads:
                            # Append to the existing leads list
                            st.session_state['leads'].extend(unique_url_leads)
                            logger.debug(f"Unique URL leads: {unique_url_leads[:5]}")
                            
                            st.success(f"Added {len(unique_url_leads)} new URL leads successfully!")
                            logger.info(f"Added {len(unique_url_leads)} URL leads.")
                        else:
                            st.info("All found URL leads are already present in the leads list.")
                    else:
                        st.warning("No leads found for the provided URL. Please try a different URL name or URL.")

    # Display and Manage Leads
    if st.session_state['leads']:
        st.write("### Leads")
        st.write("You can view your leads below:")

        # Display leads as JSON
        st.json(st.session_state['leads'], expanded=False)

        # Optionally, allow editing leads using text input or other Streamlit widgets
        # Example: Allowing users to remove leads
        leads_to_remove = st.multiselect(
            "Select leads to remove:",
            options=[lead["Entity"] for lead in st.session_state['leads']],
            key='leads_to_remove'
        )

        if st.button('Remove Selected Leads', key='remove_leads_btn'):
            if leads_to_remove:
                st.session_state['leads'] = [lead for lead in st.session_state['leads'] if lead["Entity"] not in leads_to_remove]
                st.success(f"Removed {len(leads_to_remove)} leads successfully!")
                logger.info(f"Removed leads: {leads_to_remove}")
            else:
                st.info("No leads selected for removal.")

        # Provide Download Option for All Leads
        st.write("### Download All Leads")
        display_lead_information(
            leads=st.session_state['leads'],
            button_label="All Leads",
            filename="all_leads"
        )

    # Final Section
    st.session_state['input_leads'] = True
    st.success("Input Leads section loaded successfully!")

def scrape_lead_information(leads_to_process=None, is_url=False):
    """
    Section for Analyse Lead Information: Perform per-field searches and scrape information,
    then extract and analyze persons associated with each lead.
    
    :param leads_to_process: Optional list of leads to process. If None, process all leads in st.session_state['leads'].
    :param is_url: Boolean indicating if the leads are URL leads.
    """
    st.subheader("Search and Analyse Lead Information")
    
    # Initialize progress elements
    progress_bar = st.empty()
    status_text = st.empty()
    
    person_status_text = st.empty()
    person_progress_bar = st.progress(0)
    
    st.header("Analyze Lead Information")
    default_columns = [
        "Entity", "CEO/PI", "Researchers", "Grants",
        "Phone Number", "Email", "Country", "University",
        "Summary", "Contacts"
    ]
    columns_to_retrieve = st_tags(
        label='',
        text='Add or remove information fields:',
        value=default_columns,
        suggestions=default_columns,
        maxtags=20,
        key='columns_to_retrieve'
    )
    
    # Add input for additional search keywords for persons
    default_person_keywords = [
        "Education",
        "Current Position",
        "Expertise",
        "Email",
        "Phone Number",
        "Faculty",
        "University",
        "Bio",
        "Academic/Work Website Profile Link",
        "LinkedIn/Profile Link",
        "Facebook/Profile Link",
        "Grant",
        "Curriculum Vitae"
    ]

    person_search_keywords = st_tags(
        label='',
        text='Add or remove keywords to guide the person search:',
        value=default_person_keywords,
        suggestions=default_person_keywords,
        maxtags=20,
        key='person_search_keywords'
    )
    
    # Input for maximum number of persons to process per lead
    max_persons_to_process = st.number_input(
        label='Maximum number of persons to process per lead:',
        min_value=1,
        max_value=100,
        value=10,
        step=1,
        key='max_persons_to_process'
    )
    
    search_btn = st.button("Search and Analyse Leads", key='search_leads_btn')
    if search_btn:
        # Validate input context
        if not st.session_state.get('context', '').strip():
            logger.error("Lead scraping attempted with empty context.")
            st.warning("Please provide a context for lead scraping.")
            return
        
        # Determine which leads to process
        leads = leads_to_process if leads_to_process else st.session_state.get('leads', [])
        
        # Validate that leads are available
        if not leads:
            logger.error("Lead scraping attempted without any generated or added leads.")
            st.error("No leads available. Please add or generate leads first.")
            return
        
        # Validate that at least one information field is selected
        if not columns_to_retrieve:
            logger.error("Lead scraping attempted without specifying information fields.")
            st.error("Please select at least one information field to retrieve.")
            return
        
        # Initialize scraping state if not already in progress
        if not st.session_state.get('is_scraping', False):
            st.session_state['is_scraping'] = True
            st.session_state['processed_leads'] = []
            st.session_state['leads_info'] = []
            st.session_state['person_leads'] = []
        
        total_leads = len(leads)
        
        status_text.text("Starting lead information scraping...")
        
        # Initialize overall progress bar
        progress_bar.progress(0)
        
        for idx, lead in enumerate(leads):
            lead_name = lead.get("Entity", "Unknown")
                    
            # Update overall progress based on current lead index
            current_progress = (idx + 1) / total_leads
            progress_bar.progress(current_progress)
            status_text.text(f"Processing Lead {idx + 1} of {total_leads}: {lead_name}")
            
            lead_info = extract_lead_info_with_llm_per_field(lead_name, columns_to_retrieve)
            if lead_info:
                # Append lead information to the leads_info list
                st.session_state['leads_info'].append(lead_info)
            
                # Update the UI with the newly processed lead
                with st.container():
                    st.markdown(f"### {lead_info.get('Entity', 'Unknown')}")
                    st.json(lead_info, expanded=False)
                    st.markdown("---")
            
            # Extract and Analyze Persons Associated with the Lead
            if not is_url:
                persons = extract_persons([lead_info])
            else:
                persons = extract_persons(st.session_state['leads_info'])

            if not persons:
                logger.warning(f"No persons found associated with the lead '{lead_name}'.")
                st.warning(f"No persons found associated with the lead '{lead_name}'.")
                person_progress_bar.progress(1.0)  # Mark personnel progress as complete for this lead
            else:
                # Limit the number of persons to process per lead
                if len(persons) > max_persons_to_process:
                    persons = persons[:max_persons_to_process]
                    logger.info(f"Processing first {max_persons_to_process} persons for lead '{lead_name}'.")
                
                logger.info(f"Found {len(persons)} unique persons associated with the lead '{lead_name}'.")
                person_progress_increment = 1.0 / len(persons) if len(persons) > 0 else 1
                person_progress = 0
                
                for p_idx, (person_name, associated_lead) in enumerate(persons):
                    person_status_text.text(f"Processing Person {p_idx + 1}/{len(persons)}: {person_name}")
                    logger.info(f"Processing Person {p_idx + 1}/{len(persons)}: {person_name}\n" + "="*50 + "\n")
                    
                    # Include the additional keywords in the search
                    person_urls = perform_google_search(person_name, num_results=3, keywords=person_search_keywords, lead_name=associated_lead)
                    if not person_urls:
                        logger.warning(f"No URLs found for '{person_name}'.")
                        continue
                    scraped_text = ""
                    sources = []
                    # Analyze the URLs
                    for key, urls in person_urls.items():
                        logger.info(f"Looking at urls for person:{person_name} and keyword: {key}")

                        for url in urls:
                            with st.spinner(f"Looking at URL for '{person_name}': {url}"):
                                scraped_content = scrape_landing_page(url)
                                if scraped_content:
                                    scraped_text += scraped_content + " "
                                    sources.append(url)
                                time.sleep(random.uniform(1, 2))  # Respect rate limits
                    if not scraped_text.strip():
                        logger.warning(f"No text scraped from URLs for '{person_name}'. Skipping.")
                        continue
                    cleaned_text = clean_text(scraped_text)
                    os.makedirs("database", exist_ok=True)
                    with open(os.path.join("database", person_name + ".txt"), "w") as f:
                        f.write(cleaned_text)
                    source_urls = sources
                    person_info = extract_person_info_with_llm(cleaned_text, person_name, source_urls, key_words=person_search_keywords)
                    if person_info:
                        person_info['Name'] = person_name  # Ensure correct name assignment
                        person_info['Associated Lead'] = associated_lead  # Associate person with the lead company
                        
                        st.session_state['person_leads'].append(person_info)
                    
                        # Update personnel progress bar
                        person_progress += person_progress_increment
                        person_progress_bar.progress(min(person_progress, 1.0))
                
                if is_url:
                    st.success(f"Person information scraped successfully for URL leads!")
                    logger.info(f"Person information scraped successfully for URL leads.")
                else:
                    st.success(f"Person information scraped successfully for lead '{lead_name}'!")
                    logger.info(f"Person information scraped successfully for lead '{lead_name}'.")
                
                # Display the person leads information
                st.write(f"### Persons Associated with {lead_name}")

                # Filter the list to include only persons associated with the current lead
                filtered_persons = [person for person in st.session_state['person_leads'] if person.get('Associated Lead') == lead_name]

                display_lead_information(
                    leads=filtered_persons,
                    button_label="Download Person Leads Information",
                    filename=f"person_leads_info_{lead_name.replace(' ', '_')}"
                )
    
        # After all leads are processed, mark progress as complete
        progress_bar.progress(1.0)
        person_progress_bar.progress(1.0)
        status_text.text("Lead information scraping completed!")
        st.success("All leads have been processed successfully.")
        st.session_state['is_scraping'] = False

def download_data_section():
    st.subheader("Download Data")

    # Option to download Leads Information
    if st.session_state['leads_info']:
        st.markdown("### Download Leads Information")
        download_leads_btn = st.button("Download Leads Information", key='download_leads_info_btn')
        if download_leads_btn:
            try:
                download_leads(st.session_state['leads_info'], "leads_info")
                st.success("Leads information section downloaded successfully!")
                logger.info("Leads information section downloaded successfully.")
            except Exception as e:
                logger.error(f"Error downloading leads_info: {e}")
                st.error(f"Error downloading leads information: {e}")
                st.stop()

    # Option to download Person Leads
    if st.session_state['person_leads']:
        st.markdown("### Download Person Leads")
        person_json_filename = st.text_input(
            "Enter a name for the Persons JSON file (without extension):",
            value="person_leads",
            key='person_json_input'
        )
        person_download_btn = st.button("Download Person Leads Data", key='download_person_json_btn')
        if person_download_btn:
            if not person_json_filename.strip():
                st.warning("Please enter a valid filename for the JSON file.")
            else:
                try:
                    download_leads(st.session_state['person_leads'], person_json_filename)
                    st.success(f"Person leads data downloaded as {person_json_filename}.json")
                    logger.info(f"Person leads data downloaded as {person_json_filename}.json")
                except Exception as e:
                    logger.error(f"Error downloading person_leads: {e}")
                    st.error(f"Error downloading person leads: {e}")
                    st.stop()

    # Option to download all data together (ZIP functionality can be implemented as needed)
    st.markdown("### Download All Data")
    all_download_btn = st.button("Download All Data as ZIP", key='download_all_data_btn')
    if all_download_btn:
        logger.warning("ZIP download functionality is not implemented yet.")
        st.warning("ZIP download functionality is not implemented yet.")

def display_and_download(leads, button_label, filename, height=400):
    """
    Displays a list of leads (dicts) and provides options to download as JSON.
    
    :param leads: List of dictionaries containing lead information.
    :param button_label: The label for the download button.
    :param filename: The base filename for the downloaded file.
    :param height: The height of the displayed information.
    """
    st.json(leads, expanded=False)
    
    # Convert to JSON string for download
    json_data = json.dumps(leads, indent=2)
    
    st.download_button(
        label=f"Download {button_label} as JSON",
        data=json_data,
        file_name=f"{filename}.json",
        mime='application/json'
    )

# ==========================
# Main Navigation Logic
# ==========================

# Sidebar Information
st.sidebar.image("../images/bateman4.png", use_container_width=True)
st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.info(
    """
    **BATEMAN**

    This application helps in generating, scraping, ranking, and analyzing leads. Navigate through the menu to perform different tasks.

    - **Input Leads:** Generate or manually add leads.
    - **Analyse Lead Information:** Fetch detailed information about each lead.
    - **Download Data:** Download the processed data in various formats.
    """
)

# Navigation Menu
if st.session_state['menu_selection'] == "Input Leads":
    input_leads()
elif st.session_state['menu_selection'] == "Analyse Lead Information":
    scrape_lead_information()
elif st.session_state['menu_selection'] == "Download Data":
    download_data_section()
