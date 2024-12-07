# app.py

import streamlit as st
import pandas as pd
from streamlit_tags import st_tags
import time
import os
import random
import logging
from dotenv import load_dotenv
from streamlit_option_menu import option_menu

# Import functions from network.py
# from network import (
#     display_author_information_cards,
#     build_network_graph_with_relevance,
#     calculate_relevance_with_llm,
#     visualize_network,
#     fetch_recent_papers
# )  

# Import functions from analytics.py
# from analytics import (
#     rank_leads_with_bant,
#     generate_bant_report

# )

# Import utility functions from utils.py
from utils import (
    clean_text,
    extract_persons,
    download_leads,
    scrape_landing_page,
    perform_google_search,
    extract_person_info_with_llm,
    generate_leads_with_llm,
    summarize_paper,
    extract_lead_info_with_llm_per_field,
    search_leads_via_conference,
    clean_dataframe,
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
        # elif dep_key == "rank_leads_bant":
        #     rank_leads_section()
        # elif dep_key == "analytics":
        #     analytics_section()
        # elif dep_key == "author_papers":
        #     author_papers_section()
        # elif dep_key == "download_data":
        #     download_data_section()
        else:
            logger.warning(f"No function defined for dependency: {dep}")

# Set the page configuration as the very first Streamlit command
st.set_page_config(layout="wide")
st.title("BATEMAN")

# Initialize session state variables
session_vars = [
    'processed_leads', 'leads_info_df', 'is_scraping',
    'person_leads_df', 'leads', 'leads_df',
    'ranked_leads_df', 'excel_filename', 'menu_selection',
    'step', 'context',
    'selected_conference_leads',
    'conference_leads_selection',
    'leads_combined_df',  # New combined DataFrame
    # New dynamic field management
    'lead_fields', 'personnel_fields'
]

for var in session_vars:
    if var not in st.session_state:
        if var.endswith('_df'):
            st.session_state[var] = pd.DataFrame()
        elif var == 'excel_filename':
            st.session_state[var] = "leads"
        elif var == 'menu_selection':
            st.session_state[var] = "Input Leads"
        elif var == 'step':
            st.session_state[var] = 0
        elif var == 'context':
            st.session_state[var] = ""
        elif var in ['selected_conference_leads', 'conference_leads_selection']:
            st.session_state[var] = {}
        elif var == 'leads_combined_df':
            st.session_state[var] = pd.DataFrame(columns=[
                "Type",  # 'Company' or 'Personnel'
                "Entity",  # Company Name
                "Category",
                "CEO/PI",
                "Country",
                "University",
                "Summary",
                "Recommendations",
                "Source URLs",
                # Personnel-specific columns
                "Personnel Name",
                "Personnel Title",
                "Personnel Email",
                "Personnel Phone"
            ])
        elif var == 'lead_fields':
            # Initialize with default Lead fields
            st.session_state[var] = [
                "Entity", "Category", "CEO/PI", "Country",
                "University", "Summary", "Recommendations", "Source URLs"
            ]
        elif var == 'personnel_fields':
            # Initialize with default Personnel fields
            st.session_state[var] = [
                "Personnel Name", "Personnel Title",
                "Personnel Email", "Personnel Phone"
            ]
        else:
            st.session_state[var] = []

# Define menu options
menu_options = [
    "Input Leads", "Analyse Lead Information", "Rank Leads (BANT)", 
    "Analytics", "Author Papers", "Download Data"
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

def manage_fields(section_type):
    """
    Provides UI to add or remove fields dynamically for Leads or Personnel.
    
    :param section_type: 'lead' or 'personnel'
    """
    if section_type == 'lead':
        fields = st.session_state['lead_fields']
        title = "Manage Lead Fields"
        add_field_key = 'add_lead_field'
        remove_field_key = 'remove_lead_field'
    else:
        fields = st.session_state['personnel_fields']
        title = "Manage Personnel Fields"
        add_field_key = 'add_personnel_field'
        remove_field_key = 'remove_personnel_field'
    
    st.subheader(title)
    
    # Add Field
    new_field = st.text_input(f"Add new {'Lead' if section_type == 'lead' else 'Personnel'} field:", key=add_field_key)
    if st.button(f"Add {'Lead' if section_type == 'lead' else 'Personnel'} Field", key=f"add_field_btn_{section_type}"):
        if new_field and new_field not in fields:
            fields.append(new_field)
            if section_type == 'lead':
                st.session_state['lead_fields'] = fields
            else:
                st.session_state['personnel_fields'] = fields
            st.success(f"Added new {'Lead' if section_type == 'lead' else 'Personnel'} field: {new_field}")
            logger.info(f"Added new {'Lead' if section_type == 'lead' else 'Personnel'} field: {new_field}")
        else:
            st.warning("Field name is empty or already exists.")
    
    # Remove Field
    field_to_remove = st.selectbox(f"Select {'Lead' if section_type == 'lead' else 'Personnel'} field to remove:", fields, key=remove_field_key)
    if st.button(f"Remove {'Lead' if section_type == 'lead' else 'Personnel'} Field", key=f"remove_field_btn_{section_type}"):
        if field_to_remove in fields:
            fields.remove(field_to_remove)
            if section_type == 'lead':
                st.session_state['lead_fields'] = fields
            else:
                st.session_state['personnel_fields'] = fields
            st.success(f"Removed {'Lead' if section_type == 'lead' else 'Personnel'} field: {field_to_remove}")
            logger.info(f"Removed {'Lead' if section_type == 'lead' else 'Personnel'} field: {field_to_remove}")
        else:
            st.warning("Selected field does not exist.")

def input_leads():
    """
    Section for Input Leads: Generate Leads, Add Leads Manually, or Search Leads via Conference.
    Includes dynamic field management for Leads and Personnel.
    """
    leads_list = []
    # ==========================
    # Initialize Session State
    # ==========================
    if 'leads' not in st.session_state:
        st.session_state['leads'] = []

    if 'leads_df' not in st.session_state:
        st.session_state['leads_df'] = pd.DataFrame(columns=["Entity"])

    if 'context' not in st.session_state:
        st.session_state['context'] = ""

    if 'step' not in st.session_state:
        st.session_state['step'] = 1

    # ==========================
    # Input Leads Section
    # ==========================
    st.subheader("Input Leads")

    # Context Input
    context = st.text_area("Context:", value=st.session_state['context'])
    st.session_state['context'] = context

    # Lead Input Options
    option = st.radio(
        "Choose how to input leads:",
        ["Generate Leads", "Add Leads Manually", "Search Leads via Conference"],
        key='lead_input_option'
    )

    # ==========================
    # Manage Dynamic Fields
    # ==========================
    with st.expander("Manage Lead and Personnel Fields"):
        manage_fields('lead')
        manage_fields('personnel')

    # ==========================
    # Generate Leads Section
    # ==========================
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
                for lead in leads:
                    # Initialize lead data with dynamic fields
                    lead_data = {field: "Not Available" for field in st.session_state['lead_fields']}
                    lead_data["Entity"] = lead["Entity"]
                    lead_data["Type"] = "Company"
                    lead_data["Source URLs"] = lead.get("Source URLs", "")
                    
                    # Initialize Personnel fields
                    personnel_data = {field: "Not Available" for field in st.session_state['personnel_fields']}
                    personnel_data["Type"] = "Personnel"
                    personnel_data["Entity"] = lead["Entity"]
                    
                    # Append to leads_combined_df
                    combined_row = {**lead_data, **personnel_data}
                    st.session_state['leads_combined_df'] = pd.concat(
                        [st.session_state['leads_combined_df'], pd.DataFrame([combined_row])],
                        ignore_index=True
                    )

                # Append to the existing leads list
                st.session_state['leads'].extend(leads)

                # Create a DataFrame from the new leads with 'Entity' column
                new_leads_df = pd.DataFrame(leads, columns=["Entity"])

                # Append to the existing leads_df DataFrame
                st.session_state['leads_df'] = pd.concat(
                    [st.session_state['leads_df'], new_leads_df],
                    ignore_index=True
                )

                # Update the 'step' if necessary
                st.session_state['step'] = max(st.session_state['step'], 2)

                st.success(f"Leads generated successfully! Total leads: {len(st.session_state['leads'])}")
                logger.info(f"Generated {len(leads)} leads.")
            else:
                logger.error("Lead generation failed.")
                st.error("Failed to generate leads. Please check the logs for more details.")
                st.stop()

    # ==========================
    # Add Leads Manually Section
    # ==========================
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

                    # Initialize lead data with dynamic fields
                    lead_data = {field: "Not Available" for field in st.session_state['lead_fields']}
                    lead_data["Entity"] = name
                    lead_data["Type"] = "Company"
                    lead_data["Source URLs"] = ""

                    # Initialize Personnel fields
                    personnel_data = {field: "Not Available" for field in st.session_state['personnel_fields']}
                    personnel_data["Type"] = "Personnel"
                    personnel_data["Entity"] = name

                    # Append to leads_combined_df
                    combined_row = {**lead_data, **personnel_data}
                    st.session_state['leads_combined_df'] = pd.concat(
                        [st.session_state['leads_combined_df'], pd.DataFrame([combined_row])],
                        ignore_index=True
                    )

                    leads_list.append({
                        "Entity": name
                    })
                else:
                    logger.warning(f"Invalid format in line: {line}")
                    st.warning(f"Invalid format in line: {line}. Please use 'Entity'.")

            if leads_list:
                # Convert existing leads to a set for faster lookup
                existing_entities = set(st.session_state['leads_df']['Entity'].tolist()) if 'Entity' in st.session_state['leads_df'].columns else set()

                # Filter out any leads that already exist
                new_unique_leads = [lead for lead in leads_list if lead["Entity"] not in existing_entities]

                if new_unique_leads:
                    # Append to the existing leads list
                    st.session_state['leads'].extend(new_unique_leads)
                    logger.debug(f"new_unique_leads: {new_unique_leads[:5]}") 
                    
                    # Create DataFrame for new leads with 'Entity' column
                    new_leads_df = pd.DataFrame(new_unique_leads, columns=["Entity"])

                    # Append to the existing leads_df DataFrame
                    st.session_state['leads_df'] = pd.concat(
                        [st.session_state['leads_df'], new_leads_df],
                        ignore_index=True
                    )

                    # Add company entries to the combined DataFrame
                    for lead in new_unique_leads:
                        # Initialize lead data with dynamic fields
                        lead_data = {field: "Not Available" for field in st.session_state['lead_fields']}
                        lead_data["Entity"] = lead["Entity"]
                        lead_data["Type"] = "Company"
                        lead_data["Source URLs"] = ""

                        # Initialize Personnel fields
                        personnel_data = {field: "Not Available" for field in st.session_state['personnel_fields']}
                        personnel_data["Type"] = "Personnel"
                        personnel_data["Entity"] = lead["Entity"]

                        # Append to leads_combined_df
                        combined_row = {**lead_data, **personnel_data}
                        st.session_state['leads_combined_df'] = pd.concat(
                            [st.session_state['leads_combined_df'], pd.DataFrame([combined_row])],
                            ignore_index=True
                        )

                    st.success(f"Added {len(new_unique_leads)} new leads successfully! Total leads: {len(st.session_state['leads'])}")
                    logger.info(f"Added {len(new_unique_leads)} leads manually.")
                else:
                    st.info("No new unique leads to add.")
            else:
                logger.error("No valid leads entered.")
                st.error("No valid leads entered. Please ensure each line contains 'Entity'.")
                st.stop()

    # ==========================
    # Search Leads via Conference Section
    # ==========================
    elif option == "Search Leads via Conference":
        st.write("### Search Leads via Conference")
        st.write("Enter the conference details below:")
        conference_input = st.text_input("Enter the conference name or URL:", key='conference_input')

        search_btn = st.button("Search Leads", key='search_conference_leads_btn')
        if search_btn:
            if not context.strip():
                logger.error("Lead search via conference attempted with empty context.")
                st.error("Please provide a context for lead search.")
                st.stop()
            if not conference_input.strip():
                st.warning("Please enter a conference name or URL.")
            else:
                with st.spinner('Searching for leads associated with the conference...'):
                    leads_found = search_leads_via_conference(conference_input, context)
                    if leads_found:
                        st.success(f"Found {len(leads_found)} leads related to the conference.")
                        
                        # Ensure leads_found is a list of dictionaries with 'Entity' key
                        new_conference_leads = [{"Entity": lead["Entity"]} for lead in leads_found if "Entity" in lead]

                        # Remove duplicates based on 'Entity'
                        existing_entities = set(st.session_state['leads_df']['Entity'].tolist()) if 'Entity' in st.session_state['leads_df'].columns else set()
                        unique_conference_leads = [lead for lead in new_conference_leads if lead["Entity"] not in existing_entities]

                        if unique_conference_leads:
                            # Append to the existing leads list
                            st.session_state['leads'].extend(unique_conference_leads)
                            logger.debug(f"Unique conference leads: {unique_conference_leads[:5]}")
                            
                            # Create DataFrame for new conference leads with 'Entity' column
                            conference_leads_df = pd.DataFrame(unique_conference_leads, columns=["Entity"])

                            # Append to the existing leads_df DataFrame
                            st.session_state['leads_df'] = pd.concat(
                                [st.session_state['leads_df'], conference_leads_df],
                                ignore_index=True
                            )

                            # Add company entries to the combined DataFrame
                            for lead in unique_conference_leads:
                                # Initialize lead data with dynamic fields
                                lead_data = {field: "Not Available" for field in st.session_state['lead_fields']}
                                lead_data["Entity"] = lead["Entity"]
                                lead_data["Type"] = "Company"
                                lead_data["Source URLs"] = ""

                                # Initialize Personnel fields
                                personnel_data = {field: "Not Available" for field in st.session_state['personnel_fields']}
                                personnel_data["Type"] = "Personnel"
                                personnel_data["Entity"] = lead["Entity"]

                                # Append to leads_combined_df
                                combined_row = {**lead_data, **personnel_data}
                                st.session_state['leads_combined_df'] = pd.concat(
                                    [st.session_state['leads_combined_df'], pd.DataFrame([combined_row])],
                                    ignore_index=True
                                )

                            st.success(f"Added {len(unique_conference_leads)} new conference leads successfully!")
                            logger.info(f"Added {len(unique_conference_leads)} conference leads.")
                        else:
                            st.info("All found conference leads are already present in the leads list.")
                    else:
                        st.warning("No leads found for the provided conference. Please try a different conference name or URL.")

    # ==========================
    # Display and Manage Leads
    # ==========================
    if 'leads_df' in st.session_state and not st.session_state['leads_df'].empty:
        st.write("### Leads")
        st.write("You can edit the leads below:")

        # Streamlit's data_editor inherently provides row selection on the left.
        # No need for additional checkboxes or selection columns.

        # Prepare column configurations dynamically
        column_config = {}
        for field in st.session_state['lead_fields']:
            column_config[field] = st.column_config.TextColumn(field)

        for field in st.session_state['personnel_fields']:
            column_config[field] = st.column_config.TextColumn(field)

        column_config["Entity"] = st.column_config.TextColumn("Entity")

        edited_leads_df = st.data_editor(
            st.session_state['leads_df'],
            num_rows="dynamic",
            use_container_width=True,
            column_config=column_config,
            key='leads_editor',
            height=400
        )

        # Save Changes Button to commit the edits
        save_btn = st.button('Save Selection', key='save_leads_changes')
        if save_btn:
            # Update the leads_df with any edits made in the data editor
            if isinstance(edited_leads_df, pd.DataFrame):
                st.session_state['leads_df'] = edited_leads_df.copy()
                # Update the 'leads' list to reflect changes
                st.session_state['leads'] = edited_leads_df.to_dict('records')
                st.success("Selections saved successfully!")
                logger.info("Selections saved successfully via data editor.")
            else:
                logger.error("Edited leads_df is not a DataFrame.")
                st.error("An error occurred while saving selections. Please try again.")

        # Provide Download Option for All Leads
        st.write("### Download All Leads")
        display_lead_information(
            df=st.session_state['leads_combined_df'],
            button_label="All Leads",
            filename="all_leads"
        )

    # ==========================
    # Final Section
    # ==========================
    st.session_state['input_leads'] = True
    st.success("Input Leads section loaded successfully!")

def scrape_lead_information(leads_to_process=None, is_conference=False):
    """
    Section for Analyze Lead Information: Perform per-field searches and scrape information,
    then extract and analyze persons associated with each lead.
    Includes dynamic field handling for Leads and Personnel.

    :param leads_to_process: Optional list of leads to process. If None, process all leads in st.session_state['leads'].
    :param is_conference: Boolean indicating if the leads are conference leads.
    """
    st.subheader("Search and Analyse Lead Information")
    
    # Initialize overall progress elements
    progress_bar = st.empty()
    status_text = st.empty()
    
    # Initialize personnel progress elements outside the loop
    person_status_text = st.empty()
    person_progress_bar = st.progress(0)
    
    st.header("Analyze Lead Information")
    default_columns = st.session_state['lead_fields'] + st.session_state['personnel_fields']
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
    
    # **Add input for maximum number of persons to process per lead**
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
        if st.session_state.get('context', '') == '':
            logger.error("Lead scraping attempted with empty context.")
            st.warning("Please provide a context for lead scraping.")
            return
        
        # Determine which leads to process
        if leads_to_process is not None:
            leads = leads_to_process
        else:
            leads = st.session_state.get('leads', [])
        
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
            st.session_state['leads_info_df'] = pd.DataFrame()
            st.session_state['person_leads_df'] = pd.DataFrame()
        
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
                # Append lead information to the combined DataFrame as a company entry
                combined_lead_data = {field: lead_info.get(field, "Not Available") for field in st.session_state['lead_fields']}
                combined_lead_data["Entity"] = lead_info.get("Entity", lead_name)
                combined_lead_data["Type"] = "Company"
                combined_lead_data["Source URLs"] = lead_info.get("Source URLs", "")
                
                # Initialize Personnel fields with default values
                personnel_data = {field: "Not Available" for field in st.session_state['personnel_fields']}
                personnel_data["Type"] = "Personnel"
                personnel_data["Entity"] = lead_info.get("Entity", lead_name)
                
                # Append to leads_combined_df
                combined_row = {**combined_lead_data, **personnel_data}
                st.session_state['leads_combined_df'] = pd.concat(
                    [st.session_state['leads_combined_df'], pd.DataFrame([combined_row])],
                    ignore_index=True
                )
                
                st.session_state['processed_leads'].append(lead_info)

                # Update the UI with the newly processed lead
                with st.container():
                    st.markdown(f"### {lead_info.get('Entity', 'Unknown')}")
                    st.json(lead_info, expanded=False)
                    st.markdown("---")
            
            # Extract and Analyze Persons Associated with the Lead
            if not is_conference:
                # Create a DataFrame for the current lead's information
                current_lead_df = pd.DataFrame([lead_info])
                persons = extract_persons(current_lead_df)
            else:
                # If it's a conference, process all processed leads
                processed_leads_df = pd.DataFrame(st.session_state['processed_leads'])
                persons = extract_persons(processed_leads_df)

            if not persons:
                logger.warning(f"No persons found associated with the lead '{lead_name}'.")
                st.warning(f"No persons found associated with the lead '{lead_name}'.")
                person_progress_bar.progress(1.0)  # Mark personnel progress as complete for this lead
            else:
                # **Limit the number of persons to process per lead**
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
                    person_urls = perform_google_search(person_name, num_results=1, keywords=person_search_keywords, lead_name=associated_lead)
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
                    with open(os.path.join("database", person_name.replace(" ", "_") + ".txt"), "w") as f:
                        f.write(cleaned_text)
                    source_urls = sources
                    person_info = extract_person_info_with_llm(cleaned_text, person_name, sources, key_words=person_search_keywords)
                    if person_info:
                        # Append personnel information as a new row linked to the company
                        combined_personnel_data = {field: person_info.get(field.replace("Personnel ", ""), "Not Available") for field in st.session_state['personnel_fields']}
                        combined_personnel_data["Type"] = "Personnel"
                        combined_personnel_data["Entity"] = associated_lead
                        combined_personnel_data["Source URLs"] = ", ".join(person_info.get("Source URLs", []))
                        combined_personnel_data["Personnel Name"] = person_info.get("Name", "Not Available")
                        combined_personnel_data["Personnel Title"] = person_info.get("Title", "Not Available")
                        combined_personnel_data["Personnel Email"] = person_info.get("Email", "Not Available")
                        combined_personnel_data["Personnel Phone"] = person_info.get("Phone Number", "Not Available")
                        
                        st.session_state['leads_combined_df'] = pd.concat(
                            [st.session_state['leads_combined_df'], pd.DataFrame([combined_personnel_data])],
                            ignore_index=True
                        )
                        
                        st.success(f"Added personnel '{person_name}' successfully!")
                        logger.info(f"Added personnel '{person_name}' successfully.")
                    else:
                        logger.info(f"Skipping person '{person_name}' due to incomplete information.")
                        st.info(f"Skipped adding incomplete entry for '{person_name}'.")

                    # Update personnel progress bar
                    person_progress += person_progress_increment
                    person_progress_bar.progress(min(person_progress, 1.0))
                
                if is_conference:
                    st.success(f"Person information scraped successfully for conference leads!")
                    logger.info(f"Person information scraped successfully for conference leads.")
                else:
                    st.success(f"Person information scraped successfully for lead '{lead_name}'!")
                    logger.info(f"Person information scraped successfully for lead '{lead_name}'.")
                
                # Display the person leads information
                st.write(f"### Persons Associated with {lead_name}")

                # Filter the DataFrame to include only persons associated with the current lead
                filtered_persons_df = st.session_state['leads_combined_df'][
                    (st.session_state['leads_combined_df']['Entity'] == lead_name) &
                    (st.session_state['leads_combined_df']['Type'] == "Personnel")
                ]

                display_lead_information(
                    df=filtered_persons_df,
                    button_label="Download Person Leads Information",
                    filename=f"person_leads_info_{lead_name.replace(' ', '_')}"
                )
    
        # After all leads are processed, mark progress as complete
        progress_bar.progress(1.0)
        person_progress_bar.progress(1.0)
        status_text.text("Lead information scraping completed!")
        st.success("All leads have been processed successfully.")
        st.session_state['is_scraping'] = False

    # ==========================
    # Other Sections (Rank Leads, Analytics, Author Papers, Download Data)
    # ==========================

    # def rank_leads_section():
    #     st.subheader("Rank Leads (BANT)")
    #     # Existing functionality remains unchanged
    #     st.write("Ranking leads based on BANT criteria.")
    #     # Implement the ranking logic here

    # def analytics_section():
    #     st.subheader("Analytics")
    #     # Existing functionality remains unchanged
    #     st.write("Performing analytics on the leads data.")
    #     # Implement the analytics logic here

    # def author_papers_section():
    #     st.subheader("Author Papers")
    #     # Existing functionality remains unchanged
    #     st.write("Fetching and displaying recent papers from authors.")
    #     # Implement the author papers logic here

    # def download_data_section():
    #     st.subheader("Download Data")
    #     # Existing functionality remains unchanged
    #     st.write("Download your leads data in various formats.")
    #     # Implement the download logic here

# Run the selected section
if st.session_state['menu_selection'] == "Input Leads":
    input_leads()
elif st.session_state['menu_selection'] == "Analyse Lead Information":
    scrape_lead_information()
# elif st.session_state['menu_selection'] == "Rank Leads (BANT)":
#     rank_leads_section()
# elif st.session_state['menu_selection'] == "Analytics":
#     analytics_section()
# elif st.session_state['menu_selection'] == "Author Papers":
#     author_papers_section()
# elif st.session_state['menu_selection'] == "Download Data":
#     download_data_section()
else:
    st.write("Select a section from the menu above.")