# app.py

import streamlit as st
import pandas as pd
from streamlit_tags import st_tags
from io import BytesIO
import re
import seaborn as sns
import time
import random
import json
from wordcloud import WordCloud
import logging
from dotenv import load_dotenv
import numpy as np
import matplotlib.pyplot as plt
from streamlit_option_menu import option_menu

# Import functions from network.py
from network import (
    display_author_information_cards,
    build_network_graph_with_relevance,
    calculate_relevance_with_llm,
    visualize_network,
    fetch_recent_papers
)  

# Import functions from analytics.py
from analytics import (
    rank_leads_with_bant,
    generate_bant_report

)
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
    scrape_information_field,
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
        elif dep_key == "rank_leads_bant":
            rank_leads_section()
        elif dep_key == "analytics":
            analytics_section()
        elif dep_key == "extract_persons":
            extract_persons_section()
        elif dep_key == "author_papers":
            author_papers_section()
        elif dep_key == "download_data":
            download_data_section()
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
    'step', 'context'
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
        else:
            st.session_state[var] = []

# Define menu options
menu_options = [
    "Input Leads", "Scrape Lead Information", "Rank Leads (BANT)", 
    "Analytics", "Extract Persons", "Author Papers", "Download Data"
]

# Create the option menu
selected = option_menu(
    menu_title=None,
    options=menu_options,
    icons=["pencil", "search", "bar-chart", "activity", "people", "book", "download"],
    menu_icon="cast",
    orientation="horizontal",
    key='main_menu'
)

# Update menu selection in session state
if selected != st.session_state['menu_selection']:
    st.session_state['menu_selection'] = selected

# ==========================
# Define Section Functions
# ==========================

def input_leads():
    """
    Section for Input Leads: Generate Leads or Add Leads Manually.
    """
    st.subheader("Input Leads")

    context = st.text_area("Context:", value=st.session_state['context'])
    st.session_state['context'] = context

    option = st.radio(
            "Choose how to input leads:",
            ["Generate Leads", "Add Leads Manually", "Search Leads via Conference"], # "Search Leads via Conference" - maybe this is not a necessary addition, but for now we keep it - otherwise we need an LLM pass
            key='lead_input_option'
        )

    if option == "Generate Leads":
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
                st.error("Please provide a context for lead generation.")
                st.stop()
            elif not lead_types:
                logger.error("Lead generation attempted without specifying lead types.")
                st.warning("Please specify at least one lead type.")
            else:
                with st.spinner('Generating leads...'):
                    leads = generate_leads_with_llm(context, num_leads_total, lead_types)
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
                    st.stop()

    if option == "Add Leads Manually":
        st.write("Enter your leads below:")

        leads_input = st.text_area(
            "Enter one lead per line, in the format 'Name,Type':",
            height=150,
            key='manual_leads_input'
        )

        add_leads_btn = st.button("Add Leads", key='add_leads_btn')
        if add_leads_btn:
            if not context.strip():
                logger.error("Lead generation attempted with empty context.")
                st.warning("Please provide a context for lead generation.")
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
                st.stop()

    elif option == "Search Leads via Conference":
        st.write("Enter the conference details below:")
        conference_input = st.text_input("Enter the conference name or URL:", key='conference_input')

        search_btn = st.button("Search Leads", key='search_conference_leads_btn')
        if search_btn:
            if not context.strip():
                logger.error("Lead generation attempted with empty context.")
                st.error("Please provide a context for lead generation.")
                st.stop()
            if not conference_input.strip():
                st.warning("Please enter a conference name or URL.")
            else:
                with st.spinner('Searching for leads associated with the conference...'):
                    leads_list = search_leads_via_conference(conference_input, context)
                    if leads_list:
                        # Append to the existing leads list
                        st.session_state['leads'].extend(leads_list)
                        
                        # Create a DataFrame from the new leads
                        new_leads_df = pd.DataFrame(leads_list, columns=["Name", "Type"])
                        
                        # Append to the existing leads_df DataFrame
                        st.session_state['leads_df'] = pd.concat([st.session_state['leads_df'], new_leads_df], ignore_index=True)
                        
                        st.success(f"Leads extracted successfully! Total leads: {len(st.session_state['leads'])}")
                        logger.info(f"Extracted {len(leads_list)} leads from conference input.")
                    else:
                        st.warning("No leads found for the provided conference.")

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
            display_leads_df = st.session_state['leads_df']
            display_leads_df = display_leads_df.rename(columns={"Name": "Company/Group Name", "Type": "Category"})
            display_leads_df = display_leads_df[["Company/Group Name", "Category"]]

            display_leads_df = display_leads_df.rename(columns={"Company/Group Name": "Name", "Category": "Type"})

            display_and_download(
                df=display_leads_df,
                button_label="Leads",
                filename="leads"
            )
        except Exception as e:
            logger.error(f"Error editing leads: {e}")
            st.error(f"Error editing leads: {e}")
            st.stop()
    
    existing_leads = set(st.session_state['leads'])
    unique_leads = [lead for lead in leads_list if lead not in existing_leads]
    st.session_state['leads'] = unique_leads
    st.session_state['input_leads'] = True
    st.success("Input Leads completed successfully!")

def scrape_lead_information():
    """
    Section for Scrape Lead Information: Perform per-field searches and scrape information.
    """
    st.subheader("Search and Scrape Lead Information")
    progress_bar = st.empty()
    status_text = st.empty()
    lead_placeholder = st.empty()
    
    st.header("Scrape Lead Information")
    default_columns = ["Company/Group Name", "CEO/PI", "Researchers", "Grants", "Phone Number", "Email", "Country", "University", "Summary", "Contacts"]
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
        if not st.session_state.get('context_input', '').strip():
                logger.error("Lead generation attempted with empty context.")
                st.warning("Please provide a context for lead generation.")
        if not st.session_state['leads']:
            logger.error("Lead scraping attempted without any generated or added leads.")
            st.error("No leads available. Please add or generate leads first.")
            st.stop()
        elif not columns_to_retrieve:
            logger.error("Lead scraping attempted without specifying information fields.")
            st.error("Please select at least one information field to retrieve.")
            st.stop()
        else:
            if not st.session_state['is_scraping']:
                st.session_state['is_scraping'] = True
                st.session_state['processed_leads'] = []
                st.session_state['leads_info_df'] = pd.DataFrame()
            
            total_leads = len(st.session_state['leads'])
            progress_increment = 100 / total_leads
            progress = 0
            progress_bar.progress(0)
            status_text.text("Starting lead information scraping...")
            
            for idx, (lead_name, lead_category) in enumerate(st.session_state['leads']):
                status_text.text(f"Processing Lead {idx + 1} of {total_leads}: {lead_name}")
                
                # Scrape per-field information
                field_data_list = []
                for field in columns_to_retrieve:
                    if field in ["Company/Group Name", "Summary", "Category"]:
                        continue  # Skip fields that don't require scraping
                    field_data = scrape_information_field(lead_name, field, num_search_results=1)
                    field_data_list.append(field_data)
                    time.sleep(random.uniform(1, 3))  # Respect rate limits
                
                # Extract lead information using GPT with chunking
                lead_info = extract_lead_info_with_llm_per_field(lead_name, lead_category, field_data_list)
                if lead_info:
                    st.session_state['leads_info_df'] = pd.concat(
                        [st.session_state['leads_info_df'], pd.DataFrame([lead_info])],
                        ignore_index=True
                    )
                    st.session_state['processed_leads'].append(lead_info)
                    
                    # Update the UI with the newly processed lead
                    with lead_placeholder.container():
                        st.markdown(f"### {lead_info.get('Company/Group Name', 'Unknown')}")
                        st.json(lead_info, expanded=False)
                        st.markdown("---")
                
                # Update progress bar
                progress += progress_increment
                progress_bar.progress(min(progress, 100) / 100)
            
            status_text.text("Lead information scraping completed!")
            st.success("All leads have been processed successfully.")
            st.session_state['is_scraping'] = False

    # Display the Leads Information DataFrame if it exists
    if not st.session_state['leads_info_df'].empty:
        st.write("### Leads Information")
        display_lead_information(
            df=st.session_state['leads_info_df'],
            button_label="Leads Information",
            filename="leads_info"
        )
    st.session_state['scrape_lead_information'] = True
    st.success("Lead information scraped successfully!")

def analytics_section():
    """
    Section for Analytics: Provide various analytics and visualizations.
    """
    st.subheader("Analytics")
    # if st.session_state.get('ranked_leads_df', pd.DataFrame()).empty:
    #     st.warning("No ranked leads available. Please rank the leads first.")
    #     if st.button("Go to Rank Leads"):
    #         st.session_state['menu_selection'] = "Rank Leads (BANT)"
    #         st.rerun()
    #     return

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

            # Plotting Number of Grants per Lead
            fig, ax = plt.subplots(figsize=(10, 6))
            x = np.arange(len(st.session_state['ranked_leads_df']))
            ax.bar(x, st.session_state['ranked_leads_df']['Grant Count'])
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
                st.stop()

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
                st.stop()
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
                st.stop()

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
                st.stop()
            except Exception as e:
                logger.error(f"Error plotting individual names: {e}")
                st.error(f"Error plotting individual names: {e}")
                st.stop()
        except Exception as e:
            logger.error(f"Error counting individual names: {e}")
            st.error(f"Error counting individual names: {e}")
            st.stop()

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
                st.markdown("---")

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
                    st.stop()
            else:
                logger.error("'Company/Group Name' column missing in person_leads_df after renaming.")
                st.error("'Company/Group Name' column missing in person_leads_df after renaming.")
                st.stop()
        else:
            logger.warning("No person leads data available.")
            st.warning("No person leads data available. Please extract and scrape persons first.")

        # ==========================
        # BANT-Specific Analytics
        # ==========================
        if 'ranked_leads_df' in st.session_state and not st.session_state['ranked_leads_df'].empty:
            st.markdown("### BANT Analysis Insights")
            
            # Distribution of BANT Scores
            st.markdown("#### Distribution of BANT Scores")
            bant_scores = ['Budget Score', 'Authority Score', 'Need Score', 'Timeline Score', 'Overall BANT Score']
            for score in ['Budget Score', 'Authority Score', 'Need Score', 'Timeline Score']:
                plt.figure(figsize=(8, 4))
                plt.hist(st.session_state['ranked_leads_df'][score], bins=10, color='skyblue', edgecolor='black')
                plt.title(f'Distribution of {score}')
                plt.xlabel(score)
                plt.ylabel('Number of Leads')
                plt.tight_layout()
                st.pyplot(plt)
            
            # Overall BANT Score
            plt.figure(figsize=(8, 4))
            plt.hist(st.session_state['ranked_leads_df']['Overall BANT Score'], bins=10, color='salmon', edgecolor='black')
            plt.title('Distribution of Overall BANT Scores')
            plt.xlabel('Overall BANT Score')
            plt.ylabel('Number of Leads')
            plt.tight_layout()
            st.pyplot(plt)
            
            # Recommendations Breakdown
            st.markdown("#### Recommendations Breakdown")
            recommendations_counts = st.session_state['ranked_leads_df']['Recommendations'].value_counts()
            plt.figure(figsize=(8, 6))
            plt.pie(recommendations_counts.values, labels=recommendations_counts.index, autopct='%1.1f%%', startangle=140)
            plt.title('Recommendations Distribution')
            plt.axis('equal')
            plt.tight_layout()
            st.pyplot(plt)
            
            # Correlation Heatmap of BANT Scores
            st.markdown("#### Correlation Between BANT Scores")
            correlation = st.session_state['ranked_leads_df'][['Budget Score', 'Authority Score', 'Need Score', 'Timeline Score', 'Overall BANT Score']].corr()
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(correlation, annot=True, cmap='coolwarm', ax=ax)
            plt.title('Correlation Heatmap of BANT Scores')
            plt.tight_layout()
            st.pyplot(fig)
        else:
            st.warning("No BANT analysis data available for analytics.")


def extract_persons_section():
    """
    Section for Extract Persons: Extract and Scrape Persons Associated with Leads.
    """
    st.subheader("Extract and Scrape Persons Associated with Leads")
    
    # Add input for additional search keywords
    default_person_keywords = ["email", "phone number", "profile", "CV", "LinkedIn", "research", "publications"]
    person_search_keywords = st_tags(
        label='',
        text='Add or remove keywords to guide the person search:',
        value=default_person_keywords,
        suggestions=default_person_keywords,
        maxtags=20,
        key='person_search_keywords'
    )

    extract_persons_btn = st.button("Extract and Scrape Persons", key='extract_persons_btn')
    if extract_persons_btn:
        if st.session_state['leads_info_df'].empty:
            logger.error("Person extraction attempted without lead information.")
            st.error("No lead information available. Please scrape lead information first.")
            st.stop()
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
                person_placeholder = st.empty()
                for idx, (person_name, associated_lead) in enumerate(persons):
                    person_placeholder.text(f"Processing Person {idx + 1}/{len(persons)}: {person_name} (Company: {associated_lead})")
                    # Include the additional keywords in the search
                    person_urls = perform_google_search(person_name, additional_keywords=person_search_keywords)
                    if not person_urls:
                        logger.warning(f"No URLs found for '{person_name}'.")
                        continue
                    scraped_text = ""
                    sources = []
                    # Scrape the URLs
                    for url in person_urls:
                        logger.info(f"Scraping URL for '{person_name}': {url}")
                        with st.spinner(f"Scraping URL for '{person_name}': {url}"):
                            scraped_text += scrape_landing_page(url) + " "
                            sources.append(url)
                    if not scraped_text.strip():
                        logger.warning(f"No text scraped from URLs for '{person_name}'.")
                        continue
                    cleaned_text = clean_text(scraped_text)
                    source_urls = ', '.join(sources)
                    person_info = extract_person_info_with_llm(cleaned_text, person_name, source_urls)
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
                    st.stop()
            else:
                logger.error("Person information scraping failed.")
                st.error("Failed to scrape person information. Please check the logs for more details.")
                st.stop()

    # Display the Person Leads DataFrame if it exists
    if not st.session_state['person_leads_df'].empty:
        st.write("### Person Leads Information")
        display_and_download(
            df=st.session_state['person_leads_df'],
            button_label="Person Leads Information",
            filename="person_leads_info"
        )

def author_papers_section():
    """
    Section for Author Papers: Fetch, summarize, and visualize author papers and co-authors.
    """
    st.subheader("Author Papers and Network Visualization")
    
    # Check if person leads are available
    if 'person_leads_df' not in st.session_state or st.session_state['person_leads_df'].empty:
        st.warning("No persons available. Please extract and scrape persons first.")
        if st.button("Go to Extract Persons"):
            st.session_state['menu_selection'] = "Extract Persons"
        return
    
    authors = st.session_state['person_leads_df']['Name'].unique().tolist()
    
    # Manual Search Input
    st.markdown("### Search for a Person")
    search_name = st.text_input("Enter a person's name to search:", key='manual_search_input')
    search_btn = st.button("Search Person", key='search_person_btn')
    
    if search_btn:
        if not st.session_state.get('context_input', '').strip():
                logger.error("Lead generation attempted with empty context.")
                st.warning("Please provide a context for lead generation.")
        if not search_name.strip():
            st.warning("Please enter a valid name.")
        else:
            # Search for the person in person_leads_df
            person_data = st.session_state['person_leads_df'][
                st.session_state['person_leads_df']['Name'].str.lower() == search_name.strip().lower()
            ]
            if person_data.empty:
                st.error(f"No information found for '{search_name}'. Please ensure the name is correct.")
                st.stop()
            else:
                person_info = person_data.iloc[0].to_dict()
                st.markdown(f"### {person_info.get('Name', 'Unknown')}")
                st.json(person_info, expanded=True)
                st.markdown("---")
    
    # Select Author from Dropdown
    selected_author = st.selectbox("Select an Author to Analyze", options=authors, key='selected_author_dropdown')
    
    analyze_btn = st.button("Analyze Author", key='analyze_author_btn')
    if analyze_btn:
        if not selected_author:
            st.warning("Please select an author.")
            return
        
        with st.spinner(f'Fetching papers for {selected_author}...'):
            papers = fetch_recent_papers(selected_author)
        
        if not papers:
            st.error(f"No papers found for {selected_author}.")
            st.stop()
            return
        
        author_papers = []
        co_authors_set = set()
        for paper in papers:
            title = paper.get('title', 'No Title')
            abstract = paper.get('abstract', 'No Abstract Available.')
            summary = summarize_paper(title, abstract)
            co_authors = [author['Name'] for author in paper.get('authors', []) if author['Name'] != selected_author]
            author_papers.append({
                'title': title,
                'abstract': abstract,
                'summary': summary,
                'co_authors': co_authors
            })
            co_authors_set.update(co_authors)
        
        # Fetch co-authors' papers and build the network
        co_authors_papers = {}
        for co_author in co_authors_set:
            papers = fetch_recent_papers(co_author, limit=1)  # Fetch fewer papers for co-authors
            co_authors_papers[co_author] = papers
        
        # Combine all authors and their papers
        all_authors_papers = {selected_author: author_papers}
        all_authors_papers.update(co_authors_papers)
        
        # Summarize papers and collect abstracts per author
        authors_info = {}
        for author_name, papers in all_authors_papers.items():
            summarized_papers = []
            abstracts = []
            for paper in papers:
                title = paper.get('title', 'No Title')
                abstract = paper.get('abstract', 'No Abstract Available.')
                summary = summarize_paper(title, abstract)
                abstracts.append(abstract)
                co_authors = [a['Name'] for a in paper.get('authors', []) if a['Name'] != author_name]
                summarized_papers.append({
                    'title': title,
                    'abstract': abstract,
                    'summary': summary,
                    'co_authors': co_authors
                })
            authors_info[author_name] = {
                'papers': summarized_papers,
                'abstracts': abstracts
            }
        
        # Calculate relevance using llm
        relevance_scores = calculate_relevance_with_llm(authors_info, st.session_state.get('context_input', ''))
        
        # Build Network Graph with relevance scores
        G = build_network_graph_with_relevance(authors_info, relevance_scores)
        
        # Visualize the Network Graph with enhancements
        visualize_network(G, selected_author, context=st.session_state.get('context_input', ''))
        
        # Display Information Cards
        display_author_information_cards(G, selected_author)

def download_data_section():
    """
    Section for Download Data: Download different datasets including BANT reports.
    """
    st.subheader("Download Data")
    
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
                st.stop()
    
    # Option to download Ranked Leads with BANT
    if not st.session_state['ranked_leads_df'].empty:
        st.markdown("### Download Ranked Leads with BANT Scores")
        ranked_download_btn = st.button("Download Ranked Leads with BANT", key='download_ranked_bant_btn')
        if ranked_download_btn:
            try:
                # Generate the BANT report
                bant_report = generate_bant_report(st.session_state['ranked_leads_df'])
                if bant_report:
                    st.download_button(
                        label="Download BANT Report as Excel",
                        data=bant_report,
                        file_name="bant_report.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                    st.success("BANT report downloaded successfully!")
                    logger.info("BANT report downloaded successfully.")
                else:
                    st.error("Failed to generate BANT report.")
                    st.stop()
            except Exception as e:
                logger.error(f"Error downloading BANT report: {e}")
                st.error(f"Error downloading BANT report: {e}")
                st.stop()
    
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
                    st.stop()
    
    # Option to download all data together (ZIP functionality can be implemented as needed)
    st.markdown("### Download All Data")
    all_download_btn = st.button("Download All Data as ZIP", key='download_all_data_btn')
    if all_download_btn:
        logger.warning("ZIP download functionality is not implemented yet.")
        st.warning("ZIP download functionality is not implemented yet.")

def rank_leads_section():
    """
    Section for Rank Leads: Rank the Leads based on BANT analysis.
    """
    st.subheader("Rank the Leads using BANT Analysis")

    # Retrieve context from session_state
    context = st.session_state.get('context', '')

    if not context.strip():
        st.warning("No context provided. Please go back to 'Input Leads' to provide context.")
        if st.button("Go to Input Leads"):
            st.session_state['menu_selection'] = "Input Leads"
            st.rerun()
        return

    st.markdown("**Context for BANT Analysis:**")
    st.text_area("Context for BANT analysis:", value=context, height=150, key='bant_context_input', disabled=True)

    rank_btn = st.button("Perform BANT Analysis and Rank Leads", key='bant_rank_leads_btn')
    if rank_btn:
        if st.session_state['leads_info_df'].empty:
            logger.error("Lead ranking attempted without lead information.")
            st.error("No lead information available. Please scrape lead information first.")
            st.stop()
        else:
            with st.spinner('Performing BANT analysis and ranking leads...'):
                ranked_leads_df = rank_leads_with_bant(st.session_state['leads_info_df'], context)
                st.session_state['ranked_leads_df'] = ranked_leads_df
                st.success("Leads ranked successfully based on BANT analysis!")
                logger.info("Leads ranked successfully using BANT analysis.")

    # Display the Ranked Leads DataFrame if it exists
    if not st.session_state['ranked_leads_df'].empty:
        st.write("### Ranked Leads with BANT Scores")
        try:
            display_leads_df = st.session_state['ranked_leads_df']
            # Select relevant columns for display
            display_columns = ["Company/Group Name", "Category", "Budget Score", "Authority Score",
                               "Need Score", "Timeline Score", "Overall BANT Score", "Recommendations"]
            st.dataframe(display_leads_df[display_columns], height=600)

            # Provide download options
            csv = display_leads_df.to_csv(index=False).encode('utf-8')
            excel_buffer = BytesIO()
            try:
                with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                    display_leads_df.to_excel(writer, index=False)
                excel_data = excel_buffer.getvalue()
            except Exception as e:
                logger.error(f"Error converting ranked_leads_df to Excel: {e}")
                excel_data = None

            st.download_button(
                label="Download Ranked Leads as CSV",
                data=csv,
                file_name="ranked_leads_bant.csv",
                mime='text/csv'
            )

            if excel_data:
                st.download_button(
                    label="Download Ranked Leads as Excel",
                    data=excel_data,
                    file_name="ranked_leads_bant.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
        except Exception as e:
            logger.error(f"Error displaying ranked_leads_df: {e}")
            st.error(f"Error displaying ranked leads: {e}")
            st.stop()


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
    - **Scrape Lead Information:** Fetch detailed information about each lead.
    - **Rank Leads:** Prioritize leads based on configurable weights.
    - **Analytics:** Visualize and analyze lead data.
    - **Extract Persons:** Extract and gather information about individuals associated with leads.
    - **Author Papers:** Analyze authors' papers and visualize co-authorship networks.
    - **Download Data:** Download the processed data in various formats.
    """
)

# Navigation Menu
if st.session_state['menu_selection'] == "Input Leads":
    input_leads()
elif st.session_state['menu_selection'] == "Scrape Lead Information":
    scrape_lead_information()
elif st.session_state['menu_selection'] == "Rank Leads (BANT)":
    rank_leads_section()
elif st.session_state['menu_selection'] == "Analytics":
    analytics_section()
elif st.session_state['menu_selection'] == "Extract Persons":
    extract_persons_section()
elif st.session_state['menu_selection'] == "Author Papers":
    author_papers_section()
elif st.session_state['menu_selection'] == "Download Data":
    download_data_section()