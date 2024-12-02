# app.py

import streamlit as st
import streamlit_authenticator as stauth
import pandas as pd
from streamlit_tags import st_tags
from io import BytesIO
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
import yaml
from yaml.loader import SafeLoader

# Import functions from network.py
from network import (
    display_author_information_cards,
    build_network_graph_with_relevance,
    calculate_relevance_with_llm,
    visualize_network,
    fetch_recent_papers,
)

# Import functions from analytics.py
from analytics import rank_leads_with_bant, generate_bant_report

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
    display_lead_information,
)


st.set_page_config(layout="wide")


# ==========================
# Configure Auth
# ==========================
with open("./auth_config.yml") as file:
    auth_config = yaml.load(file, Loader=SafeLoader)

# Pre-hashing all plain text passwords once
# stauth.Hasher.hash_passwords(config['credentials'])

authenticator = stauth.Authenticate(
    auth_config["credentials"],
    auth_config["cookie"]["name"],
    auth_config["cookie"]["key"],
    auth_config["cookie"]["expiry_days"],
)

# ==========================
# Configure Logging
# ==========================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler("app.log"), logging.StreamHandler()],
)


logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


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
        elif dep_key == "author_papers":
            author_papers_section()
        elif dep_key == "download_data":
            download_data_section()
        else:
            logger.warning(f"No function defined for dependency: {dep}")


try:
    authenticator.login(location="main")
except Exception as e:
    st.error(e)

if st.session_state["authentication_status"] is False:
    st.error("Username/password is incorrect")
elif st.session_state["authentication_status"]:
    st.title("Bateman")

    # Initialize session state variables
    session_vars = [
        "processed_leads",
        "leads_info_df",
        "is_scraping",
        "person_leads_df",
        "leads",
        "leads_df",
        "ranked_leads_df",
        "excel_filename",
        "menu_selection",
        "step",
        "context",
    ]

    for var in session_vars:
        if var not in st.session_state:
            if var.endswith("_df"):
                st.session_state[var] = pd.DataFrame()
            elif var == "excel_filename":
                st.session_state[var] = "leads"
            elif var == "menu_selection":
                st.session_state[var] = "Input Leads"
            elif var == "step":
                st.session_state[var] = 0
            elif var == "context":
                st.session_state[var] = ""
            else:
                st.session_state[var] = []

    # Define menu options
    menu_options = [
        "Input Leads",
        "Analyse Lead Information",
        "Rank Leads (BANT)",
        "Analytics",
        "Author Papers",
        "Download Data",
    ]

    # Create the option menu
    selected = option_menu(
        menu_title=None,
        options=menu_options,
        icons=[
            "pencil",
            "search",
            "bar-chart",
            "activity",
            "people",
            "book",
            "download",
        ],
        menu_icon="cast",
        orientation="horizontal",
        key="main_menu",
    )

    # Update menu selection in session state
    if selected != st.session_state["menu_selection"]:
        st.session_state["menu_selection"] = selected

    # ==========================
    # Define Section Functions
    # ==========================

    def input_leads():
        """
        Section for Input Leads: Generate Leads or Add Leads Manually.
        """
        leads_list = []

        st.subheader("Input Leads")

        context = st.text_area("Context:", value=st.session_state["context"])
        st.session_state["context"] = context

        option = st.radio(
            "Choose how to input leads:",
            [
                "Generate Leads",
                "Add Leads Manually",
                "Search Leads via Conference",
            ],  # "Search Leads via Conference" - maybe this is not a necessary addition, but for now we keep it - otherwise we need an LLM pass
            key="lead_input_option",
        )

        if option == "Generate Leads":
            num_leads_total = st.number_input(
                "Number of leads per type:",
                min_value=1,
                max_value=100,
                value=10,
                step=1,
                key="num_leads_total",
            )
            default_lead_types = ["Research Groups"]
            lead_types = st_tags(
                label="",
                text="Add or remove lead types:",
                value=default_lead_types,
                suggestions=default_lead_types,
                maxtags=10,
                key="lead_types",
            )
            generate_btn = st.button("Generate Leads", key="generate_leads_btn")
            if generate_btn:
                if not context.strip():
                    logger.error("Lead generation attempted with empty context.")
                    st.error("Please provide a context for lead generation.")
                    st.stop()
                elif not lead_types:
                    logger.error(
                        "Lead generation attempted without specifying lead types."
                    )
                    st.warning("Please specify at least one lead type.")
                else:
                    with st.spinner("Generating leads..."):
                        leads = generate_leads_with_llm(
                            context, num_leads_total, lead_types
                        )
                    if leads:
                        # Append to the existing leads list
                        st.session_state["leads"].extend(leads)

                        # Create a DataFrame from the new leads
                        new_leads_df = pd.DataFrame(
                            leads,
                            columns=["Entity", "Type", "University", "City", "Country"],
                        )

                        # Append to the existing leads_df DataFrame
                        st.session_state["leads_df"] = pd.concat(
                            [st.session_state["leads_df"], new_leads_df],
                            ignore_index=True,
                        )

                        # Update the 'step' if necessary
                        st.session_state["step"] = max(st.session_state["step"], 2)

                        st.success(
                            f"Leads generated successfully! Total leads: {len(st.session_state['leads'])}"
                        )
                        logger.info(f"Generated {len(leads)} leads.")
                    else:
                        logger.error("Lead generation failed.")
                        st.error(
                            "Failed to generate leads. Please check the logs for more details."
                        )
                        st.stop()

        if option == "Add Leads Manually":
            st.write("Enter your leads below:")

            leads_input = st.text_area(
                "Enter one lead per line, in the format 'Entity, Type, University, City, Country':",
                height=150,
                key="manual_leads_input",
            )

            add_leads_btn = st.button("Add Leads", key="add_leads_btn")
            if add_leads_btn:
                if not context.strip():
                    logger.error("Lead generation attempted with empty context.")
                    st.warning("Please provide a context for lead generation.")
                for line in leads_input.strip().split("\n"):
                    parts = line.strip().split(",")
                    if len(parts) == 5:
                        name = parts[0].strip()
                        lead_type = parts[1].strip()
                        university = parts[2].strip()
                        city = parts[3].strip()
                        country = parts[4].strip()

                        leads_list.append(
                            {
                                "Entity": name,
                                "Type": lead_type,
                                "University": university,
                                "City": city,
                                "Country": country,
                            }
                        )
                    else:
                        logger.warning(f"Invalid format in line: {line}")
                        st.warning(
                            f"Invalid format in line: {line}. Please use 'Entity, Type, University, City, Country'."
                        )
                if leads_list:
                    # Convert existing leads to a set for faster lookup
                    existing_leads_set = set(
                        tuple(lead.values()) for lead in st.session_state["leads"]
                    )

                    # Filter out any leads that already exist
                    new_unique_leads = [
                        lead
                        for lead in leads_list
                        if tuple(lead.values()) not in existing_leads_set
                    ]

                    if new_unique_leads:
                        # Append to the existing leads list
                        st.session_state["leads"].extend(new_unique_leads)

                        # Append to the existing leads_df DataFrame
                        new_leads_df = pd.DataFrame(
                            new_unique_leads,
                            columns=["Entity", "Type", "University", "City", "Country"],
                        )
                        st.session_state["leads_df"] = pd.concat(
                            [st.session_state["leads_df"], new_leads_df],
                            ignore_index=True,
                        )

                        # Update the 'step' if necessary
                        st.session_state["step"] = max(st.session_state["step"], 2)

                        st.success(
                            f"Added {len(new_unique_leads)} new leads successfully! Total leads: {len(st.session_state['leads'])}"
                        )
                        logger.info(f"Added {len(new_unique_leads)} leads manually.")
                    else:
                        st.info("No new unique leads to add.")
                else:
                    logger.error("No valid leads entered.")
                    st.error(
                        "No valid leads entered. Please ensure each line is in the format 'Entity, Type, University, City, Country'."
                    )
                    st.stop()

        elif option == "Search Leads via Conference":
            st.write("Enter the conference details below:")
            conference_input = st.text_input(
                "Enter the conference name or URL:", key="conference_input"
            )

            search_btn = st.button("Search Leads", key="search_conference_leads_btn")
            if search_btn:
                if not context.strip():
                    logger.error("Lead generation attempted with empty context.")
                    st.error("Please provide a context for lead generation.")
                    st.stop()
                if not conference_input.strip():
                    st.warning("Please enter a conference name or URL.")
                else:
                    with st.spinner(
                        "Searching for leads associated with the conference..."
                    ):
                        leads_list = search_leads_via_conference(
                            conference_input, context
                        )
                        if leads_list:
                            # Convert existing leads to a set for faster lookup
                            existing_leads_set = set(st.session_state["leads"])

                            # Filter out any leads that already exist
                            new_unique_leads = [
                                lead
                                for lead in leads_list
                                if lead not in existing_leads_set
                            ]

                            if new_unique_leads:
                                # Append to the existing leads list
                                st.session_state["leads"].extend(new_unique_leads)

                                # Create a DataFrame from the new leads
                                new_leads_df = pd.DataFrame(
                                    new_unique_leads,
                                    columns=[
                                        "Entity",
                                        "Type",
                                        "University",
                                        "City",
                                        "Country",
                                    ],
                                )

                                # Append to the existing leads_df DataFrame
                                st.session_state["leads_df"] = pd.concat(
                                    [st.session_state["leads_df"], new_leads_df],
                                    ignore_index=True,
                                )

                                st.success(
                                    f"Leads extracted successfully! Added {len(new_unique_leads)} new leads. Total leads: {len(st.session_state['leads'])}"
                                )
                                logger.info(
                                    f"Extracted {len(new_unique_leads)} leads from conference input."
                                )
                            else:
                                st.info(
                                    "No new unique leads found for the provided conference."
                                )
                        else:
                            st.warning("No leads found for the provided conference.")

        # Display the Leads DataFrame if it exists
        if not st.session_state["leads_df"].empty:
            st.write("### Leads")
            st.write("You can edit the leads below:")

            try:
                # Create a temporary DataFrame to work with
                temp_leads_df = st.session_state["leads_df"].copy()

                # Use the data editor on the temporary DataFrame
                edited_leads_df = st.data_editor(
                    temp_leads_df, num_rows="dynamic", key="leads_editor"
                )

                # Add a 'Save Changes' button to commit the edits
                if st.button("Save Changes", key="save_leads_changes"):
                    # Update the session state with the edited leads
                    st.session_state["leads_df"] = edited_leads_df

                    # Synchronize the 'leads' list with the updated DataFrame
                    st.session_state["leads"] = list(
                        zip(
                            edited_leads_df["Entity"],
                            edited_leads_df["Type"],
                            edited_leads_df["University"],
                            edited_leads_df["City"],
                            edited_leads_df["Country"],
                        )
                    )

                    st.success("Leads updated successfully!")
                    logger.info("Leads edited via data editor.")

                # Display the edited DataFrame
                # display_leads_df = st.session_state['leads_df']

                display_leads_df = st.session_state["leads_df"].copy()
                # display_leads_df.rename(columns={
                #     "Entity": "Name",
                #     "Type": "Category",
                #     "University": "University",
                #     "City": "City",
                #     "Country": "Country"
                # }, inplace=True)

                display_and_download(
                    df=display_leads_df, button_label="Leads", filename="leads"
                )

            except Exception as e:
                logger.error(f"Error editing leads: {e}")
                st.error(f"Error editing leads: {e}")
                st.stop()

        st.session_state["input_leads"] = True
        st.success("Input Leads section loaded successfully!")

    def scrape_lead_information():
        """
        Section for Analyze Lead Information: Perform per-field searches and scrape information,
        then extract and analyze persons associated with each lead.
        """
        st.subheader("Search and Analyse Lead Information")
        progress_bar = st.empty()
        status_text = st.empty()
        lead_placeholder = st.empty()
        person_placeholder = st.empty()

        st.header("Analyze Lead Information")
        default_columns = [
            "Entity",
            "CEO/PI",
            "Researchers",
            "Grants",
            "Phone Number",
            "Email",
            "Country",
            "University",
            "Summary",
            "Contacts",
        ]
        columns_to_retrieve = st_tags(
            label="",
            text="Add or remove information fields:",
            value=default_columns,
            suggestions=default_columns,
            maxtags=20,
            key="columns_to_retrieve",
        )

        # Add input for additional search keywords for persons
        default_person_keywords = [
            "email",
            "phone number",
            "profile",
            "CV",
            "LinkedIn",
            "research",
            "publications",
            "grant",
        ]
        person_search_keywords = st_tags(
            label="",
            text="Add or remove keywords to guide the person search:",
            value=default_person_keywords,
            suggestions=default_person_keywords,
            maxtags=20,
            key="person_search_keywords",
        )

        search_btn = st.button("Search and Analyse Leads", key="search_leads_btn")
        if search_btn:
            # Validate input context
            if st.session_state.get("context", "") == "":
                logger.error("Lead scraping attempted with empty context.")
                st.warning("Please provide a context for lead scraping.")
                return

            # Validate that leads are available
            if not st.session_state.get("leads"):
                logger.error(
                    "Lead scraping attempted without any generated or added leads."
                )
                st.error("No leads available. Please add or generate leads first.")
                return

            # Validate that at least one information field is selected
            if not columns_to_retrieve:
                logger.error(
                    "Lead scraping attempted without specifying information fields."
                )
                st.error("Please select at least one information field to retrieve.")
                return

            # Initialize scraping state if not already in progress
            if not st.session_state.get("is_scraping", False):
                st.session_state["is_scraping"] = True
                st.session_state["processed_leads"] = []
                st.session_state["leads_info_df"] = pd.DataFrame()
                st.session_state["person_leads_df"] = pd.DataFrame()

            total_leads = len(st.session_state["leads"])
            progress_increment = 100 / total_leads
            progress = 0
            progress_bar.progress(0)
            status_text.text("Starting lead information scraping...")
            print("dic:", st.session_state["leads"])
            print(f"st.session_state['leads']: {st.session_state['leads']}")
            for idx, lead in enumerate(st.session_state["leads"]):

                lead_name = lead.get("Entity", "Unknown")
                lead_category = lead.get("Type", "Unknown")
                lead_university = lead.get("University", "Unknown")
                lead_city = lead.get("City", "Unknown")
                lead_country = lead.get("Country", "Unknown")

                status_text.text(
                    f"Processing Lead {idx + 1} of {total_leads}: {lead_name}"
                )
                # Analyse per-field information
                field_data_list = []
                for field in columns_to_retrieve:
                    if field in ["Entity", "Summary", "Type"]:
                        continue  # Skip fields that don't require scraping
                    field_data = scrape_information_field(
                        lead_name, field, num_search_results=1
                    )
                    field_data_list.append(field_data)
                    time.sleep(random.uniform(1, 3))  # Respect rate limits

                # Extract lead information using GPT with chunking
                lead_info = extract_lead_info_with_llm_per_field(
                    lead_name, lead_category, field_data_list
                )
                if lead_info:
                    # Append lead information to the DataFrame
                    st.session_state["leads_info_df"] = pd.concat(
                        [st.session_state["leads_info_df"], pd.DataFrame([lead_info])],
                        ignore_index=True,
                    )
                    st.session_state["processed_leads"].append(lead_info)

                    # Update the UI with the newly processed lead
                    with lead_placeholder.container():
                        st.markdown(f"### {lead_info.get('Entity', 'Unknown')}")
                        st.json(lead_info, expanded=False)
                        st.markdown("---")

                # Extract and Analyze Persons Associated with the Lead
                with person_placeholder.container():
                    st.subheader(f"Extracting Persons for {lead_name}")
                    person_status_text = st.empty()
                    person_progress_bar = st.progress(0)

                    persons = extract_persons(st.session_state["leads_info_df"])

                    if not persons:
                        logger.warning(
                            f"No persons found associated with the lead '{lead_name}'."
                        )
                        st.warning(
                            f"No persons found associated with the lead '{lead_name}'."
                        )
                    else:
                        logger.info(
                            f"Found {len(persons)} unique persons associated with the lead '{lead_name}'."
                        )
                        person_progress_increment = 100 / len(persons)
                        person_progress = 0
                        person_leads = []

                        print("dic:", st.session_state["leads_info_df"])
                        print(
                            f"st.session_state['leads_info_df']: {st.session_state['leads_info_df']}"
                        )

                        for p_idx, (person_name, associated_lead) in enumerate(persons):
                            person_status_text.text(
                                f"Processing Person {p_idx + 1}/{len(persons)}: {person_name}"
                            )
                            # Include the additional keywords in the search
                            person_urls = perform_google_search(
                                person_name, additional_keywords=person_search_keywords
                            )
                            if not person_urls:
                                logger.warning(f"No URLs found for '{person_name}'.")
                                continue
                            scraped_text = ""
                            sources = []
                            # Analyse the URLs
                            for url in person_urls:
                                logger.info(
                                    f"Looking at URL for '{person_name}': {url}"
                                )
                                with st.spinner(
                                    f"Looking at URL for '{person_name}': {url}"
                                ):
                                    scraped_content = scrape_landing_page(url)
                                    if scraped_content:
                                        scraped_text += scraped_content + " "
                                        sources.append(url)
                                    time.sleep(
                                        random.uniform(1, 2)
                                    )  # Respect rate limits
                            if not scraped_text.strip():
                                logger.warning(
                                    f"No text scraped from URLs for '{person_name}' Skipping."
                                )
                                continue
                            cleaned_text = clean_text(scraped_text)
                            source_urls = sources
                            person_info = extract_person_info_with_llm(
                                cleaned_text, person_name, source_urls
                            )
                            if person_info:
                                person_info["Name"] = (
                                    person_name  # Ensure correct name assignment
                                )
                                person_leads.append(person_info)

                            # Update person progress bar
                            person_progress += person_progress_increment
                            person_progress_bar.progress(
                                min(person_progress, 100) / 100
                            )

                        # Append person leads to the session state DataFrame
                        if person_leads:
                            try:
                                person_leads_df = pd.DataFrame(person_leads)
                                st.session_state["person_leads_df"] = pd.concat(
                                    [
                                        st.session_state["person_leads_df"],
                                        person_leads_df,
                                    ],
                                    ignore_index=True,
                                )
                                st.success(
                                    f"Person information scraped successfully for lead '{lead_name}'!"
                                )
                                logger.info(
                                    f"Person information scraped successfully for lead '{lead_name}'."
                                )

                                # Display the person leads information
                                st.write(f"### Persons Associated with {lead_name}")
                                display_lead_information(
                                    df=person_leads_df,
                                    button_label="Download Person Leads Information",
                                    filename=f"person_leads_info_{lead_name.replace(' ', '_')}",
                                )

                            except Exception as e:
                                logger.error(f"Error creating person_leads_df: {e}")
                                st.error(
                                    f"Error processing scraped person information for lead '{lead_name}': {e}"
                                )
                        else:
                            logger.error(
                                f"Person information scraping failed for lead '{lead_name}'."
                            )
                            st.error(
                                f"Failed to scrape person information for lead '{lead_name}'. Please check the logs for more details."
                            )

                # Update overall progress bar
                progress += progress_increment
                progress_bar.progress(min(progress, 100) / 100)

            status_text.text("Lead information scraping completed!")
            st.success("All leads have been processed successfully.")
            st.session_state["is_scraping"] = False

        # Display the Leads Information DataFrame if it exists
        if not st.session_state.get("leads_info_df", pd.DataFrame()).empty:
            st.write("### Leads Information")
            display_lead_information(
                df=st.session_state["leads_info_df"],
                button_label="Download Leads Information",
                filename="leads_info",
            )

        st.session_state["scrape_lead_information"] = True
        st.success("Lead information section loaded successfully!")

    def analytics_section():
        """
        Section for Analytics: Provide various analytics and visualizations.
        """
        st.subheader("Analytics")

        analytics_expander = st.expander("View Analytics")
        with analytics_expander:
            # ------------------------------
            # Company-Level Analytics
            # ------------------------------
            if (
                "company_bant_df" in st.session_state
                and not st.session_state["company_bant_df"].empty
            ):
                # Number of Grants per Company
                if "Grants" in st.session_state["company_bant_df"].columns:
                    st.markdown("### Number of Grants per Company")

                    def count_grants(grants_entry):
                        if isinstance(grants_entry, list):
                            return len(grants_entry)
                        elif isinstance(grants_entry, str) and grants_entry.strip():
                            return 1
                        else:
                            return 0

                    st.session_state["company_bant_df"]["Grant Count"] = (
                        st.session_state["company_bant_df"]["Grants"].apply(
                            count_grants
                        )
                    )

                    # Plotting Number of Grants per Company
                    fig, ax = plt.subplots(figsize=(10, 6))
                    sns.barplot(
                        data=st.session_state["company_bant_df"],
                        x="Entity",
                        y="Grant Count",
                        palette="viridis",
                    )
                    ax.set_xlabel("Entity")
                    ax.set_ylabel("Number of Grants")
                    ax.set_title("Number of Grants per Company")
                    plt.xticks(rotation=90)
                    plt.tight_layout()
                    st.pyplot(fig)
                else:
                    logger.warning("No 'Grants' column available in company_bant_df.")
                    st.warning("No 'Grants' data available for grants analysis.")

                # Distribution of Overall BANT Scores (Companies)
                st.markdown("### Distribution of Overall BANT Scores (Companies)")
                if "Overall BANT Score" in st.session_state["company_bant_df"].columns:
                    plt.figure(figsize=(10, 6))
                    sns.histplot(
                        st.session_state["company_bant_df"]["Overall BANT Score"],
                        bins=10,
                        kde=True,
                        color="salmon",
                    )
                    plt.title("Distribution of Overall BANT Scores (Companies)")
                    plt.xlabel("Overall BANT Score")
                    plt.ylabel("Number of Companies")
                    plt.tight_layout()
                    st.pyplot(plt)
                else:
                    logger.warning(
                        "'Overall BANT Score' column missing in company_bant_df."
                    )
                    st.warning(
                        "No 'Overall BANT Score' data available for distribution analysis."
                    )

                # Correlation Heatmap of BANT Scores (Companies)
                st.markdown("### Correlation Between BANT Components (Companies)")
                bant_components = [
                    "Average Budget Score",
                    "Average Authority Score",
                    "Average Need Score",
                    "Average Timeline Score",
                    "Overall BANT Score",
                ]
                if all(
                    component in st.session_state["company_bant_df"].columns
                    for component in bant_components
                ):
                    correlation = st.session_state["company_bant_df"][
                        bant_components
                    ].corr()
                    fig, ax = plt.subplots(figsize=(8, 6))
                    sns.heatmap(correlation, annot=True, cmap="coolwarm", ax=ax)
                    plt.title("Correlation Heatmap of BANT Scores (Companies)")
                    plt.tight_layout()
                    st.pyplot(fig)
                else:
                    missing_components = [
                        comp
                        for comp in bant_components
                        if comp not in st.session_state["company_bant_df"].columns
                    ]
                    logger.warning(
                        f"Missing columns for correlation heatmap: {missing_components}"
                    )
                    st.warning(
                        f"Missing columns for correlation heatmap: {', '.join(missing_components)}"
                    )

                # Recommendations Breakdown (Companies)
                st.markdown("### Recommendations Breakdown (Companies)")
                if "Recommendations" in st.session_state["company_bant_df"].columns:
                    recommendations_counts = st.session_state["company_bant_df"][
                        "Recommendations"
                    ].value_counts()
                    fig, ax = plt.subplots(figsize=(8, 6))
                    sns.barplot(
                        x=recommendations_counts.index,
                        y=recommendations_counts.values,
                        palette="magma",
                    )
                    plt.title("Recommendations Distribution (Companies)")
                    plt.xlabel("Recommendation")
                    plt.ylabel("Number of Companies")
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    st.pyplot(fig)
                else:
                    logger.warning(
                        "'Recommendations' column missing in company_bant_df."
                    )
                    st.warning("No 'Recommendations' data available for companies.")

            else:
                st.warning(
                    "No company-level BANT analysis data available. Please perform BANT ranking first."
                )

            # ------------------------------
            # Individual-Level Analytics
            # ------------------------------
            if (
                "person_leads_df" in st.session_state
                and not st.session_state["person_leads_df"].empty
            ):
                # Distribution of Overall BANT Scores (Individuals)
                st.markdown("### Distribution of Overall BANT Scores (Individuals)")
                if "Overall BANT Score" in st.session_state["person_leads_df"].columns:
                    plt.figure(figsize=(10, 6))
                    sns.histplot(
                        st.session_state["person_leads_df"]["Overall BANT Score"],
                        bins=10,
                        kde=True,
                        color="skyblue",
                    )
                    plt.title("Distribution of Overall BANT Scores (Individuals)")
                    plt.xlabel("Overall BANT Score")
                    plt.ylabel("Number of Individuals")
                    plt.tight_layout()
                    st.pyplot(plt)
                else:
                    logger.warning(
                        "'Overall BANT Score' column missing in person_leads_df."
                    )
                    st.warning(
                        "No 'Overall BANT Score' data available for individual distribution analysis."
                    )

                # Recommendations Breakdown (Individuals)
                st.markdown("### Recommendations Breakdown (Individuals)")
                if "Recommendations" in st.session_state["person_leads_df"].columns:
                    individual_recommendations = st.session_state["person_leads_df"][
                        "Recommendations"
                    ].value_counts()
                    fig, ax = plt.subplots(figsize=(8, 6))
                    sns.barplot(
                        x=individual_recommendations.index,
                        y=individual_recommendations.values,
                        palette="coolwarm",
                    )
                    plt.title("Recommendations Distribution (Individuals)")
                    plt.xlabel("Recommendation")
                    plt.ylabel("Number of Individuals")
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    st.pyplot(fig)
                else:
                    logger.warning(
                        "'Recommendations' column missing in person_leads_df."
                    )
                    st.warning("No 'Recommendations' data available for individuals.")

                # Correlation Heatmap of BANT Scores (Individuals)
                st.markdown("### Correlation Between BANT Components (Individuals)")
                bant_components_individual = [
                    "Budget Score",
                    "Authority Score",
                    "Need Score",
                    "Timeline Score",
                    "Overall BANT Score",
                ]
                if all(
                    component in st.session_state["person_leads_df"].columns
                    for component in bant_components_individual
                ):
                    correlation_individual = st.session_state["person_leads_df"][
                        bant_components_individual
                    ].corr()
                    fig, ax = plt.subplots(figsize=(8, 6))
                    sns.heatmap(
                        correlation_individual, annot=True, cmap="coolwarm", ax=ax
                    )
                    plt.title("Correlation Heatmap of BANT Scores (Individuals)")
                    plt.tight_layout()
                    st.pyplot(fig)
                else:
                    missing_components_individual = [
                        comp
                        for comp in bant_components_individual
                        if comp not in st.session_state["person_leads_df"].columns
                    ]
                    logger.warning(
                        f"Missing columns for individual correlation heatmap: {missing_components_individual}"
                    )
                    st.warning(
                        f"Missing columns for individual correlation heatmap: {', '.join(missing_components_individual)}"
                    )

                # Number of Phone Numbers and Emails per Individual
                st.markdown("### Number of Phone Numbers and Emails per Individual")
                if "Contacts" in st.session_state["person_leads_df"].columns:

                    def count_emails_and_phones_individual(contacts_entry):
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
                            email = contact.get("Email")
                            if email and email.strip().lower() not in [
                                "not provided",
                                "not available",
                                "",
                            ]:
                                num_emails += 1
                            phone = contact.get("Phone Number")
                            if phone and phone.strip().lower() not in [
                                "not provided",
                                "not available",
                                "",
                            ]:
                                num_phones += 1
                        return pd.Series(
                            {"Num Emails": num_emails, "Num Phones": num_phones}
                        )

                    contacts_counts_individual = st.session_state["person_leads_df"][
                        "Contacts"
                    ].apply(count_emails_and_phones_individual)
                    st.session_state["person_leads_df"] = st.session_state[
                        "person_leads_df"
                    ].join(contacts_counts_individual, rsuffix="_new")

                    # Display the counts in a table
                    try:
                        st.write("#### Emails and Phone Numbers per Individual")
                        st.write(
                            st.session_state["person_leads_df"][
                                ["Name", "Num Emails", "Num Phones"]
                            ]
                        )
                    except KeyError:
                        logger.error(
                            "'Name', 'Num Emails', or 'Num Phones' columns missing in person_leads_df."
                        )
                        st.error(
                            "Required columns for individual contact analysis are missing."
                        )
                        st.stop()

                    # Plot number of emails and phone numbers per individual
                    fig, ax = plt.subplots(figsize=(10, 6))
                    x = np.arange(len(st.session_state["person_leads_df"]))
                    width = 0.35
                    try:
                        ax.bar(
                            x - width / 2,
                            st.session_state["person_leads_df"]["Num Emails"],
                            width,
                            label="Emails",
                        )
                        ax.bar(
                            x + width / 2,
                            st.session_state["person_leads_df"]["Num Phones"],
                            width,
                            label="Phone Numbers",
                        )
                        ax.set_xlabel("Individual Name")
                        ax.set_ylabel("Count")
                        ax.set_title(
                            "Number of Emails and Phone Numbers per Individual"
                        )
                        ax.set_xticks(x)
                        try:
                            ax.set_xticklabels(
                                st.session_state["person_leads_df"]["Name"], rotation=90
                            )
                        except KeyError:
                            logger.error(
                                "'Name' column missing in person_leads_df when setting x-tick labels."
                            )
                            ax.set_xticklabels(["Unknown"] * len(x), rotation=90)
                        ax.legend()
                        plt.tight_layout()
                        st.pyplot(fig)
                    except KeyError:
                        logger.error(
                            "'Name', 'Num Emails', or 'Num Phones' columns missing in person_leads_df."
                        )
                        st.error(
                            "Required columns for individual contact plotting are missing."
                        )
                        st.stop()
                else:
                    logger.warning("No 'Contacts' data available in person_leads_df.")
                    st.warning(
                        "No 'Contacts' data available for individual contact analysis."
                    )

                # Number of Individual Names per Company
                st.markdown("### Number of Individual Names per Company")
                if (
                    "Entity" in st.session_state["person_leads_df"].columns
                    and "Entity" in st.session_state["person_leads_df"].columns
                ):

                    def count_individual_names(row):
                        names = set()
                        # From CEO/PI
                        ceo_pi = row.get("CEO/PI")
                        if (
                            isinstance(ceo_pi, str)
                            and ceo_pi.strip()
                            and ceo_pi.strip().lower() != "not available"
                        ):
                            names.add(ceo_pi.strip())
                        # From Researchers
                        researchers = row.get("Researchers")
                        if isinstance(researchers, list):
                            names.update(
                                [
                                    r
                                    for r in researchers
                                    if isinstance(r, str)
                                    and r.strip().lower() != "not available"
                                ]
                            )
                        elif (
                            isinstance(researchers, str)
                            and researchers.strip()
                            and researchers.strip().lower() != "not available"
                        ):
                            names.add(researchers.strip())
                        # From Contacts
                        contacts = row.get("Contacts")
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
                            name = contact.get("Name")
                            if (
                                isinstance(name, str)
                                and name.strip().lower() != "not available"
                            ):
                                names.add(name.strip())
                        return len(names)

                    try:
                        st.session_state["company_bant_df"][
                            "Num Individuals"
                        ] = st.session_state["company_bant_df"].apply(
                            lambda row: st.session_state["person_leads_df"][
                                st.session_state["person_leads_df"]["Name"]
                                == row["Name"]
                            ].shape[0],
                            axis=1,
                        )

                        # Display the counts in a table
                        try:
                            st.write("#### Number of Individual Names per Company")
                            st.write(
                                st.session_state["company_bant_df"][
                                    ["Name", "Num Individuals"]
                                ]
                            )
                        except KeyError:
                            logger.error(
                                "'Name' or 'Num Individuals' columns missing in company_bant_df."
                            )
                            st.error(
                                "Required columns for individual names count are missing."
                            )
                            st.stop()

                        # Plot number of individuals per company
                        try:
                            fig, ax = plt.subplots(figsize=(10, 6))
                            sns.barplot(
                                data=st.session_state["company_bant_df"],
                                x="Entity",
                                y="Num Individuals",
                                palette="deep",
                            )
                            ax.set_xlabel("Entity")
                            ax.set_ylabel("Number of Individuals")
                            ax.set_title("Number of Individual Names per Company")
                            plt.xticks(rotation=90)
                            plt.tight_layout()
                            st.pyplot(fig)
                        except KeyError:
                            logger.error(
                                "'Name' or 'Num Individuals' columns missing in company_bant_df."
                            )
                            st.error(
                                "Required columns for individual names plotting are missing."
                            )
                            st.stop()
                    except Exception as e:
                        logger.error(
                            f"Error counting individual names per company: {e}"
                        )
                        st.error(f"Error counting individual names per company: {e}")
                        st.stop()
                else:
                    logger.warning(
                        "'Name' or 'Name' columns missing in person_leads_df."
                    )
                    st.warning(
                        "Required columns for individual names per company analysis are missing."
                    )

                # Summaries
                if "Summary" in st.session_state["company_bant_df"].columns:
                    st.markdown("### Summaries")
                    for idx, row in st.session_state["company_bant_df"].iterrows():
                        company_name = row["Entity"]
                        summary = row["Summary"]
                        if pd.notnull(summary) and summary.strip():
                            st.write(f"**{company_name}:** {summary}")
                        else:
                            st.write(f"**{company_name}:** No summary available.")
                        st.markdown("---")

                    # Word Cloud for Summaries
                    summaries = (
                        st.session_state["company_bant_df"]["Summary"].dropna().tolist()
                    )
                    combined_text = " ".join(summaries)
                    if combined_text:
                        st.markdown("### Word Cloud for Summaries")
                        wordcloud = WordCloud(
                            width=800, height=400, background_color="white"
                        ).generate(combined_text)
                        fig, ax = plt.subplots(figsize=(10, 5))
                        ax.imshow(wordcloud, interpolation="bilinear")
                        ax.axis("off")
                        plt.tight_layout()
                        st.pyplot(fig)
                    else:
                        logger.warning("No summaries available for word cloud.")
                        st.warning("No summaries available to generate a word cloud.")
                else:
                    logger.warning("'Summary' column missing in company_bant_df.")
                    st.warning(
                        "No 'Summary' data available for summaries and word cloud."
                    )

            # ------------------------------
            # Additional Individual-Level Analytics
            # ------------------------------
            if (
                "person_leads_df" in st.session_state
                and not st.session_state["person_leads_df"].empty
            ):
                # Number of Grants per Individual
                if "Grants" in st.session_state["person_leads_df"].columns:
                    st.markdown("### Number of Grants per Individual")

                    def count_grants_individual(grants_entry):
                        if isinstance(grants_entry, list):
                            return len(grants_entry)
                        elif isinstance(grants_entry, str) and grants_entry.strip():
                            return 1
                        else:
                            return 0

                    st.session_state["person_leads_df"]["Grant Count"] = (
                        st.session_state["person_leads_df"]["Grants"].apply(
                            count_grants_individual
                        )
                    )

                    # Plotting Number of Grants per Individual
                    fig, ax = plt.subplots(figsize=(10, 6))
                    sns.barplot(
                        data=st.session_state["person_leads_df"],
                        x="Name",
                        y="Grant Count",
                        palette="inferno",
                    )
                    ax.set_xlabel("Individual Name")
                    ax.set_ylabel("Number of Grants")
                    ax.set_title("Number of Grants per Individual")
                    plt.xticks(rotation=90)
                    plt.tight_layout()
                    st.pyplot(fig)
                else:
                    logger.warning("'Grants' column missing in person_leads_df.")
                    st.warning(
                        "No 'Grants' data available for individual grants analysis."
                    )

                # Distribution of Overall BANT Scores (Individuals)
                st.markdown("### Distribution of Overall BANT Scores (Individuals)")
                if "Overall BANT Score" in st.session_state["person_leads_df"].columns:
                    plt.figure(figsize=(10, 6))
                    sns.histplot(
                        st.session_state["person_leads_df"]["Overall BANT Score"],
                        bins=10,
                        kde=True,
                        color="lightgreen",
                    )
                    plt.title("Distribution of Overall BANT Scores (Individuals)")
                    plt.xlabel("Overall BANT Score")
                    plt.ylabel("Number of Individuals")
                    plt.tight_layout()
                    st.pyplot(plt)
                else:
                    logger.warning(
                        "'Overall BANT Score' column missing in person_leads_df."
                    )
                    st.warning(
                        "No 'Overall BANT Score' data available for individual distribution analysis."
                    )

                # Recommendations Breakdown (Individuals)
                st.markdown("### Recommendations Breakdown (Individuals)")
                if "Recommendations" in st.session_state["person_leads_df"].columns:
                    individual_recommendations = st.session_state["person_leads_df"][
                        "Recommendations"
                    ].value_counts()
                    fig, ax = plt.subplots(figsize=(8, 6))
                    sns.barplot(
                        x=individual_recommendations.index,
                        y=individual_recommendations.values,
                        palette="viridis",
                    )
                    plt.title("Recommendations Distribution (Individuals)")
                    plt.xlabel("Recommendation")
                    plt.ylabel("Number of Individuals")
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    st.pyplot(fig)
                else:
                    logger.warning(
                        "'Recommendations' column missing in person_leads_df."
                    )
                    st.warning("No 'Recommendations' data available for individuals.")

                # Correlation Heatmap of BANT Scores (Individuals)
                st.markdown("### Correlation Between BANT Components (Individuals)")
                bant_components_individual = [
                    "Budget Score",
                    "Authority Score",
                    "Need Score",
                    "Timeline Score",
                    "Overall BANT Score",
                ]
                if all(
                    component in st.session_state["person_leads_df"].columns
                    for component in bant_components_individual
                ):
                    correlation_individual = st.session_state["person_leads_df"][
                        bant_components_individual
                    ].corr()
                    fig, ax = plt.subplots(figsize=(8, 6))
                    sns.heatmap(
                        correlation_individual, annot=True, cmap="coolwarm", ax=ax
                    )
                    plt.title("Correlation Heatmap of BANT Scores (Individuals)")
                    plt.tight_layout()
                    st.pyplot(fig)
                else:
                    missing_components_individual = [
                        comp
                        for comp in bant_components_individual
                        if comp not in st.session_state["person_leads_df"].columns
                    ]
                    logger.warning(
                        f"Missing columns for individual correlation heatmap: {missing_components_individual}"
                    )
                    st.warning(
                        f"Missing columns for individual correlation heatmap: {', '.join(missing_components_individual)}"
                    )

                # Number of Phone Numbers and Emails per Individual
                st.markdown("### Number of Phone Numbers and Emails per Individual")
                if "Contacts" in st.session_state["person_leads_df"].columns:

                    def count_emails_and_phones_individual(contacts_entry):
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
                            email = contact.get("Email")
                            if email and email.strip().lower() not in [
                                "not provided",
                                "not available",
                                "",
                            ]:
                                num_emails += 1
                            phone = contact.get("Phone Number")
                            if phone and phone.strip().lower() not in [
                                "not provided",
                                "not available",
                                "",
                            ]:
                                num_phones += 1
                        return pd.Series(
                            {"Num Emails": num_emails, "Num Phones": num_phones}
                        )

                    try:
                        contacts_counts_individual = st.session_state[
                            "person_leads_df"
                        ]["Contacts"].apply(count_emails_and_phones_individual)
                        st.session_state["person_leads_df"] = st.session_state[
                            "person_leads_df"
                        ].join(contacts_counts_individual, rsuffix="_new")

                        # Display the counts in a table
                        try:
                            st.write("#### Emails and Phone Numbers per Individual")
                            st.write(
                                st.session_state["person_leads_df"][
                                    ["Name", "Num Emails", "Num Phones"]
                                ]
                            )
                        except KeyError:
                            logger.error(
                                "'Name', 'Num Emails', or 'Num Phones' columns missing in person_leads_df."
                            )
                            st.error(
                                "Required columns for individual contact analysis are missing."
                            )
                            st.stop()

                        # Plot number of emails and phone numbers per individual
                        fig, ax = plt.subplots(figsize=(10, 6))
                        sns.barplot(
                            data=st.session_state["person_leads_df"],
                            x="Entity",
                            y="Num Emails",
                            color="steelblue",
                            label="Emails",
                        )
                        sns.barplot(
                            data=st.session_state["person_leads_df"],
                            x="Name",
                            y="Num Phones",
                            color="orange",
                            label="Phone Numbers",
                        )
                        ax.set_xlabel("Individual Name")
                        ax.set_ylabel("Count")
                        ax.set_title(
                            "Number of Emails and Phone Numbers per Individual"
                        )
                        plt.xticks(rotation=90)
                        plt.legend()
                        plt.tight_layout()
                        st.pyplot(fig)
                    except KeyError:
                        logger.error(
                            "'Name', 'Contacts', 'Num Emails', or 'Num Phones' columns missing in person_leads_df."
                        )
                        st.error(
                            "Required columns for individual contact analysis are missing."
                        )
                        st.stop()
                    except Exception as e:
                        logger.error(f"Error during individual contact analysis: {e}")
                        st.error(f"Error during individual contact analysis: {e}")
                        st.stop()
                else:
                    logger.warning("'Contacts' column missing in person_leads_df.")
                    st.warning(
                        "No 'Contacts' data available for individual contact analysis."
                    )

                # Number of Individual Names per Company
                st.markdown("### Number of Individual Names per Company")
                if (
                    "Name" in st.session_state["person_leads_df"].columns
                    and "Name" in st.session_state["person_leads_df"].columns
                ):
                    try:
                        individuals_per_company = (
                            st.session_state["person_leads_df"]
                            .groupby("Name")["Name"]
                            .nunique()
                            .reset_index()
                        )
                        individuals_per_company = individuals_per_company.rename(
                            columns={"Name": "Num Individuals"}
                        )

                        st.write("#### Number of Individual Names per Company")
                        st.write(individuals_per_company)

                        # Plot number of individuals per company
                        fig, ax = plt.subplots(figsize=(10, 6))
                        sns.barplot(
                            data=individuals_per_company,
                            x="Entity",
                            y="Num Individuals",
                            palette="coolwarm",
                        )
                        ax.set_xlabel("Name")
                        ax.set_ylabel("Number of Individuals")
                        ax.set_title("Number of Individual Names per Company")
                        plt.xticks(rotation=90)
                        plt.tight_layout()
                        st.pyplot(fig)
                    except Exception as e:
                        logger.error(
                            f"Error counting individual names per company: {e}"
                        )
                        st.error(f"Error counting individual names per company: {e}")
                        st.stop()
                else:
                    logger.warning(
                        "'Name' or 'Name' columns missing in person_leads_df."
                    )
                    st.warning(
                        "Required columns for individual names per company analysis are missing."
                    )

                # Summaries and Word Cloud for Companies
                if "Summary" in st.session_state["company_bant_df"].columns:
                    st.markdown("### Summaries")
                    for idx, row in st.session_state["company_bant_df"].iterrows():
                        company_name = row["Entity"]
                        summary = row["Summary"]
                        if pd.notnull(summary) and summary.strip():
                            st.write(f"**{company_name}:** {summary}")
                        else:
                            st.write(f"**{company_name}:** No summary available.")
                        st.markdown("---")

                    # Word Cloud for Summaries
                    summaries = (
                        st.session_state["company_bant_df"]["Summary"].dropna().tolist()
                    )
                    combined_text = " ".join(summaries)
                    if combined_text:
                        st.markdown("### Word Cloud for Summaries")
                        wordcloud = WordCloud(
                            width=800, height=400, background_color="white"
                        ).generate(combined_text)
                        fig, ax = plt.subplots(figsize=(10, 5))
                        ax.imshow(wordcloud, interpolation="bilinear")
                        ax.axis("off")
                        plt.tight_layout()
                        st.pyplot(fig)
                    else:
                        logger.warning("No summaries available for word cloud.")
                        st.warning("No summaries available to generate a word cloud.")
                else:
                    logger.warning("'Summary' column missing in company_bant_df.")
                    st.warning(
                        "No 'Summary' data available for summaries and word cloud."
                    )

            # ------------------------------
            # Additional Individual-Level Analytics
            # ------------------------------
            if (
                "person_leads_df" in st.session_state
                and not st.session_state["person_leads_df"].empty
            ):
                # Number of Grants per Individual
                if "Grants" in st.session_state["person_leads_df"].columns:
                    st.markdown("### Number of Grants per Individual")

                    def count_grants_individual(grants_entry):
                        if isinstance(grants_entry, list):
                            return len(grants_entry)
                        elif isinstance(grants_entry, str) and grants_entry.strip():
                            return 1
                        else:
                            return 0

                    st.session_state["person_leads_df"]["Grant Count"] = (
                        st.session_state["person_leads_df"]["Grants"].apply(
                            count_grants_individual
                        )
                    )

                    # Plotting Number of Grants per Individual
                    fig, ax = plt.subplots(figsize=(10, 6))
                    sns.barplot(
                        data=st.session_state["person_leads_df"],
                        x="Name",
                        y="Grant Count",
                        palette="inferno",
                    )
                    ax.set_xlabel("Individual Name")
                    ax.set_ylabel("Number of Grants")
                    ax.set_title("Number of Grants per Individual")
                    plt.xticks(rotation=90)
                    plt.tight_layout()
                    st.pyplot(fig)
                else:
                    logger.warning("'Grants' column missing in person_leads_df.")
                    st.warning(
                        "No 'Grants' data available for individual grants analysis."
                    )

                # Distribution of Overall BANT Scores (Individuals)
                st.markdown("### Distribution of Overall BANT Scores (Individuals)")
                if "Overall BANT Score" in st.session_state["person_leads_df"].columns:
                    plt.figure(figsize=(10, 6))
                    sns.histplot(
                        st.session_state["person_leads_df"]["Overall BANT Score"],
                        bins=10,
                        kde=True,
                        color="lightgreen",
                    )
                    plt.title("Distribution of Overall BANT Scores (Individuals)")
                    plt.xlabel("Overall BANT Score")
                    plt.ylabel("Number of Individuals")
                    plt.tight_layout()
                    st.pyplot(plt)
                else:
                    logger.warning(
                        "'Overall BANT Score' column missing in person_leads_df."
                    )
                    st.warning(
                        "No 'Overall BANT Score' data available for individual distribution analysis."
                    )

                # Recommendations Breakdown (Individuals)
                st.markdown("### Recommendations Breakdown (Individuals)")
                if "Recommendations" in st.session_state["person_leads_df"].columns:
                    individual_recommendations = st.session_state["person_leads_df"][
                        "Recommendations"
                    ].value_counts()
                    fig, ax = plt.subplots(figsize=(8, 6))
                    sns.barplot(
                        x=individual_recommendations.index,
                        y=individual_recommendations.values,
                        palette="viridis",
                    )
                    plt.title("Recommendations Distribution (Individuals)")
                    plt.xlabel("Recommendation")
                    plt.ylabel("Number of Individuals")
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    st.pyplot(fig)
                else:
                    logger.warning(
                        "'Recommendations' column missing in person_leads_df."
                    )
                    st.warning("No 'Recommendations' data available for individuals.")

                # Correlation Heatmap of BANT Scores (Individuals)
                st.markdown("### Correlation Between BANT Components (Individuals)")
                bant_components_individual = [
                    "Budget Score",
                    "Authority Score",
                    "Need Score",
                    "Timeline Score",
                    "Overall BANT Score",
                ]
                if all(
                    component in st.session_state["person_leads_df"].columns
                    for component in bant_components_individual
                ):
                    correlation_individual = st.session_state["person_leads_df"][
                        bant_components_individual
                    ].corr()
                    fig, ax = plt.subplots(figsize=(8, 6))
                    sns.heatmap(
                        correlation_individual, annot=True, cmap="coolwarm", ax=ax
                    )
                    plt.title("Correlation Heatmap of BANT Scores (Individuals)")
                    plt.tight_layout()
                    st.pyplot(fig)
                else:
                    missing_components_individual = [
                        comp
                        for comp in bant_components_individual
                        if comp not in st.session_state["person_leads_df"].columns
                    ]
                    logger.warning(
                        f"Missing columns for individual correlation heatmap: {missing_components_individual}"
                    )
                    st.warning(
                        f"Missing columns for individual correlation heatmap: {', '.join(missing_components_individual)}"
                    )

                # Number of Phone Numbers and Emails per Individual
                st.markdown("### Number of Phone Numbers and Emails per Individual")
                if "Contacts" in st.session_state["person_leads_df"].columns:

                    def count_emails_and_phones_individual(contacts_entry):
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
                            email = contact.get("Email")
                            if email and email.strip().lower() not in [
                                "not provided",
                                "not available",
                                "",
                            ]:
                                num_emails += 1
                            phone = contact.get("Phone Number")
                            if phone and phone.strip().lower() not in [
                                "not provided",
                                "not available",
                                "",
                            ]:
                                num_phones += 1
                        return pd.Series(
                            {"Num Emails": num_emails, "Num Phones": num_phones}
                        )

                    try:
                        contacts_counts_individual = st.session_state[
                            "person_leads_df"
                        ]["Contacts"].apply(count_emails_and_phones_individual)
                        st.session_state["person_leads_df"] = st.session_state[
                            "person_leads_df"
                        ].join(contacts_counts_individual, rsuffix="_new")

                        # Display the counts in a table
                        try:
                            st.write("#### Emails and Phone Numbers per Individual")
                            st.write(
                                st.session_state["person_leads_df"][
                                    ["Name", "Num Emails", "Num Phones"]
                                ]
                            )
                        except KeyError:
                            logger.error(
                                "'Name', 'Num Emails', or 'Num Phones' columns missing in person_leads_df."
                            )
                            st.error(
                                "Required columns for individual contact analysis are missing."
                            )
                            st.stop()

                        # Plot number of emails and phone numbers per individual
                        fig, ax = plt.subplots(figsize=(10, 6))
                        sns.barplot(
                            data=st.session_state["person_leads_df"],
                            x="Name",
                            y="Num Emails",
                            color="steelblue",
                            label="Emails",
                        )
                        sns.barplot(
                            data=st.session_state["person_leads_df"],
                            x="Name",
                            y="Num Phones",
                            color="orange",
                            label="Phone Numbers",
                        )
                        ax.set_xlabel("Individual Name")
                        ax.set_ylabel("Count")
                        ax.set_title(
                            "Number of Emails and Phone Numbers per Individual"
                        )
                        plt.xticks(rotation=90)
                        plt.legend()
                        plt.tight_layout()
                        st.pyplot(fig)
                    except KeyError:
                        logger.error(
                            "'Name', 'Contacts', 'Num Emails', or 'Num Phones' columns missing in person_leads_df."
                        )
                        st.error(
                            "Required columns for individual contact analysis are missing."
                        )
                        st.stop()
                    except Exception as e:
                        logger.error(f"Error during individual contact analysis: {e}")
                        st.error(f"Error during individual contact analysis: {e}")
                        st.stop()
                else:
                    logger.warning("'Contacts' column missing in person_leads_df.")
                    st.warning(
                        "No 'Contacts' data available for individual contact analysis."
                    )

            else:
                st.warning(
                    "No individual-level BANT analysis data available. Please perform BANT ranking first."
                )

    def author_papers_section():
        """
        Section for Author Papers: Fetch, summarize, and visualize author papers and co-authors.
        """
        st.subheader("Author Papers and Network Visualization")

        authors = st.session_state["person_leads_df"]["Name"].unique().tolist()

        # Manual Search Input
        st.markdown("### Search for a Person")
        search_name = st.text_input(
            "Enter a person's name to search:", key="manual_search_input"
        )
        search_btn = st.button("Search Person", key="search_person_btn")

        if search_btn:
            if not st.session_state.get("context", "").strip():
                logger.error("Lead generation attempted with empty context.")
                st.warning("Please provide a context for lead generation.")
            if not search_name.strip():
                st.warning("Please enter a valid name.")
            else:
                # Search for the person in person_leads_df
                person_data = st.session_state["person_leads_df"][
                    st.session_state["person_leads_df"]["Name"].str.lower()
                    == search_name.strip().lower()
                ]
                if person_data.empty:
                    st.error(
                        f"No information found for '{search_name}'. Please ensure the name is correct."
                    )
                    st.stop()
                else:
                    person_info = person_data.iloc[0].to_dict()
                    st.markdown(f"### {person_info.get('Name', 'Unknown')}")
                    st.json(person_info, expanded=True)
                    st.markdown("---")

        # Select Author from Dropdown
        selected_author = st.selectbox(
            "Select an Author to Analyze",
            options=authors,
            key="selected_author_dropdown",
        )

        analyze_btn = st.button("Analyze Author", key="analyze_author_btn")
        if analyze_btn:
            if not selected_author:
                st.warning("Please select an author.")
                return

            with st.spinner(f"Fetching papers for {selected_author}..."):
                papers = fetch_recent_papers(selected_author)

            if not papers:
                st.error(f"No papers found for {selected_author}.")
                st.stop()
                return

            author_papers = []
            co_authors_set = set()
            for paper in papers:
                title = paper.get("title", "No Title")
                abstract = paper.get("abstract", "No Abstract Available.")
                summary = summarize_paper(title, abstract)
                co_authors = [
                    author["Name"]
                    for author in paper.get("authors", [])
                    if author["Name"] != selected_author
                ]
                author_papers.append(
                    {
                        "title": title,
                        "abstract": abstract,
                        "summary": summary,
                        "co_authors": co_authors,
                    }
                )
                co_authors_set.update(co_authors)

            # Fetch co-authors' papers and build the network
            co_authors_papers = {}
            for co_author in co_authors_set:
                papers = fetch_recent_papers(
                    co_author, limit=1
                )  # Fetch fewer papers for co-authors
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
                    title = paper.get("title", "No Title")
                    abstract = paper.get("abstract", "No Abstract Available.")
                    summary = summarize_paper(title, abstract)
                    abstracts.append(abstract)
                    co_authors = [
                        a["Name"]
                        for a in paper.get("authors", [])
                        if a["Name"] != author_name
                    ]
                    summarized_papers.append(
                        {
                            "title": title,
                            "abstract": abstract,
                            "summary": summary,
                            "co_authors": co_authors,
                        }
                    )
                authors_info[author_name] = {
                    "papers": summarized_papers,
                    "abstracts": abstracts,
                }

            # Calculate relevance using llm
            relevance_scores = calculate_relevance_with_llm(
                authors_info, st.session_state.get("context", "")
            )

            # Build Network Graph with relevance scores
            G = build_network_graph_with_relevance(authors_info, relevance_scores)

            # Visualize the Network Graph with enhancements
            visualize_network(
                G, selected_author, context=st.session_state.get("context", "")
            )

            # Display Information Cards
            display_author_information_cards(G, selected_author)

    def download_data_section():
        st.subheader("Download Data")

        # Option to download Leads Information
        if not st.session_state["leads_info_df"].empty:
            st.markdown("### Download Leads Information")
            download_leads_btn = st.button(
                "Download Leads Information", key="download_leads_info_btn"
            )
            if download_leads_btn:
                try:
                    download_leads(st.session_state["leads_info_df"], "leads_info")
                    st.success("Leads information section downloaded successfully!")
                    logger.info("Leads information section downloaded successfully.")
                except Exception as e:
                    logger.error(f"Error downloading leads_info: {e}")
                    st.error(f"Error downloading leads information: {e}")
                    st.stop()

        # Option to download Ranked Leads with BANT
        if (
            "company_bant_df" in st.session_state
            and not st.session_state["company_bant_df"].empty
        ):
            st.markdown("### Download Ranked Leads with BANT Scores")
            ranked_download_btn = st.button(
                "Download Ranked Leads with BANT", key="download_ranked_bant_btn"
            )
            if ranked_download_btn:
                try:
                    # Generate the BANT report with both individual and company data
                    bant_report = generate_bant_report(
                        st.session_state["person_leads_df"],
                        st.session_state["company_bant_df"],
                    )
                    if bant_report:
                        st.download_button(
                            label="Download BANT Report as Excel",
                            data=bant_report,
                            file_name="bant_report.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
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
        if not st.session_state["person_leads_df"].empty:
            st.markdown("### Download Person Leads")
            person_excel_filename = st.text_input(
                "Enter a name for the Persons Excel file (without extension):",
                value="person_leads",
                key="person_excel_input",
            )
            person_download_btn = st.button(
                "Download Person Leads Data", key="download_person_excel_btn"
            )
            if person_download_btn:
                if not person_excel_filename.strip():
                    st.warning("Please enter a valid filename for the Excel file.")
                else:
                    try:
                        download_leads(
                            st.session_state["person_leads_df"], person_excel_filename
                        )
                        st.success(
                            f"Person leads data downloaded as {person_excel_filename}.xlsx"
                        )
                        logger.info(
                            f"Person leads data downloaded as {person_excel_filename}.xlsx"
                        )
                    except Exception as e:
                        logger.error(f"Error downloading person_leads: {e}")
                        st.error(f"Error downloading person leads: {e}")
                        st.stop()

        # Option to download Company BANT Scores
        if (
            "company_bant_df" in st.session_state
            and not st.session_state["company_bant_df"].empty
        ):
            st.markdown("### Download Company BANT Scores")
            company_bant_download_btn = st.button(
                "Download Company BANT Scores", key="download_company_bant_btn"
            )
            if company_bant_download_btn:
                try:
                    download_leads(
                        st.session_state["company_bant_df"], "company_bant_scores"
                    )
                    st.success("Company BANT scores downloaded successfully!")
                    logger.info("Company BANT scores downloaded successfully.")
                except Exception as e:
                    logger.error(f"Error downloading company_bant_df: {e}")
                    st.error(f"Error downloading company BANT scores: {e}")
                    st.stop()

        # Option to download all data together (ZIP functionality can be implemented as needed)
        st.markdown("### Download All Data")
        all_download_btn = st.button(
            "Download All Data as ZIP", key="download_all_data_btn"
        )
        if all_download_btn:
            logger.warning("ZIP download functionality is not implemented yet.")
            st.warning("ZIP download functionality is not implemented yet.")

    def rank_leads_section():
        st.subheader("Rank the Leads using BANT Analysis")

        # Retrieve context from session_state
        context = st.session_state.get("context", "")

        if not context.strip():
            st.warning(
                "No context provided. Please go back to 'Input Leads' to provide context."
            )
            if st.button("Go to Input Leads"):
                st.session_state["menu_selection"] = "Input Leads"
                st.experimental_rerun()
            return

        st.markdown("**Context for BANT Analysis:**")
        st.text_area(
            "Context for BANT analysis:",
            value=context,
            height=150,
            key="bant_context",
            disabled=True,
        )

        rank_btn = st.button(
            "Perform BANT Analysis and Rank Leads", key="bant_rank_leads_btn"
        )
        if rank_btn:
            if st.session_state["person_leads_df"].empty:
                logger.error("Lead ranking attempted without person leads information.")
                st.error(
                    "No person leads information available. Please scrape lead information first."
                )
                return
            else:
                with st.spinner("Performing BANT analysis and ranking leads..."):
                    person_leads_df, company_bant_df = rank_leads_with_bant(
                        st.session_state["leads_info_df"],
                        st.session_state["person_leads_df"],
                        st.session_state.get("context", ""),
                    )
                    st.session_state["person_leads_df"] = person_leads_df
                    st.session_state["company_bant_df"] = company_bant_df
                    st.success("BANT analysis and ranking completed successfully!")
                    logger.info("BANT analysis and ranking completed successfully.")

        # Display the Company-Level BANT DataFrame
        if (
            "company_bant_df" in st.session_state
            and not st.session_state["company_bant_df"].empty
        ):
            st.write("### Company-Level BANT Scores and Recommendations")
            try:
                company_bant_df = st.session_state["company_bant_df"]
                display_and_download(
                    df=company_bant_df,
                    button_label="Company BANT Scores",
                    filename="company_bant_scores",
                )
            except Exception as e:
                logger.error(f"Error displaying company_bant_df: {e}")
                st.error(f"Error displaying company BANT scores: {e}")
                return

        # Display the Individual-Level BANT DataFrame
        if not st.session_state["person_leads_df"].empty:
            st.write("### Individual-Level BANT Scores and Recommendations")
            display_lead_information(
                df=st.session_state["person_leads_df"],
                button_label="Individual BANT Scores",
                filename="individual_bant_scores",
            )

        st.session_state["rank_leads_bant"] = True
        st.success("Rank Leads section loaded successfully!")

    def display_and_download(df, button_label, filename, height=400):
        """
        Displays a DataFrame and provides options to download it as CSV or Excel.
        """
        df = clean_dataframe(df)
        st.dataframe(df, height=height)

        csv = df.to_csv(index=False).encode("utf-8")

        excel_buffer = BytesIO()
        try:
            with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
                df.to_excel(writer, index=False)
            excel_data = excel_buffer.getvalue()
        except Exception as e:
            logger.error(f"Error converting DataFrame to Excel: {e}")
            excel_data = None

        st.download_button(
            label=f"Download {button_label} as CSV",
            data=csv,
            file_name=f"{filename}.csv",
            mime="text/csv",
        )

        if excel_data:
            st.download_button(
                label=f"Download {button_label} as Excel",
                data=excel_data,
                file_name=f"{filename}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
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
        - **Rank Leads:** Prioritize leads based on configurable weights.
        - **Analytics:** Visualize and analyze lead data.
        - **Author Papers:** Analyze authors' papers and visualize co-authorship networks.
        - **Download Data:** Download the processed data in various formats.
        """
    )
    authenticator.logout("Logout", "sidebar")

    # Navigation Menu
    if st.session_state["menu_selection"] == "Input Leads":
        input_leads()
    elif st.session_state["menu_selection"] == "Analyse Lead Information":
        scrape_lead_information()
    elif st.session_state["menu_selection"] == "Rank Leads (BANT)":
        rank_leads_section()
    elif st.session_state["menu_selection"] == "Analytics":
        analytics_section()
    elif st.session_state["menu_selection"] == "Author Papers":
        author_papers_section()
    elif st.session_state["menu_selection"] == "Download Data":
        download_data_section()
