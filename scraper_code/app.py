# app.py

import streamlit as st
import pandas as pd
import json
import os
import config
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re

# Set global font settings for the plots
plt.rcParams.update({
    'axes.labelsize': 18,  # Axis label size
    'axes.labelweight': 'bold',  # Axis label weight
    'xtick.labelsize': 18,  # X-axis tick label size
    'ytick.labelsize': 18,  # Y-axis tick label size
    'font.size': 18,  # General font size
    'font.weight': 'bold',  # General font weight
    'axes.titlesize': 20,  # Title size
    'axes.titleweight': 'bold',  # Title weight
})

def load_metadata(csv_path: str) -> pd.DataFrame:
    """Load metadata from CSV."""
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        return df
    else:
        st.error(f"CSV file not found at {csv_path}")
        return pd.DataFrame()

def display_summary(df: pd.DataFrame):
    st.header("Overall Dataset Summary")
    st.write(f"**Total Papers Collected:** {len(df)}")
    st.write(f"**Number of Valid Leads:** {df['Contact Emails'].notna().sum()}")

    # Display plots
    st.subheader("Data Visualizations")

    # 1. Distribution of Journals
    plt.figure(figsize=(10, 6))
    sns.countplot(y='Journal', data=df, order=df['Journal'].value_counts().index[:10], palette='Set2')
    plt.title('Top 10 Journals by Number of Papers')
    plt.xlabel('Count')
    plt.ylabel('Journal')
    st.pyplot(plt)
    plt.close()

    # 2. Distribution of Keywords
    all_keywords = ', '.join(df['Keywords'].dropna()).split(', ')
    keyword_counts = Counter(all_keywords)
    keywords_df = pd.DataFrame(keyword_counts.items(), columns=['Keyword', 'Frequency'])
    keywords_df = keywords_df.sort_values(by='Frequency', ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(x='Frequency', y='Keyword', data=keywords_df.head(10), palette='muted')
    plt.title('Top 10 Most Frequent Keywords')
    plt.xlabel('Frequency')
    plt.ylabel('Keyword')
    st.pyplot(plt)
    plt.close()

    # 3. Author Analysis
    all_authors = ', '.join(df['Authors'].dropna()).split(', ')
    excluded_authors = {'et al.', 'et al', 'author two', 'author three', 'author four', 'author five'}
    filtered_authors = [author.strip() for author in all_authors if author.lower() not in excluded_authors and len(author.strip()) > 2]
    author_counts = Counter(filtered_authors)
    authors_df = pd.DataFrame(author_counts.items(), columns=['Author', 'Frequency'])
    authors_df = authors_df.sort_values(by='Frequency', ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(x='Frequency', y='Author', data=authors_df.head(10), palette='coolwarm')
    plt.title('Top 10 Most Frequent Authors')
    plt.xlabel('Frequency')
    plt.ylabel('Author')
    st.pyplot(plt)
    plt.close()

    # 4. University Analysis
    all_universities = df['Institute of Origin'].dropna().tolist()
    all_universities = [university for university in all_universities if 'Materials Theory' not in university]
    university_counts = Counter(all_universities)
    universities_df = pd.DataFrame(university_counts.items(), columns=['University', 'Frequency'])
    universities_df = universities_df.sort_values(by='Frequency', ascending=False)

    plt.figure(figsize=(12, 8))
    sns.barplot(x='Frequency', y='University', data=universities_df.head(10), palette='coolwarm')
    plt.title('Top 10 Most Frequent Universities (Excluding Materials Theory)')
    plt.xlabel('Frequency')
    plt.ylabel('University')
    st.pyplot(plt)
    plt.close()

def display_leads(df: pd.DataFrame):
    st.sidebar.header("Leads")
    valid_leads = df[df['Contact Emails'].notna()]
    top_leads = valid_leads.head(10)
    selected_doi = st.sidebar.selectbox("Select a Lead", options=top_leads['DOI'], index=0)

    return selected_doi

def display_actions():
    st.sidebar.header("Actions")
    st.sidebar.button("Delete Lead")
    st.sidebar.button("Add Lead")
    st.sidebar.button("Send Email")
    st.sidebar.button("Send Mass Email")
    st.sidebar.button("Edit Lead")
    st.sidebar.button("Export Data")
    # Add more dummy buttons as needed

def display_report(selected_lead: pd.Series):
    if selected_lead.empty:
        st.info("Select a lead from the sidebar to view the report.")
        return

    st.header("Lead Report")

    st.subheader("Reason for Being a Good Lead")
    st.write(selected_lead.get("reasoning", "N/A"))

    st.subheader("Background Information of Principal Investigator and Co-authors")
    st.write(selected_lead.get("researcher_background", "N/A"))
    st.write(selected_lead.get("team_background", "N/A"))

    st.subheader("How They Could Use Quantum Nuova (QT-Sense Device)")
    st.write(selected_lead.get("summary", "N/A"))

    st.subheader("Contact Details and Address")
    st.write(f"**Contact Emails:** {selected_lead.get('Contact Emails', 'N/A')}")
    st.write(f"**Contact Phones:** {selected_lead.get('Contact Phones', 'N/A')}")
    st.write(f"**Address:** {selected_lead.get('address', 'N/A')}")

    st.subheader("Recent Grants")
    st.write(selected_lead.get("recent_grants", "N/A"))

def main():
    st.set_page_config(layout="wide")
    st.title("Insights App for Quantum Sensing Tech Leads")

    # Load metadata
    df = load_metadata(config.CSV_PATH)

    if df.empty:
        st.warning("No data available to display.")
        return

    # Sidebar for leads and actions
    selected_doi = display_leads(df)
    display_actions()

    # Main layout: two columns
    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Top 10 Clickable Leads")
        display_leads_info = df[df['DOI'] == selected_doi]
        if not display_leads_info.empty:
            lead_info = display_leads_info.iloc[0]
            st.write(f"**DOI:** {lead_info['DOI']}")
            st.write(f"**Title:** {lead_info['Title']}")
            st.write(f"**Authors:** {lead_info['Authors']}")
            st.write(f"**Journal:** {lead_info['Journal']}")

    with col2:
        st.subheader("Report on Lead")
        selected_lead = df[df['DOI'] == selected_doi].iloc[0] if not df[df['DOI'] == selected_doi].empty else pd.Series()
        display_report(selected_lead)

    # Display summary and visualizations
    st.subheader("Overall Dataset Summary")
    display_summary(df)

if __name__ == "__main__":
    main()
