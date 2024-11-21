# analysis.py

import pandas as pd
import re
import time
import logging
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from semanticscholar import SemanticScholar
import json
from gpt_integration import connect_interests_to_context
import streamlit as st
from openai import OpenAI
from constants import OPENAI_API_KEY, GPT_MODEL

client = OpenAI(api_key=OPENAI_API_KEY)

logger = logging.getLogger(__name__)

# Initialize the Semantic Scholar client
sch = SemanticScholar()

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

def build_individual_network(people, context):
    """
    Builds a network of individuals including co-authors from Semantic Scholar.
    Returns a dictionary with network data and a matplotlib figure.
    """
    try:
        G = nx.Graph()
        network_data = []

        # Add initial individuals to the graph
        for person in people:
            person_name = person.get('name')
            G.add_node(person_name, role=person.get('role'), work_area=person.get('work_area'))
            # Get co-authors
            coauthors = get_coauthors_from_semantic_scholar(person_name)
            for coauthor in coauthors:
                coauthor_name = coauthor.get('name')
                G.add_node(coauthor_name)
                G.add_edge(person_name, coauthor_name)
                # Get co-author details
                coauthor_details = get_person_details_from_semantic_scholar(coauthor.get('authorId'), context)
                if coauthor_details:
                    network_data.append(coauthor_details)

        # Create a DataFrame from network_data
        network_df = pd.DataFrame(network_data)

        # Create a network graph
        pos = nx.spring_layout(G)
        fig, ax = plt.subplots(figsize=(12, 8))
        nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=1500, font_size=10, ax=ax)
        plt.tight_layout()

        return {'network_df': network_df, 'graph_fig': fig}
    except Exception as e:
        logger.error(f"Error building individual network: {e}")
        return None

def get_coauthors_from_semantic_scholar(person_name):
    """
    Given a person's name, use Semantic Scholar to find co-authors.
    """
    try:
        # Search for the author
        results = sch.search_author(person_name, limit=1)
        if not results['data']:
            logger.warning(f"No author found for '{person_name}' in Semantic Scholar.")
            return []
        author = results['data'][0]
        author_id = author['authorId']

        # Get the author's publications
        author_details = sch.get_author(author_id, params={'include_papers': 'true'})
        papers = author_details.get('papers', [])

        coauthors = {}
        for paper in papers:
            paper_authors = paper.get('authors', [])
            for coauthor in paper_authors:
                coauthor_id = coauthor.get('authorId')
                coauthor_name = coauthor.get('name')
                if coauthor_id != author_id:
                    coauthors[coauthor_id] = coauthor_name

        coauthor_list = [{'authorId': k, 'name': v} for k, v in coauthors.items()]
        return coauthor_list

    except Exception as e:
        logger.error(f"Error getting co-authors for '{person_name}': {e}")
        return []

def get_person_details_from_semantic_scholar(author_id, context):
    """
    Given an author's Semantic Scholar ID, get details about them.
    """
    try:
        author_details = sch.get_author(author_id, params={'include_papers': 'true'})
        name = author_details.get('name', 'Not Available')
        affiliation = author_details.get('affiliations', 'Not Available')
        # Semantic Scholar may not provide emails or positions, so we might need to scrape that separately

        # Get research interests (could be based on their top publications)
        papers = author_details.get('papers', [])
        research_interests = set()
        recent_papers = []
        for paper in papers[:5]:  # Get top 5 papers
            title = paper.get('title', '')
            abstract = paper.get('abstract', '')
            research_interests.update(abstract.split())
            recent_papers.append({
                'title': title,
                'year': paper.get('year', 'Not Available'),
                'abstract': abstract
            })

        # Connect research interests to context using GPT
        interest_connection = connect_interests_to_context(research_interests, context)

        person_info = {
            'Name': name,
            'Affiliation': affiliation,
            'Research Interests': ', '.join(list(research_interests)[:10]),
            'Recent Publications': recent_papers,
            'Interest Connection': interest_connection,
            'Email': 'Not Available',
            'Position': 'Not Available',
            'Author ID': author_id
        }

        return person_info

    except Exception as e:
        logger.error(f"Error getting details for author ID '{author_id}': {e}")
        return None

def connect_interests_to_context(research_interests, context):
    """
    Uses GPT to explain how the research interests connect to the context.
    """

    interests_text = ', '.join(list(research_interests))
    prompt = f"""
Given the research interests: {interests_text}

And the context: {context}

Explain how the research interests connect to the context.
"""

    try:
        response = client.chat.completions.create(
            model=GPT_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )
        connection = response.choices[0].message.content.strip()
        return connection
    except Exception as e:
        logger.error(f"Error connecting interests to context: {e}")
        return "Not Available"
