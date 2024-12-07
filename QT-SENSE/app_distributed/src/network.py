import streamlit as st
import json
import networkx as nx
import requests
import plotly.graph_objects as go
import matplotlib.pyplot as plt

import logging
from dotenv import load_dotenv

# import from openai_api.py
from llm_center import llm_reasoning

# import from utils.py
from utils import (
    summarize_paper
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

SEMANTIC_SCHOLAR_BASE_URL = "https://api.semanticscholar.org/graph/v1"
SEMANTIC_SCHOLAR_HEADERS = {
    "Content-Type": "application/json"
}

load_dotenv()

def build_network_graph_with_relevance(authors_info, relevance_scores):
    """
    Builds a network graph from authors' papers and their co-authors, including relevance scores.
    """
    G = nx.Graph()
    for author, info in authors_info.items():
        G.add_node(author, type='Author', info={
            'Name': author,
            'papers': info['papers'],
            'relevance': relevance_scores.get(author, 0.5)
        })
        for paper in info['papers']:
            co_authors = paper.get('co_authors', [])
            for co_author in co_authors:
                if co_author not in authors_info:
                    continue  # Skip co-authors not in our list
                if not G.has_node(co_author):
                    G.add_node(co_author, type='Co-Author', info={
                        'Name': co_author,
                        'papers': authors_info[co_author]['papers'],
                        'relevance': relevance_scores.get(co_author, 0.5)
                    })
                # Store papers shared between authors
                if G.has_edge(author, co_author):
                    G[author][co_author]['papers'].append({
                        'title': paper.get('title'),
                        'summary': paper.get('summary')
                    })
                else:
                    G.add_edge(author, co_author, papers=[{
                        'title': paper.get('title'),
                        'summary': paper.get('summary')
                    }])
    return G

def relevance_to_color(score):
    """
    Maps a relevance score between 0 and 1 to a color from blue (cold) to red (warm).
    """
    # RdBu colormap ranges from blue to red
    colorscale = plt.cm.RdBu
    rgba = colorscale(score)
    # Convert RGBA to hex
    color = '#%02x%02x%02x' % (int(rgba[0]*255), int(rgba[1]*255), int(rgba[2]*255))
    return color

def calculate_relevance_with_llm(authors_info, context):
    """
    Uses llm to rank authors based on their relevance to the context.
    Returns a dictionary with authors and their relevance scores.
    """
    prompt = f"""
You are an expert assistant specialized in evaluating academic authors' relevance based on their research abstracts and a given context.

**Context:**
{context}

**Authors and Their Paper Abstracts:**

"""
    for author_name, info in authors_info.items():
        prompt += f"\n**Author:** {author_name}\n"
        for idx, abstract in enumerate(info['abstracts'], 1):
            prompt += f"Paper {idx} Abstract: {abstract}\n"
    
    prompt += """
**Task:**
Evaluate each author based on how closely their research aligns with the provided context. Assign a relevance score between 0 and 1, where 1 indicates the highest relevance.

**Instructions:**
- Analyze the abstracts of each author's papers in relation to the context.
- Consider factors like research topics, methodologies, findings, and applications.
- Provide a concise justification for each score in the "details" field.

**Output Format:**
Provide the results as a JSON array of objects with the following structure, sorted from highest to lowest score:

[
  {{
    "author": "Author A",
    "score": 0.95,
    "details": "Author A's research on [specific topic] closely matches the context."
  }},
  {{
    "author": "Author B",
    "score": 0.75,
    "details": "Author B's work on [another topic] is somewhat relevant to the context."
  }},
  ...
]

**Example:**
[
  {{
    "author": "Dr. Jane Smith",
    "score": 0.90,
    "details": "Dr. Smith's research on renewable energy technologies is highly relevant to the provided context."
  }},
  {{
    "author": "Prof. John Doe",
    "score": 0.65,
    "details": "Prof. Doe's studies on environmental policy have some alignment with the context."
  }}
]
"""

    try:
        relevance_data = llm_reasoning(prompt)
        relevance_list = json.loads(relevance_data)
        relevance_scores = {item['author']: item['score'] for item in relevance_list}
        return relevance_scores
    except Exception as e:
        logger.error(f"Error calculating relevance with llm: {e}")
        # If error occurs, assign neutral relevance
        return {author: 0.5 for author in authors_info.keys()}

def fetch_recent_papers(author_name, limit=10):
    """
    Fetches the most recent papers of an author from Semantic Scholar.
    """
    try:
        # Search for the author
        search_url = f"{SEMANTIC_SCHOLAR_BASE_URL}/author/search"
        params = {
            "query": author_name,
            "limit": 1  # We assume the first result is the correct author
        }
        response = requests.get(search_url, headers=SEMANTIC_SCHOLAR_HEADERS, params=params)
        response.raise_for_status()
        data = response.json()
        if not data.get('data'):
            logger.warning(f"No author found for name: {author_name}")
            return []
        author_id = data['data'][0]['authorId']
        
        # Fetch author's papers
        papers_url = f"{SEMANTIC_SCHOLAR_BASE_URL}/author/{author_id}/papers"
        params = {
            "limit": limit,
            "fields": "title,abstract,authors,year,url"
        }
        response = requests.get(papers_url, headers=SEMANTIC_SCHOLAR_HEADERS, params=params)
        response.raise_for_status()
        papers_data = response.json()
        papers = papers_data.get('data', [])
        return papers
    except Exception as e:
        logger.error(f"Error fetching papers for {author_name}: {e}")
        return []

def fetch_author_details(author_ids):
    """
    Fetches detailed information about multiple authors using the batch endpoint.
    """
    try:
        url = f"{SEMANTIC_SCHOLAR_BASE_URL}/author/batch"
        query_params = {
            "fields": "name,url,paperCount,hIndex,papers.title,papers.abstract"
        }
        data = {
            "ids": author_ids
        }
        response = requests.post(url, params=query_params, json=data, headers=SEMANTIC_SCHOLAR_HEADERS)
        response.raise_for_status()
        authors = response.json()
        return authors
    except Exception as e:
        logger.error(f"Error fetching author details: {e}")
        return []

def display_author_information_cards(G, selected_author):
    """
    Displays information cards for each author with their details and network visualization.
    """
    authors = [node for node in G.nodes if node != selected_author and G.nodes[node]['info'].get('type') == 'Person']
    relevance_scores = [calculate_relevance_with_llm(name, '') for name in authors]
    
    # Sort authors based on relevance score
    sorted_authors = [author for _, author in sorted(zip(relevance_scores, authors), key=lambda x: x[0], reverse=True)]
    total_authors = len(sorted_authors)
    
    if total_authors == 0:
        st.info("No co-authors found.")
        return
    
    # Slider to select author index
    author_idx = st.slider("Select Co-Author", min_value=0, max_value=total_authors - 1, value=0, key='author_slider')
    co_author_name = sorted_authors[author_idx]
    
    # Fetch co-author details
    co_author_info = G.nodes[co_author_name]['info']
    co_author_papers = co_author_info.get('papers', [])
    relevance = 0.5  # Placeholder if relevance is calculated
    
    # Display the information card
    st.markdown(f"### Author: {co_author_name}")
    st.write(f"**Relevance Score:** {relevance:.2f}")
    st.write(f"**Number of Papers:** {len(co_author_papers)}")
    
    # Display co-author's papers
    st.markdown("#### Papers:")
    for paper in co_author_papers:
        st.write(f"- **Title:** {paper.get('title')}")
        st.write(f"  **Summary:** {paper.get('summary')}")
        st.markdown("---")
    
    # Optionally, display a mini-network centered on the co-author
    st.markdown("#### Co-Authorship Network:")
    sub_G = G.subgraph([selected_author, co_author_name] + list(G.neighbors(co_author_name)))
    visualize_network(sub_G, selected_author)

def visualize_network(G, selected_author, context=''):
    """
    Visualizes the network graph using Plotly with detailed hover information.
    Applies color coding based on relevance to the context.
    Includes both companies and persons as nodes.
    """
    pos = nx.spring_layout(G, k=0.5, iterations=50)
    
    # Prepare edge traces
    edge_x = []
    edge_y = []
    edge_text = []
    for edge in G.edges(data=True):
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        shared_papers = edge[2].get('papers', [])
        titles = [paper['title'] for paper in shared_papers]
        edge_text.append('<br>'.join(titles))
    
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1, color='#888'),
        hoverinfo='text',
        mode='lines',
        text=edge_text
    )
    
    # Prepare node traces
    node_x = []
    node_y = []
    node_text = []
    node_color = []
    node_size = []
    for node in G.nodes(data=True):
        x, y = pos[node[0]]
        node_x.append(x)
        node_y.append(y)
        info = node[1].get('info', {})
        name = info.get('Name', node[0])
        node_text.append(f"Name: {name}")
        
        # Determine node type: Company or Person
        node_type = info.get('type', 'Person')  # Default to 'Person' if not specified
        if node_type == 'Company':
            color = 'blue'
            size = 20
        else:
            color = 'green'
            size = 10
        node_color.append(color)
        node_size.append(size)
    
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        text=node_text,
        marker=dict(
            color=node_color,
            size=node_size,
            line_width=2,
        )
    )
    
    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        title=f'Co-Authorship Network for {selected_author}',
                        titlefont_size=16,
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20, l=5, r=5, t=40),
                        annotations=[dict(
                            text="",
                            showarrow=False,
                            xref="paper", yref="paper")],
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                    )
    
    st.plotly_chart(fig, use_container_width=True)


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
            st.rerun()
        return
    
    authors = st.session_state['person_leads_df']['Name'].unique().tolist()
    
    # Manual Search Input
    st.markdown("### Search for a Person")
    search_name = st.text_input("Enter a person's name to search:", key='manual_search_input')
    search_btn = st.button("Search Person", key='search_person_btn')
    
    if search_btn:
        if not search_name.strip():
            st.warning("Please enter a valid name.")
        else:
            # Search for the person in person_leads_df
            person_data = st.session_state['person_leads_df'][
                st.session_state['person_leads_df']['Name'].str.lower() == search_name.strip().lower()
            ]
            if person_data.empty:
                st.error(f"No information found for '{search_name}'. Please ensure the name is correct.")
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
