# gpt_integration.py

from openai import OpenAI
from constants import OPENAI_API_KEY, GPT_MODEL
from utils import count_tokens, clean_response
import logging
import json

logger = logging.getLogger(__name__)

client = OpenAI(api_key=OPENAI_API_KEY)

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
                model=GPT_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2
            )

            names_text = clean_response(response.choices[0].message.content)

            logger.info(f"Raw GPT Response for Lead Generation ({lead_type}):")
            logger.info(names_text)

            lead_names = json.loads(names_text)
            leads.extend([(lead['name'], lead['type']) for lead in lead_names[:num_leads]])
        except json.JSONDecodeError:
            logger.error(f"Failed to parse the response from GPT for {lead_type}. Please try again.")
        except Exception as e:
            logger.error(f"An error occurred while generating leads for {lead_type}: {e}")

    return leads

def extract_lead_info_with_gpt(text, columns_to_retrieve, lead_name, lead_category, source_urls):
    """
    Sends scraped lead information to GPT for extraction based on user-specified fields.
    """
    encoding = 'cl100k_base'
    model_max_tokens = 8192  # Adjust based on your OpenAI plan

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
            model=GPT_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )

        lead_info_text = clean_response(response.choices[0].message.content)

        logger.info(f"Raw GPT Response for Lead Information Extraction ('{lead_name}'):")
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
    model_max_tokens = 8192  # Adjust based on your OpenAI plan

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
            model=GPT_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )

        person_info_text = clean_response(response.choices[0].message.content)

        logger.info(f"Raw GPT Response for Person Information Extraction ('{person_name}'):")
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

def analyze_text_with_gpt(text, context):
    """
    Uses GPT to analyze the text and extract information, and generate a report.
    """
    # Build the prompt
    prompt = f"""
You are an AI assistant that analyzes the given text and extracts the following information:

- Identify any companies or research groups mentioned.
- For each company or research group, extract:
    - The name of the company or research group.
    - What the company or research group is working on.
    - A detailed BANT analysis, including:
        - Budget: Provide a score (1-5), explanation, and sources from the text.
        - Authority: Provide a score (1-5), explanation, and sources from the text.
        - Need: Provide a score (1-5), explanation, and sources from the text.
        - Timeline: Provide a score (1-5), explanation, and sources from the text.
    - A list of people associated with it, including:
        - Name
        - Role
        - What they are working on
        - A detailed BANT analysis for each person, including:
            - Budget: Provide a score (1-5), explanation, and sources from the text.
            - Authority: Provide a score (1-5), explanation, and sources from the text.
            - Need: Provide a score (1-5), explanation, and sources from the text.
            - Timeline: Provide a score (1-5), explanation, and sources from the text.
- Ensure that all explanations and justifications are linked to specific information found in the text.
- Use the context provided to align the analysis.

Context (if any): {context}

Text to analyze:
{text}

Please provide the output in JSON format with the following structure:

{{
    "companies": [
        {{
            "name": "Company or Research Group Name",
            "work_area": "What the company/research group is working on",
            "BANT": {{
                "Budget": {{
                    "score": 1-5,
                    "explanation": "Explanation of the score",
                    "sources": ["Relevant excerpts from the text"]
                }},
                "Authority": {{
                    "score": 1-5,
                    "explanation": "Explanation of the score",
                    "sources": ["Relevant excerpts from the text"]
                }},
                "Need": {{
                    "score": 1-5,
                    "explanation": "Explanation of the score",
                    "sources": ["Relevant excerpts from the text"]
                }},
                "Timeline": {{
                    "score": 1-5,
                    "explanation": "Explanation of the score",
                    "sources": ["Relevant excerpts from the text"]
                }}
            }},
            "people": [
                {{
                    "name": "Person Name",
                    "role": "Role",
                    "work_area": "What they are working on",
                    "BANT": {{
                        "Budget": {{
                            "score": 1-5,
                            "explanation": "Explanation of the score",
                            "sources": ["Relevant excerpts from the text"]
                        }},
                        "Authority": {{
                            "score": 1-5,
                            "explanation": "Explanation of the score",
                            "sources": ["Relevant excerpts from the text"]
                        }},
                        "Need": {{
                            "score": 1-5,
                            "explanation": "Explanation of the score",
                            "sources": ["Relevant excerpts from the text"]
                        }},
                        "Timeline": {{
                            "score": 1-5,
                            "explanation": "Explanation of the score",
                            "sources": ["Relevant excerpts from the text"]
                        }}
                    }}
                }},
                ...
            ]
        }},
        ...
    ]
}}

Ensure that the output is valid JSON.
"""

    # Now send the prompt to GPT and get the response
    try:
        response = client.chat.completions.create(
            model=GPT_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )

        response_text = clean_response(response.choices[0].message.content)
        logger.info("GPT response for document analysis:")
        logger.info(response_text)

        analysis_result = json.loads(response_text)

        return analysis_result

    except json.JSONDecodeError:
        logger.error("Failed to parse the GPT response for document analysis.")
        return None
    except Exception as e:
        logger.error(f"An error occurred during document analysis: {e}")
        return None
