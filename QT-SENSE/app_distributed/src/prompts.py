prompt_extract_companies_from_text = """
You are an AI assistant that extracts company names from the following text. The companies are likely to be exhibitors, attendees, sponsors, or partners of a url related to the context provided.

Please provide the extracted information in JSON format only. Ensure the JSON is valid and properly formatted. Do not include any markdown or additional text.

Example:
["Company A", "Company B", "Company C"]
"""

prompt_generate_leads_with_llm = """Output the leads as a JSON array with each lead in the following format:
    [
        {"Entity": "Entity Name 1"},
        {"Entity": "Entity Name 2"},
        ...
    ]
    Ensure that:
    - The JSON syntax is correct with proper commas and brackets.
    - Each lead object contains only the "Entity" key with a string value.
    """

prompt_extract_person_info_with_llm = """You are an AI assistant specialized in extracting detailed person profiles from text. Extract the following information about the person and provide it in the specified JSON format only. Do not include any additional text, explanations, or markdown.

    **Instructions:**
    - Populate each field with the most relevant information extracted from the text.
    - If a field's information is not available, set it to "Not Available".
    - Ensure the JSON is valid, properly formatted, and includes all specified keyword fields.
    
    **Output:**
    {
        "keyword 1": "...",
        "keyword 2": "..."
    }
    """

prompt_extract_lead_info_with_llm_per_field = """
You are an AI assistant specialized in extracting business information. Based on the provided scraped data, extract the following details about the company in JSON format only. Do not include any additional text, explanations, or markdown.



**Instructions:**
- Populate each field with the most relevant information extracted from the text.
- If a field's information is not available, set it to "Not Available".
- Ensure the JSON is valid, properly formatted, and includes all specified fields.

**Output Structure Example (fields may differ):**
{
    "Entity": "Genentech",
    "Category": "Biotechnology Company",
    "CEO/PI": "Ashley Magargee",
    "Researchers": [
        "Aaron Wecksler",
        "Adeyemi Adedeji",
        ...
    ],
    "Grants": [
        {
            "Name": "Cancer Research Initiative",
            "Amount": "$500,000",
            "Period": "2 years",
            "Start Date": "2022-01-15"
        },
        ...
    ],
    "Phone Number": {
        "number": "(650) 225-1000",
        "purpose": "Corporate Office"
    },
    "Email": {
        "address": "patientinfo@gene.com",
        "purpose": "Patient Resource Center"
    },
    "Country": "USA",
    "University": "Stanford University",
    "Summary": "Genentech is a leading biotechnology company focused on developing medicines for serious medical conditions.",
    "Contacts": [
        {
            "Name": "Ashley Magargee",
            "Title": "Chief Executive Officer"
        },
        ...
    ]
}

===========
"""