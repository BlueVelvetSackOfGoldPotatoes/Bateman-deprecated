import pandas as pd
import json
import openpyxl
import matplotlib.pyplot as plt
from io import BytesIO
import logging
from dotenv import load_dotenv

# import from openai_api.py
from llm_center import llm_reasoning

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
logger = logging.getLogger(__name__)

load_dotenv()

def generate_bant_report(person_leads_df, company_bant_df):
    """
    Generates a comprehensive BANT report including individual and company-level analyses.
    
    :param person_leads_df: DataFrame containing individual-level BANT scores.
    :param company_bant_df: DataFrame containing company-level aggregated BANT scores.
    :return: BytesIO object containing the report (Excel format).
    """
    try:
        # Initialize the Excel writer
        report_buffer = BytesIO()
        with pd.ExcelWriter(report_buffer, engine='openpyxl') as writer:
            # Sheet 1: Individual BANT Scores
            person_leads_df.to_excel(writer, sheet_name='Individual BANT', index=False)
            
            # Sheet 2: Company-Level BANT Scores
            company_bant_df.to_excel(writer, sheet_name='Company BANT', index=False)
            
            # Sheet 3: Summary Statistics for Individuals
            individual_summary = person_leads_df.describe()
            individual_summary.to_excel(writer, sheet_name='Individual Summary')
            
            # Sheet 4: Summary Statistics for Companies
            company_summary = company_bant_df.describe()
            company_summary.to_excel(writer, sheet_name='Company Summary')
            
            # Sheet 5: Recommendations for Individuals
            individual_recommendations = person_leads_df[['Name', 'Name', 'Recommendations']]
            individual_recommendations.to_excel(writer, sheet_name='Individual Recommendations', index=False)
            
            # Sheet 6: Recommendations for Companies
            company_recommendations = company_bant_df[['Entitys', 'Recommendations']]
            company_recommendations.to_excel(writer, sheet_name='Company Recommendations', index=False)
            
            # Sheet 7: Graphs
            worksheet = writer.book.create_sheet("Graphs")
            
            # Graph 1: Distribution of Overall BANT Scores (Individuals)
            plt.figure(figsize=(10, 6))
            plt.hist(person_leads_df['Overall BANT Score'], bins=10, color='skyblue', edgecolor='black')
            plt.title('Distribution of Individual Overall BANT Scores')
            plt.xlabel('Overall BANT Score')
            plt.ylabel('Number of Individuals')
            plt.tight_layout()
            graph_buffer = BytesIO()
            plt.savefig(graph_buffer, format='png')
            plt.close()
            graph_buffer.seek(0)
            img = openpyxl.drawing.image.Image(graph_buffer)
            worksheet.add_image(img, 'A1')
            
            # Graph 2: Distribution of Overall BANT Scores (Companies)
            plt.figure(figsize=(10, 6))
            plt.hist(company_bant_df['Overall BANT Score'], bins=10, color='salmon', edgecolor='black')
            plt.title('Distribution of Company Overall BANT Scores')
            plt.xlabel('Overall BANT Score')
            plt.ylabel('Number of Companies')
            plt.tight_layout()
            graph_buffer2 = BytesIO()
            plt.savefig(graph_buffer2, format='png')
            plt.close()
            graph_buffer2.seek(0)
            img2 = openpyxl.drawing.image.Image(graph_buffer2)
            worksheet.add_image(img2, 'A20')
            
            # Graph 3: Average BANT Components per Company
            plt.figure(figsize=(12, 8))
            bant_components = ['Average Budget Score', 'Average Authority Score', 'Average Need Score', 'Average Timeline Score']
            for component in bant_components:
                plt.hist(company_bant_df[component], bins=10, alpha=0.5, label=component)
            plt.title('Distribution of Average BANT Components Scores (Companies)')
            plt.xlabel('Score')
            plt.ylabel('Number of Companies')
            plt.legend()
            plt.tight_layout()
            graph_buffer3 = BytesIO()
            plt.savefig(graph_buffer3, format='png')
            plt.close()
            graph_buffer3.seek(0)
            img3 = openpyxl.drawing.image.Image(graph_buffer3)
            worksheet.add_image(img3, 'A40')
        
        report_buffer.seek(0)
        return report_buffer
    except Exception as e:
        logger.error(f"Error generating BANT report: {e}")
        return None
    
def summarize_paper(paper_title, paper_abstract):
    """
    Generates a summary for a given paper abstract using llm.
    """
    try:
        prompt = f"Provide a summary for the following paper. Together with a summary of the overall paper, be sure to inlude the methods used, what was the experiment, what technologies were used, what was the result, and what were the hypotheses as well as the relevant field. \n\nTitle: {paper_title}\n\nAbstract: {paper_abstract}\n\nSummary:"
        
        summary = llm_reasoning(prompt)
        return summary
    except Exception as e:
        logger.error(f"Error summarizing paper '{paper_title}': {e}")
        return "Summary not available."
    

def perform_bant_analysis(lead_info, context):
    """
    Performs BANT analysis on a single lead using llm.

    :param lead_info: Dictionary containing lead information.
    :param context: The context provided by the user.
    :return: Dictionary containing BANT analysis.
    """
    # Ensure `lead_info` keys have valid default values
    sanitized_lead_info = {}
    for key, value in lead_info.items():
        if isinstance(value, (int, float)):
            sanitized_lead_info[key] = value
        elif isinstance(value, str):
            sanitized_lead_info[key] = value.strip()  # Remove excess whitespace
        elif value is None or value == '':
            sanitized_lead_info[key] = "Not Available"  # Default for missing data
        else:
            sanitized_lead_info[key] = str(value)  # Convert other types to strings

    # Convert the sanitized dictionary to a JSON string
    sanitized_lead_info_json = json.dumps(sanitized_lead_info, indent=4)

    # Use the provided context or a fallback if none is provided
    context_text = context.strip() if context and context.strip() else "No specific context provided."

    # Construct the BANT prompt
    bant_prompt = f"""
You are an AI assistant specialized in sales lead qualification using the BANT framework.

**Context:**
{context_text}

**Lead Information:**
{sanitized_lead_info_json}

**Task:**
Analyze the lead information based on the BANT criteria (Budget, Authority, Need, Timeline). Provide the analysis in JSON format only, adhering strictly to the following structure without any additional text or explanations.

**Instructions:**
- **Budget:** Score between 0 and 1 indicating the lead's budget alignment.
- **Authority:** Score between 0 and 1 indicating the lead's decision-making authority.
- **Need:** Score between 0 and 1 indicating the lead's need for your product/service.
- **Timeline:** Score between 0 and 1 indicating the lead's readiness to engage.
- **Overall BANT Score:** Average of the four BANT component scores.
- **Recommendations:** Based on the scores, recommend whether to pursue or deprioritize the lead.

**Output Example:**
{{
    "Budget": {{"score": 0.8, "details": "Adequate budget available for our services."}},
    "Authority": {{"score": 0.7, "details": "Decision-maker identified within the organization."}},
    "Need": {{"score": 0.9, "details": "Clear need for our product based on current challenges."}},
    "Timeline": {{"score": 0.6, "details": "Interested in implementing within the next 6 months."}},
    "Overall BANT Score": 0.75,
    "Recommendations": "Pursue the lead with priority."
}}

**Please ensure:**
- The JSON is valid and properly formatted.
- Only the specified fields are included.
- No additional commentary, explanations, or markdown is present.
"""

    try:
        # Send the prompt to LLM and parse the response
        bant_analysis_text = llm_reasoning(bant_prompt)
        bant_analysis = json.loads(bant_analysis_text)
        return bant_analysis
    except json.JSONDecodeError:
        logger.error("Failed to parse llm response for BANT analysis.")
        logger.debug(f"Received Response: {bant_analysis_text}")
        return {
            "Budget": {"score": 0, "details": "Not Available"},
            "Authority": {"score": 0, "details": "Not Available"},
            "Need": {"score": 0, "details": "Not Available"},
            "Timeline": {"score": 0, "details": "Not Available"},
            "Overall BANT Score": 0,
            "Recommendations": "Not Available"
        }
    except Exception as e:
        logger.error(f"An error occurred during BANT analysis: {e}")
        return {
            "Budget": {"score": 0, "details": "Error occurred"},
            "Authority": {"score": 0, "details": "Error occurred"},
            "Need": {"score": 0, "details": "Error occurred"},
            "Timeline": {"score": 0, "details": "Error occurred"},
            "Overall BANT Score": 0,
            "Recommendations": "Error occurred"
        }

def rank_leads_with_bant(leads_info_df, person_leads_df, context):
    """
    Performs BANT analysis at the individual level and aggregates scores at the company level.
    
    :param leads_info_df: DataFrame containing company-level lead information.
    :param person_leads_df: DataFrame containing individual-level lead information.
    :param context: The context provided by the user.
    :return: Tuple containing updated person_leads_df and company_bant_df.
    """
    # Initialize lists to store aggregated company scores
    company_scores = {}
    
    # Iterate over each person in person_leads_df
    for idx, row in person_leads_df.iterrows():
        person_info = row.to_dict()
        company_name = person_info.get("Entity", "Unknown")
        
        # Perform BANT analysis for the individual
        bant_analysis = perform_bant_analysis(person_info, context)
        
        # Extract individual BANT scores
        budget_score = bant_analysis.get("Budget", {}).get("score", 0)
        authority_score = bant_analysis.get("Authority", {}).get("score", 0)
        need_score = bant_analysis.get("Need", {}).get("score", 0)
        timeline_score = bant_analysis.get("Timeline", {}).get("score", 0)
        overall_score = bant_analysis.get("Overall BANT Score", 0)
        recommendation = bant_analysis.get("Recommendations", "Not Available")
        
        # Append individual BANT scores to person_leads_df
        person_leads_df.at[idx, "Budget Score"] = budget_score
        person_leads_df.at[idx, "Authority Score"] = authority_score
        person_leads_df.at[idx, "Need Score"] = need_score
        person_leads_df.at[idx, "Timeline Score"] = timeline_score
        person_leads_df.at[idx, "Overall BANT Score"] = overall_score
        person_leads_df.at[idx, "Recommendations"] = recommendation
        
        # Aggregate scores per company
        if company_name not in company_scores:
            company_scores[company_name] = {
                "Budget Score": [],
                "Authority Score": [],
                "Need Score": [],
                "Timeline Score": [],
                "Overall BANT Score": []
            }
        
        company_scores[company_name]["Budget Score"].append(budget_score)
        company_scores[company_name]["Authority Score"].append(authority_score)
        company_scores[company_name]["Need Score"].append(need_score)
        company_scores[company_name]["Timeline Score"].append(timeline_score)
        company_scores[company_name]["Overall BANT Score"].append(overall_score)
    
    # Create company_bant_df by aggregating individual scores
    company_bant_data = []
    for company, scores in company_scores.items():
        avg_budget = sum(scores["Budget Score"]) / len(scores["Budget Score"])
        avg_authority = sum(scores["Authority Score"]) / len(scores["Authority Score"])
        avg_need = sum(scores["Need Score"]) / len(scores["Need Score"])
        avg_timeline = sum(scores["Timeline Score"]) / len(scores["Timeline Score"])
        avg_overall = sum(scores["Overall BANT Score"]) / len(scores["Overall BANT Score"])
        
        # Determine company-level recommendation based on average scores
        if avg_overall >= 0.75:
            company_recommendation = "Pursue the company with high priority."
        elif 0.5 <= avg_overall < 0.75:
            company_recommendation = "Monitor the company and pursue if opportunities arise."
        else:
            company_recommendation = "Deprioritize the company."
        
        company_bant_data.append({
            "Entity": company,
            "Average Budget Score": round(avg_budget, 2),
            "Average Authority Score": round(avg_authority, 2),
            "Average Need Score": round(avg_need, 2),
            "Average Timeline Score": round(avg_timeline, 2),
            "Overall BANT Score": round(avg_overall, 2),
            "Recommendations": company_recommendation
        })
    
    company_bant_df = pd.DataFrame(company_bant_data)
    
    # Rank companies based on Overall BANT Score
    company_bant_df = company_bant_df.sort_values(by="Overall BANT Score", ascending=False).reset_index(drop=True)
    
    return person_leads_df, company_bant_df