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

def generate_bant_report(ranked_leads_df):
    """
    Generates a BANT report for all leads, including graphs and llm analysis.

    :param ranked_leads_df: DataFrame containing ranked leads with BANT scores.
    :return: BytesIO object containing the report (e.g., PDF or Excel).
    """
    try:
        # For simplicity, we'll generate an Excel report with multiple sheets
        report_buffer = BytesIO()
        with pd.ExcelWriter(report_buffer, engine='openpyxl') as writer:
            # Sheet 1: Ranked Leads with BANT Scores
            ranked_leads_df.to_excel(writer, sheet_name='Ranked Leads BANT', index=False)
            
            # Sheet 2: Summary Statistics
            summary = ranked_leads_df.describe()
            summary.to_excel(writer, sheet_name='Summary Statistics')
            
            # Sheet 3: Recommendations
            recommendations = ranked_leads_df[['Company/Group Name', 'Recommendations']]
            recommendations.to_excel(writer, sheet_name='Recommendations', index=False)
            
            # Sheet 4: Graphs
            workbook = writer.book
            worksheet = workbook.create_sheet("Graphs")
            
            # Example Graph: Distribution of Overall BANT Scores
            plt.figure(figsize=(10, 6))
            plt.hist(ranked_leads_df['Overall BANT Score'], bins=10, color='salmon', edgecolor='black')
            plt.title('Distribution of Overall BANT Scores')
            plt.xlabel('Overall BANT Score')
            plt.ylabel('Number of Leads')
            plt.tight_layout()
            graph_buffer = BytesIO()
            plt.savefig(graph_buffer, format='png')
            plt.close()
            graph_buffer.seek(0)
            img = openpyxl.drawing.image.Image(graph_buffer)
            worksheet.add_image(img, 'A1')
            
            # Additional Graph: BANT Component Scores per Company
            plt.figure(figsize=(12, 8))
            bant_components = ['Budget Score', 'Authority Score', 'Need Score', 'Timeline Score']
            for component in bant_components:
                plt.hist(ranked_leads_df[component], bins=10, alpha=0.5, label=component)
            plt.title('Distribution of BANT Components Scores')
            plt.xlabel('Score')
            plt.ylabel('Number of Leads')
            plt.legend()
            plt.tight_layout()
            graph_buffer2 = BytesIO()
            plt.savefig(graph_buffer2, format='png')
            plt.close()
            graph_buffer2.seek(0)
            img2 = openpyxl.drawing.image.Image(graph_buffer2)
            worksheet.add_image(img2, 'A20')
        
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
    Performs BANT analysis on a single lead using llm-4.

    :param lead_info: Dictionary containing lead information.
    :param context: The context provided by the user.
    :return: Dictionary containing BANT analysis.
    """
    encoding = 'cl100k_base'

    # Use the context provided
    context_text = context.strip() if context.strip() else "No specific context provided."

    bant_prompt = f"""
You are an AI assistant specialized in sales lead qualification using the BANT framework.

**Context:**
{context}

**Lead Information:**
{json.dumps(lead_info, indent=4)}

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
[{
    "Budget": {{"score": 0.8, "details": "Adequate budget available for our services."}},
    "Authority": {{"score": 0.7, "details": "Decision-maker identified within the organization."}},
    "Need": {{"score": 0.9, "details": "Clear need for our product based on current challenges."}},
    "Timeline": {{"score": 0.6, "details": "Interested in implementing within the next 6 months."}},
    "Overall BANT Score": 0.75,
    "Recommendations": "Pursue the lead with priority."
}]

**Please ensure:**
- The JSON is valid and properly formatted.
- Only the specified fields are included.
- No additional commentary, explanations, or markdown is present.
"""


    try:
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

def rank_leads_with_bant(leads_info_df, context):
    """
    Ranks leads based on BANT analysis.

    :param leads_info_df: DataFrame containing lead information.
    :param context: The initial context provided by the user.
    :return: DataFrame with BANT scores and rankings.
    """
    bant_scores = []
    overall_scores = []
    recommendations = []

    for idx, row in leads_info_df.iterrows():
        lead_info = row.to_dict()
        bant_analysis = perform_bant_analysis(lead_info, context)
        
        # Extract individual BANT scores
        budget_score = bant_analysis.get("Budget", {}).get("score", 0)
        authority_score = bant_analysis.get("Authority", {}).get("score", 0)
        need_score = bant_analysis.get("Need", {}).get("score", 0)
        timeline_score = bant_analysis.get("Timeline", {}).get("score", 0)
        overall_score = bant_analysis.get("Overall BANT Score", 0)
        recommendation = bant_analysis.get("Recommendations", "Not Available")
        
        bant_scores.append({
            "Budget Score": budget_score,
            "Authority Score": authority_score,
            "Need Score": need_score,
            "Timeline Score": timeline_score
        })
        overall_scores.append(overall_score)
        recommendations.append(recommendation)
    
    # Create a DataFrame from BANT scores
    bant_df = pd.DataFrame(bant_scores)
    leads_info_df = leads_info_df.reset_index(drop=True).join(bant_df)
    leads_info_df["Overall BANT Score"] = overall_scores
    leads_info_df["Recommendations"] = recommendations
    
    # Rank leads based on Overall BANT Score
    leads_info_df = leads_info_df.sort_values(by="Overall BANT Score", ascending=False).reset_index(drop=True)
    
    return leads_info_df