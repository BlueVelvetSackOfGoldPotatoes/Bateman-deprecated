import os
from openai import OpenAI
from dotenv import load_dotenv
import re
import logging
from transformers import pipeline

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

load_dotenv()
client = OpenAI(api_key=os.environ.get("OPENAI_KEY"))

def llm_pipe(api_messages, temperature, model):
    origin = model.split(':')[0]
    specific_model = model.split(':')[1]
    
    text_prompt = "\n\n".join([message["role"] + ":\n" + message["content"] for message in api_messages])
    
    # OpenAI
    if origin == "oa":
        completion = client.chat.completions.create(
            model=specific_model,
            messages=api_messages,
            temperature=temperature
        )

        response = completion.choices[0].message.content

    # Hugging Face
    elif origin == "hf":
        pipe = pipeline("text-generation", model=specific_model)
        result = pipe(text_prompt)
        full_text = result[0]['generated_text']
        
        response = full_text[len(text_prompt):]
        
    elif origin == "ans":
        response = specific_model
    
    # Mistral
    # elif origin == "mi":
    #     model = specific_model
    #     chat_response = client.chat(
    #         model=model,
    #         messages=[ChatMessage(role="user", content=api_messages)]
    #     )

    #     response = chat_response.choices[0].message.content

    # Google Cloud
    # elif origin == "gc":
    #     model = GenerativeModel(model_name="gemini-1.0-pro-002")

    #     model_response = model.generate_content(text_prompt)

    #     response = model_response.text

    else:
        raise ValueError(f"Unknown origin: {origin}")

    response = response.strip()

    return response

def clean_response(response_text):
    """
    Cleans the GPT response to extract valid JSON.
    """
    response_text = response_text.strip()
    response_text = re.sub(r"^'''json", "", response_text, flags=re.IGNORECASE)
    response_text = re.sub(r"'''$", "", response_text, flags=re.IGNORECASE)
    if response_text.lower().startswith('json'):
        response_text = response_text[4:].strip()
    match = re.search(r'(\[.*\]|\{.*\})', response_text, re.DOTALL)
    return match.group(1) if match else response_text

def llm_reasoning(prompt, model="gpt-4o", t=0.2):
    """
    Makes a call to the OpenAI ChatCompletion API.

    :param prompt: The prompt string to send to OpenAI.
    :param model: The OpenAI model to use.
    :param temperature: Sampling temperature.
    :param max_tokens: Maximum number of tokens in the response.
    :return: The OpenAI API response object, or None if an error occurs.
    """
    try:
        response = client.chat.completions.create(
            model= "gpt-4o-mini", # should be model, but for test purposes...
            messages=[{"role": "user", "content": prompt}],
            temperature=t
        )

        processed_response = clean_response(response.choices[0].message.content)
        return processed_response
    except Exception as e:
        logger.error(f"OpenAI API call failed: {e}")
        return None