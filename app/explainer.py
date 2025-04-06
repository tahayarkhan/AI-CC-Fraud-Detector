import cohere
import os
from dotenv import load_dotenv

load_dotenv() 

co = cohere.Client(os.getenv("COHERE_API_KEY"))

def explain_prediction(features, prediction):
    prompt = f"""
    A financial fraud detection model predicted the following transaction as {'fraudulent' if prediction == 1 else 'legitimate'}.
    Transaction details:
    {features.to_dict()}
    
     Briefly explain why the model might have predicted this way in a few short bullet points.
    """ 

    response = co.generate(
        model="command-r-plus-08-2024",  
        prompt=prompt,
        max_tokens=8000,
        temperature=2
    )

    return response.generations[0].text.strip()
