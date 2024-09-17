from flask import Flask, request, render_template
import os
import requests
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import json
from statistics import mean
import re
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from flask import Flask, redirect, render_template
app = Flask(__name__)

# Load environment variables from a .env file (make sure to create one with your API keys)
load_dotenv()

# Initialize the SentenceTransformer model for plagiarism detection
plagiarism_model = SentenceTransformer('average_word_embeddings_komninos')

# Set up Google Custom Search API credentials
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
SEARCH_ENGINE_ID = os.getenv("SEARCH_ENGINE_ID")

# Initialize the Hugging Face model for AI content detection
tokenizer = AutoTokenizer.from_pretrained("openai-community/roberta-base-openai-detector")
ai_model = AutoModelForSequenceClassification.from_pretrained("openai-community/roberta-base-openai-detector")

def detect_ai_written_text(text):
    """
    Detects if the input text is AI-generated using a pre-trained Hugging Face model.

    Args:
    - text (str): The input text to analyze.

    Returns:
    - prediction (int): 1 if AI-generated, 0 if human-written.
    - ai_prob (float): The probability that the text is AI-generated.
    """
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = ai_model(**inputs)
    logits = outputs.logits
    probabilities = torch.softmax(logits, dim=1).tolist()[0]

    ai_prob = probabilities[1]  # Probability for AI-generated class
    prediction = 1 if ai_prob > 0.5 else 0  # Threshold of 0.5

    return prediction, ai_prob

def google_search(query):
    """
    Perform a Google search for the given query using the Custom Search API.
    """
    search_url = "https://www.googleapis.com/customsearch/v1"
    params = {
        "key": GOOGLE_API_KEY,
        "cx": SEARCH_ENGINE_ID,
        "q": query,
        "num": 3  # Retrieve top 3 search results
    }
    response = requests.get(search_url, params=params)
    search_results = response.json()
    return search_results.get("items", [])

def extract_content_from_results(results):
    """
    Extract URLs, titles, and snippets from the search results.
    """
    online_contents = []
    for result in results:
        url = result.get('link')
        snippet = result.get('snippet', '')
        title = result.get('title', '')
        online_contents.append({
            "url": url,
            "title": title,
            "content": snippet
        })
    return online_contents

def check_plagiarism(assignment):
    """
    Check the assignment for plagiarism by comparing it with online content.
    """
    # Split the assignment into sentences
    assignment_sentences = re.split(r'(?<=[.!?]) +', assignment)
    plagiarism_results = []

    for sentence in assignment_sentences:
        # Skip empty sentences
        if not sentence.strip():
            continue

        # Search online for the sentence
        search_results = google_search(sentence)
        online_contents = extract_content_from_results(search_results)

        # Encode the sentence
        sentence_vector = plagiarism_model.encode([sentence])

        for content in online_contents:
            online_text = content['content']
            online_vector = plagiarism_model.encode([online_text])

            # Compute cosine similarity
            similarity = cosine_similarity(sentence_vector, online_vector)[0][0]

            # Convert similarity to a standard Python float
            similarity = float(similarity)

            # If similarity exceeds threshold, record the result
            if similarity > 0.9:  # Threshold for considering plagiarism
                plagiarism_results.append({
                    "sentence": sentence,
                    "matched_content": online_text,
                    "matched_url": content['url'],
                    "matched_title": content['title'],
                    "similarity_score": similarity
                })

    return plagiarism_results

def get_plagiarism_score(similarity_results):
    """
    Calculate the overall plagiarism score based on similarity results.
    """
    if not similarity_results:
        return 0
    similarity_scores = [result['similarity_score'] for result in similarity_results]
    return mean(similarity_scores) * 100  # Convert to percentage

@app.route("/")
def index():
    """
    Render the homepage.
    """
    return render_template("index.html")

@app.route("/api/check_plagiarism", methods=["POST"])
def check_plagiarism_route():
    """
    API endpoint to check plagiarism for the submitted content.
    """
    original_content = request.form.get("originalContent", "")
    similarity_results = check_plagiarism(original_content)
    plagiarism_score = get_plagiarism_score(similarity_results)

    response_data = {
        "plagiarism_score": plagiarism_score,
        "matches": similarity_results
    }
    return json.dumps(response_data, indent=2)

@app.route("/api/check_ai_content", methods=["POST"])
def check_ai_content_route():
    """
    API endpoint to check if the submitted content is AI-written.
    """
    content = request.form.get("content", "")
    prediction, ai_prob = detect_ai_written_text(content)

    response_data = {
        "ai_written": prediction == 1,
        "probability": ai_prob
    }
    return json.dumps(response_data, indent=2)
@app.route("/open_pdf_app")
def open_pdf_app():
    return redirect("http://127.0.0.1:5001/")

if __name__ == "__main__":
    app.run(debug=True)
