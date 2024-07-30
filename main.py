import os
import logging
import pandas as pd
import numpy as np
import openai
from openai import OpenAI
import asyncio
import nest_asyncio
import requests
import json
from flask import Flask, request, jsonify
from flask_cors import CORS
from google.cloud import secretmanager
from questions import answer_question
import pickle
import faiss
import sys

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Set up logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO,
    stream=sys.stdout
)
logging.info("Application starting...")

# Global variables
index = None
id_to_text = None

# Function to get the secret from Google Cloud Secret Manager
def get_secret(secret_name):
    client = secretmanager.SecretManagerServiceClient()
    name = f"projects/even-research-429408-s9/secrets/{secret_name}/versions/latest"
    response = client.access_secret_version(request={"name": name})
    return response.payload.data.decode("UTF-8")

# Load OpenAI API key
try:
    openai_api_key = get_secret('openai_api_key')
    os.environ['OPENAI_API_KEY'] = openai_api_key 
    client = OpenAI()  
except Exception as e:
    print(f"Error accessing secret: {str(e)}")
    raise

# Load FAISS index and id_to_text
try:
    index = faiss.read_index('/app/data/faiss_index.index')
    with open('/app/data/id_to_text.pkl', 'rb') as f:
        id_to_text = pickle.load(f)
    logging.info("FAISS index and id_to_text loaded successfully")
except Exception as e:
    logging.error(f"Failed to load FAISS index or id_to_text: {str(e)}")
    raise

examples = [
    {"role": "user", "content": "Who is Noah?"},
    {"role": "assistant", "content": "Noah is my creator! He is a Sophomore student athlete at Carleton College, currently pursuing a BA in Computer Science and Statistics, while hoping to break into the fields of Data Science and Software Engineering. If you are an employer reading this, please hire him!"},
   ]

messages = [{
    "role":"system",
    "content":"You are a helpful assistant on Noah's ePortfolio website that answers questions - especially about Noah and his experiences. Follow the examples provided and give appropriate responses. Try to limit responses to 80 words."
}]
messages.extend(examples)

logging.basicConfig(
    format='%(asctime)s - %(name)s - $(levelname)s - %(message)s',
    level=logging.INFO
)

def get_answer(question):
    return answer_question(question=question, debug=True)

def is_related_to_aix(message):
    return True

@app.route('/', methods=['GET'])
def hello():
    return jsonify({"message":"Hello world!"})

@app.route('/chat', methods=['POST'])
def chat():
    try:
        logging.info("Received chat request")
        incoming_msg = request.json.get('message')
        logging.info(f"Incoming message: {incoming_msg}")
        
        if not incoming_msg:
            logging.warning("No message provided")
            return jsonify({"error": "No message provided"}), 400
        
        if incoming_msg == 'GREETING':
            logging.info("Responding with greeting")
            return jsonify({"response": "Hello! I am AIX Bot, how can I assist you today?"}), 200
        
        if is_related_to_aix(incoming_msg):
            logging.info("Message is related to AIX, generating answer")
            retrieval_answer = get_answer(incoming_msg)
            logging.info(f"Retrieved answer: {retrieval_answer}")
            
            current_messages = messages.copy()
            current_messages.append({"role": "user", "content": incoming_msg})
            current_messages.append({"role": "system", "content": f"Context: {retrieval_answer}"})
            
            logging.info("Sending request to OpenAI")
            try:
                initial_response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=current_messages
                )
                
                logging.info("Received response from OpenAI")
                initial_response_message = initial_response.choices[0].message.content
                logging.info(f"OpenAI response: {initial_response_message}")
                
                return jsonify({"response": initial_response_message}), 200
            except openai.APIError as e:
                logging.error(f"OpenAI API error: {str(e)}")
                return jsonify({"error": "Error communicating with AI service"}), 503
            except Exception as e:
                logging.error(f"Unexpected error in OpenAI request: {str(e)}")
                return jsonify({"error": "An unexpected error occurred"}), 500
        else:
            logging.warning("Message not related to AIX")
            return jsonify({"error": "Message not related to AIX"}), 400
    
    except Exception as e:
        logging.error(f"An error occurred in chat function: {str(e)}")
        return jsonify({"error": "An internal server error occurred"}), 500
if __name__ == '__main__':
    nest_asyncio.apply()
    port = int(os.environ.get('PORT', 8080))
    logging.info(f"Starting Flask app on port {port}")
    app.run(host='0.0.0.0', port=port)
