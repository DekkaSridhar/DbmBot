# # Temporary fix for libraries using collections.Sequence
import collections.abc
import sys
sys.modules['collections.Sequence'] = collections.abc.Sequence

import logging
from flask import Flask, request, jsonify
from dataclasses import dataclass
from routes.chatengine import Tools
from flask_cors import CORS

# Initialize Flask app
app = Flask(__name__)

# Add CORS middleware
CORS(app, resources={
    r"/*": {
        "origins": "*",
        "allow_credentials": True,
        "methods": ["*"],
        "headers": ["*"]
    }
})

# Define the input and output models
@dataclass
class QueryRequest:
    query: str

@dataclass
class QueryResponse:
    response: str

# Initialize the Tools instance
tools = Tools()

# Logger setup
logger = logging.getLogger("run.py")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

@app.route("/start-chat", methods=['GET'])
def start_chat():
    """
    Endpoint to display the intro message when accessed
    """
    intro_message = """
    Hey there! I'm Debbie-M, the resident chatbot at the Philippines government.
    I'm happy to chat and assist you, but please note that my responses are powered by OpenAI.
    Let's work together to find the information you need. Feel free to ask me about documents, policies, or government procedures.
    """
    return jsonify({"message": intro_message})

@app.route("/query", methods=['POST'])
def query_endpoint():
    """
    Flask endpoint to handle chatbot queries.
    """
    try:
        request_data = request.get_json()
        query = request_data.get('query')
        
        if not query:
            return jsonify({"error": "Query field is required"}), 400
            
        logger.info(f"Received query: {query}")
        agent_response = tools.query_document_tool(query)
        
        # Extract the actual string response from AgentChatResponse
        response_text = agent_response.response
        return jsonify(QueryResponse(response=response_text).__dict__)
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500

@app.route("/", methods=['GET'])
def root():
    """
    Root endpoint to verify service is running.
    """
    return jsonify({"message": "Chatbot service is running. Use the /query endpoint to interact."})

# This is only used when running locally
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)