import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from routes.chatengine import Tools

# Initialize FastAPI app
app = FastAPI()
 
# Define the input model for the query
class QueryRequest(BaseModel):
    query: str
 
# Define the output model for the response
class QueryResponse(BaseModel):
    response: str
 
# Initialize the Tools instance
tools = Tools()
 
# Logger setup
logger = logging.getLogger("run.py")
logging.basicConfig(level=logging.INFO)

@app.get("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest):
    """
    FastAPI endpoint to handle chatbot queries.
    """
    try:
        logger.info(f"Received query: {request.query}")
        agent_response = tools.query_document_tool(request.query)
        
        # Extract the actual string response from AgentChatResponse
        response_text = agent_response.response  # Adjust this based on your actual data structure
        return QueryResponse(response=response_text)
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

# Example root endpoint
@app.get("/")
async def root():
    """
    Root endpoint to verify service is running.
    """
    return {"message": "Chatbot service is running. Use the /query endpoint to interact."}




# run this in cmd => uvicorn run:app --host 0.0.0.0 --port 8000 --workers 4