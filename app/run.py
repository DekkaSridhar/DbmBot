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
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

@app.get("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest):
    """
    FastAPI endpoint to handle chatbot queries.
    """
    try:
        logger.info(f"Received query: {request.query}")
        agent_response = tools.query_document_tool(request.query)
        
        # Extract the actual string response from AgentChatResponse
        response_text = agent_response.response
        return QueryResponse(response=response_text)
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/")
async def root():
    """
    Root endpoint to verify service is running.
    """
    return {"message": "Chatbot service is running. Use the /query endpoint to interact."}

# This is only used when running locally
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
