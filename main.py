from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any
import os
from dotenv import load_dotenv
from anthropic import Anthropic
from langgraph.graph import Graph
from browser_use import Agent
import asyncio

# Load environment variables
load_dotenv()

# Initialize Claude client
claude = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# Initialize FastAPI app
app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define models
class TaskRequest(BaseModel):
    task: str
    context: Dict[str, Any] = {}

class TaskResponse(BaseModel):
    result: str
    status: str

# Create a Claude wrapper for browser-use
class ClaudeLLM:
    def __init__(self, model="claude-3-7-sonnet-20250219"):
        self.model = model
        self.client = claude

    async def __call__(self, prompt: str) -> str:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )
        return response.content[0].text

# Initialize Claude LLM
claude_llm = ClaudeLLM()

# Define agent nodes
async def master_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    """Master agent that coordinates other agents"""
    task = state["task"]
    context = state.get("context", {})
    
    # Create browser agent with the task
    agent = Agent(
        task=task,
        llm=claude_llm,
        context=context
    )
    
    # Run the agent
    result = await agent.run()
    
    # Update state with result
    state["result"] = result
    return state

# Create the graph
workflow = Graph()

# Add nodes
workflow.add_node("master_agent", master_agent)

# Set entry point
workflow.set_entry_point("master_agent")

# Compile the graph
agent_workflow = workflow.compile()

# API endpoints
@app.post("/execute_task", response_model=TaskResponse)
async def execute_task(request: TaskRequest):
    try:
        # Execute the workflow
        result = await agent_workflow.ainvoke({
            "task": request.task,
            "context": request.context
        })
        
        return TaskResponse(
            result=result["result"],
            status="success"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True) 