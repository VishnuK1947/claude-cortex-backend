from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any
import os
from dotenv import load_dotenv
from anthropic import Anthropic
from langgraph.graph import Graph
from langgraph.prebuilt import ToolExecutor
from browser_use import BrowserAgent

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

# Initialize browser agent
browser_agent = BrowserAgent()

# Define agent tools
def search_web(query: str) -> str:
    """Search the web for information"""
    return browser_agent.search(query)

def browse_website(url: str) -> str:
    """Browse a specific website"""
    return browser_agent.visit(url)

# Create tool executor
tools = [
    ToolExecutor(search_web),
    ToolExecutor(browse_website)
]

# Define agent nodes
def master_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    """Master agent that coordinates other agents"""
    task = state["task"]
    context = state.get("context", {})
    
    # Use Claude to determine the best course of action
    response = claude.messages.create(
        model="claude-3-opus-20240229",
        max_tokens=1000,
        messages=[
            {
                "role": "user",
                "content": f"Task: {task}\nContext: {context}\n\nDetermine the best tools to use and provide a plan."
            }
        ]
    )
    
    # Extract plan from Claude's response
    plan = response.content[0].text
    
    # Update state with plan
    state["plan"] = plan
    return state

def execute_plan(state: Dict[str, Any]) -> Dict[str, Any]:
    """Execute the plan using available tools"""
    plan = state["plan"]
    
    # Use Claude to determine which tools to use
    response = claude.messages.create(
        model="claude-3-opus-20240229",
        max_tokens=1000,
        messages=[
            {
                "role": "user",
                "content": f"Plan: {plan}\n\nSelect and use the appropriate tools to execute this plan."
            }
        ]
    )
    
    # Execute tools based on Claude's instructions
    result = response.content[0].text
    state["result"] = result
    return state

# Create the graph
workflow = Graph()

# Add nodes
workflow.add_node("master_agent", master_agent)
workflow.add_node("execute_plan", execute_plan)

# Add edges
workflow.add_edge("master_agent", "execute_plan")

# Set entry point
workflow.set_entry_point("master_agent")

# Compile the graph
app = workflow.compile()

# API endpoints
@app.post("/execute_task", response_model=TaskResponse)
async def execute_task(request: TaskRequest):
    try:
        # Execute the workflow
        result = app.invoke({
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
    uvicorn.run(app, host="0.0.0.0", port=8000) 