from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, TypedDict
import os
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from browser_use import Agent
from langgraph.graph import StateGraph
import asyncio

# Load environment variables
load_dotenv()

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

# Define state type for LangGraph
class AgentState(TypedDict):
    task: str
    context: Dict[str, Any]
    result: str

# Initialize the LLM for browser_use
def create_llm():
    return ChatAnthropic(
        model_name="claude-3-5-sonnet-20240620",
        temperature=0.0,
        timeout=100,  # Increase for complex tasks
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY")
    )

# Define agent nodes
async def master_agent(state: AgentState) -> AgentState:
    """Master agent that coordinates browser agent"""
    # Create a new state to avoid modifying the input state
    new_state = state.copy()
    
    # Extract task and context
    task = new_state["task"]
    context = new_state.get("context", {})
    
    # Create browser agent with the task
    # Creating it here ensures a fresh instance for each request
    llm = create_llm()
    agent = Agent(
        task=task,
        llm=llm,
        context=context
    )
    
    # Run the agent
    result = await agent.run()
    
    # Convert AgentHistoryList to string
    result_str = str(result)
    
    # Update state with result
    new_state["result"] = result_str
    return new_state

# Create the graph
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("master_agent", master_agent)

# Set entry point and end states
workflow.set_entry_point("master_agent")
workflow.add_edge("master_agent", "end")  # Use "end" instead of None
workflow.add_node("end", lambda x: x)  # Add end node that just returns the state

# Compile the graph
agent_workflow = workflow.compile()

# API endpoints
@app.post("/execute_task", response_model=TaskResponse)
async def execute_task(request: TaskRequest):
    try:
        # Execute the workflow
        initial_state: AgentState = {
            "task": request.task,
            "context": request.context,
            "result": ""  # Initialize with empty result
        }
        
        # Use ainvoke for async execution
        result = await agent_workflow.ainvoke(initial_state)
        
        return TaskResponse(
            result=result["result"],
            status="success"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)