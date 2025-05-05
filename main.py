import os
import uuid
import base64
import asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
import logging
from playwright.async_api import async_playwright
import subprocess
from fastapi.responses import FileResponse
from master_agent import MasterAgent  # Import the new MasterAgent class

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Directory for session screenshots
SCREENSHOT_DIR = "session_screenshots"
os.makedirs(SCREENSHOT_DIR, exist_ok=True)

# Initialize FastAPI app
app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Define models
class TaskRequest(BaseModel):
    task: str
    context: Dict[str, Any] = {}


class SubtaskInfo(BaseModel):
    task_id: str
    agent_type: str
    task: str
    status: str


class TaskResponse(BaseModel):
    result: str
    status: str
    subtasks: List[SubtaskInfo] = []
    screenshot_urls: Dict[str, List[str]] = {}


def create_llm():
    return ChatAnthropic(
        model_name="claude-3-5-sonnet-20240620",
        temperature=0.0,
        timeout=100,
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
    )


async def check_browsers():
    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch()
            await browser.close()
        return True
    except Exception as e:
        logger.error(f"Browser check failed: {str(e)}")
        return False


async def install_browsers():
    try:
        logger.info("Installing Playwright browsers...")
        subprocess.run(["playwright", "install", "chromium"], check=True)
        logger.info("Browsers installed successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to install browsers: {str(e)}")
        return False


@app.post("/execute_task", response_model=TaskResponse)
async def execute_task(request: TaskRequest):
    try:
        logger.info(f"Received task request: {request.task}")
        
        # Check if browsers are installed
        if not await check_browsers():
            logger.info("Browsers not installed, attempting to install...")
            if not await install_browsers():
                raise HTTPException(
                    status_code=500, detail="Failed to install browsers"
                )
        
        # Create a session ID
        session_id = str(uuid.uuid4())
        
        # Create the master agent
        llm = create_llm()
        master_agent = MasterAgent(task=request.task, llm=llm, context=request.context)
        
        # Run the master agent (this handles planning, execution, and synthesis)
        result = await master_agent.run()
        
        # Prepare the response
        subtasks_info = [
            SubtaskInfo(
                task_id=task.task_id,
                agent_type=task.agent_type,
                task=task.task,
                status=task.status
            )
            for task in master_agent.subtasks
        ]
        
        # Collect screenshot URLs by task
        screenshot_urls = {}
        for task in master_agent.subtasks:
            if task.screenshots:
                screenshot_urls[task.task_id] = [
                    screenshot.get("url") for screenshot in task.screenshots
                    if screenshot.get("url")
                ]
        
        return TaskResponse(
            result=result,
            status="success",
            subtasks=subtasks_info,
            screenshot_urls=screenshot_urls
        )
        
    except Exception as e:
        logger.error(f"Error processing task: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.websocket("/ws/master_agent")
async def master_agent_ws(websocket: WebSocket):
    await websocket.accept()
    try:
        # Receive the task and context from the frontend
        data = await websocket.receive_json()
        task = data.get("task")
        context = data.get("context", {})
        
        # Create LLM
        llm = create_llm()
        
        # Create the master agent with the WebSocket
        master_agent = MasterAgent(
            task=task,
            llm=llm,
            context=context,
            websocket=websocket
        )
        
        # Send initial status
        await websocket.send_json({
            "type": "status_update",
            "status": "initialized",
            "message": "Master agent initialized"
        })
        
        # Run the master agent
        try:
            result = await master_agent.run()
            
            # Send final result
            await websocket.send_json({
                "type": "final_result",
                "result": result,
                "status": "success",
                "done": True
            })
            
        except Exception as e:
            logger.error(f"Error in master agent execution: {str(e)}")
            await websocket.send_json({
                "type": "error",
                "error": str(e),
                "status": "error",
                "done": True
            })
            
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
        await websocket.send_json({"type": "error", "error": str(e)})
        raise


# Serve screenshots statically
app.mount(
    "/session_screenshots",
    StaticFiles(directory=SCREENSHOT_DIR),
    name="session_screenshots",
)
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def read_root():
    return FileResponse("static/index.html")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)