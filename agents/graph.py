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
from browser_use import Agent
from browser_use.browser.views import BrowserState
from browser_use.agent.views import AgentOutput
import logging
from playwright.async_api import async_playwright
import subprocess
from fastapi.responses import FileResponse
from tools.tools import BrowserTool

# Import LangGraph components
from langgraph_implementation import execute_task_with_graph, create_empty_state

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


class TaskResponse(BaseModel):
    result: str
    status: str
    screenshot_urls: List[str] = []


def create_llm():
    return ChatAnthropic(
        model="claude-3-7-sonnet-20250219",
        temperature=0.0,
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

        # Check if browsers are installed, install if needed
        if not await check_browsers():
            logger.info("Browsers not installed, attempting to install...")
            if not await install_browsers():
                raise HTTPException(
                    status_code=500, detail="Failed to install browsers"
                )

        # Execute task using LangGraph
        result = await execute_task_with_graph(request.task, request.context)

        # Extract information from result
        result_str = result.get("final_result", "No result available")
        screenshot_urls = result.get("screenshots", [])

        return TaskResponse(
            result=result_str,
            status="success" if not result.get("errors") else "error",
            screenshot_urls=screenshot_urls,
        )
    except Exception as e:
        logger.error(f"Error processing task: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


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


def base64_to_image(base64_string: str, output_filename: str):
    if not os.path.exists(os.path.dirname(output_filename)):
        os.makedirs(os.path.dirname(output_filename))
    img_data = base64.b64decode(base64_string)
    with open(output_filename, "wb") as f:
        f.write(img_data)
    return output_filename


@app.websocket("/ws/agent")
async def agent_ws(websocket: WebSocket):
    await websocket.accept()
    try:
        # Receive the task and context from the frontend
        data = await websocket.receive_json()
        task = data.get("task")
        context = data.get("context", {})

        # Execute task using LangGraph with websocket
        await execute_task_with_graph(task, context, websocket)
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception as e:
        logger.error(f"Error in WebSocket: {str(e)}")
        try:
            await websocket.send_json({"error": str(e)})
        except:
            pass


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
