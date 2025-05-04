import os
import uuid
import base64
import asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Dict, Any
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from browser_use import Agent
from browser_use.browser.views import BrowserState
from browser_use.agent.views import AgentOutput
import logging
from playwright.async_api import async_playwright
import subprocess
from fastapi.responses import FileResponse

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
    screenshot_urls: list[str]


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


async def take_and_save_screenshot(page, session_id, step_num, description=""):
    session_path = os.path.join(SCREENSHOT_DIR, session_id)
    os.makedirs(session_path, exist_ok=True)
    filename = f"step_{step_num}.jpg"
    filepath = os.path.join(session_path, filename)
    await page.screenshot(path=filepath, type="jpeg", quality=80)
    return f"/session_screenshots/{session_id}/{filename}"


async def run_agent_with_screenshots(agent, page, session_id):
    print("Running agent with screenshots")
    screenshots = []
    step_counter = {"count": 0}
    # Take initial screenshot (blank page)
    screenshots.append(
        await take_and_save_screenshot(
            page, session_id, step_counter["count"], "initial"
        )
    )
    step_counter["count"] += 1
    # If the agent has a URL to visit, go there and take a screenshot
    if hasattr(agent, "start_url") and agent.start_url:
        await page.goto(agent.start_url)
        screenshots.append(
            await take_and_save_screenshot(
                page, session_id, step_counter["count"], "after_goto"
            )
        )
        step_counter["count"] += 1
    # Run the agent and take a screenshot after
    result = await agent.run()
    screenshots.append(
        await take_and_save_screenshot(
            page, session_id, step_counter["count"], "after_agent_run"
        )
    )
    return result, screenshots


@app.post("/execute_task", response_model=TaskResponse)
async def execute_task(request: TaskRequest):
    try:
        logger.info(f"Received task request: {request.task}")
        if not await check_browsers():
            logger.info("Browsers not installed, attempting to install...")
            if not await install_browsers():
                raise HTTPException(
                    status_code=500, detail="Failed to install browsers"
                )
        session_id = str(uuid.uuid4())
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=False)
            context = await browser.new_context()
            page = await context.new_page()
            llm = create_llm()
            agent = Agent(task=request.task, llm=llm, context=request.context)
            result, screenshot_urls = await run_agent_with_screenshots(
                agent, page, session_id
            )
            result_str = str(result)
            await context.close()
            await browser.close()
        return TaskResponse(
            result=result_str, status="success", screenshot_urls=screenshot_urls
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
        session_id = str(uuid.uuid4())
        step_counter = 0
        llm = ChatAnthropic(model="claude-3-opus-20240229")
        screenshot_urls = []

        # Callback for each step
        def new_step_callback(
            state: BrowserState, model_output: AgentOutput, steps: int
        ):
            nonlocal step_counter
            path = f"{SCREENSHOT_DIR}/{session_id}/step_{step_counter}.png"
            last_screenshot = state.screenshot
            img_path = base64_to_image(str(last_screenshot), path)
            screenshot_url = (
                f"/session_screenshots/{session_id}/step_{step_counter}.png"
            )
            screenshot_urls.append(screenshot_url)
            # Extract memory information
            status = getattr(getattr(model_output, "current_state", None), "memory", "")
            step_counter += 1

            # Send the screenshot URL, base64, and step status to the frontend
            asyncio.create_task(
                websocket.send_json(
                    {
                        "screenshot_url": screenshot_url,
                        "screenshot_base64": str(last_screenshot),
                        "step": step_counter,
                        "status": status,
                    }
                )
            )

        # Run the agent
        agent = Agent(
            task=task,
            llm=llm,
            context=context,
            register_new_step_callback=new_step_callback,
        )
        result = await agent.run()
        # Send final result and all screenshot URLs
        await websocket.send_json(
            {
                "result": str(result),
                "status": "success",
                "screenshot_urls": screenshot_urls,
                "done": True,
            }
        )
    except WebSocketDisconnect:
        pass
    except Exception as e:
        await websocket.send_json({"error": str(e)})
        raise


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
