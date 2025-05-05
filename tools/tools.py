from logging import Logger
import os
import uuid
from typing import Dict, Any, List, Optional, Union
from pydantic import BaseModel, Field
from langchain.tools import BaseTool
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
from browser_use import Agent
from langchain_anthropic import ChatAnthropic
from browser_use.browser.views import BrowserState
from browser_use.agent.views import AgentOutput
from playwright.async_api import async_playwright
from dotenv import load_dotenv
import anthropic
import json
import asyncio
import base64
from fastapi import WebSocket, WebSocketDisconnect
from agents.bedrock_claude import BedrockClaudeClient

load_dotenv()

SCREENSHOT_DIR = "session_screenshots"
os.makedirs(SCREENSHOT_DIR, exist_ok=True)


class SecureClaudeTool(BaseTool):
    name: str = "secure_claude"
    description: str = "Use Claude via AWS Bedrock for secure mode."

    def __init__(self):
        self.client = BedrockClaudeClient()

    async def run(self, task: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        prompt = f"""
        You are a helpful AI assistant that provides accurate and concise answers.\n\nContext information: {json.dumps(context) if context else '{}'}\n\nUser task: {task}\n\nPlease provide a helpful response based on the task and context provided.
        """
        # Bedrock is sync; run in thread executor for async
        import asyncio

        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, lambda: self.client.chat(prompt))
        return {
            "result": result,
            "screenshot_urls": [],
            "tool_used": "secure_claude",
        }


def base64_to_image(base64_string: str, output_filename: str):
    if not os.path.exists(os.path.dirname(output_filename)):
        os.makedirs(os.path.dirname(output_filename))
    img_data = base64.b64decode(base64_string)
    with open(output_filename, "wb") as f:
        f.write(img_data)
    return output_filename


class BrowserTool(BaseTool):
    name: str = "browser_agent"
    description: str = (
        "Use a browser to interact with websites and accomplish tasks that require web navigation"
    )

    async def _arun(
        self, task: str, context: Dict[str, Any] = None, websocket: WebSocket = None
    ) -> Dict[str, Any]:
        await websocket.accept()
        try:
            # Receive the task and context from the frontend
            data = await websocket.receive_json()
            task = data.get("task")
            context = data.get("context", {})
            session_id = str(uuid.uuid4())
            step_counter = 0
            # Use secure Claude if context['secure_mode'] is True
            if context and context.get("secure_mode"):
                secure_tool = SecureClaudeTool()

                class DummyLLM:
                    async def __call__(self, *args, **kwargs):
                        return await secure_tool.run(task, context)

                llm = DummyLLM()
            else:
                llm = ChatAnthropic(model="claude-3-7-sonnet-20250219")
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
                status = getattr(
                    getattr(model_output, "current_state", None), "memory", ""
                )
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

    def _run(self, task: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        raise NotImplementedError("BrowserTool is async only")


class DirectLLMTool:
    """Tool for handling tasks using direct Claude API calls without browser automation."""

    name = "direct_llm"
    description = "Use the LLM directly to answer questions without browser interaction"

    async def run(self, task: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        # Create Anthropic client
        client = anthropic.AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

        # Prepare the prompt
        prompt = f"""
        You are a helpful AI assistant that provides accurate and concise answers.
        
        Context information: {json.dumps(context) if context else "{}"}
        
        User task: {task}
        
        Please provide a helpful response based on the task and context provided.
        """

        # Make direct Claude API call
        response = await client.messages.create(
            model="claude-3-7-sonnet-20250219",
            max_tokens=4096,
            temperature=0,
            system="You are a helpful AI assistant that provides accurate and concise answers.",
            messages=[
                {
                    "role": "user",
                    "content": f"{prompt}",
                }
            ],
        )

        return {
            "result": response.content[0].text,
            "screenshot_urls": [],
            "tool_used": "direct_llm",
        }
