import os
import uuid
from typing import Dict, Any, List, Optional, Union
from pydantic import BaseModel, Field
from langchain.tools import BaseTool
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
from browser_use import Agent as BrowserAgent
from langchain_anthropic import ChatAnthropic
from browser_use.browser.views import BrowserState
from browser_use.agent.views import AgentOutput
from playwright.async_api import async_playwright
from dotenv import load_dotenv
import anthropic
import json

load_dotenv()

SCREENSHOT_DIR = "session_screenshots"
os.makedirs(SCREENSHOT_DIR, exist_ok=True)


# Define the tool for browser use
class BrowserTool(BaseTool):
    name: str = "browser_agent"
    description: str = (
        "Use a browser to interact with websites and accomplish tasks that require web navigation"
    )

    async def _arun(self, task: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        session_id = str(uuid.uuid4())
        screenshots = []
        step_counter = 0

        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=False)
            browser_context = await browser.new_context()
            page = await browser_context.new_page()

            # Create LLM
            llm = ChatAnthropic(
                model_name="claude-3-5-sonnet-20240620",
                temperature=0.0,
                timeout=100,
                anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
            )

            # Define screenshot callback
            async def screenshot_callback(
                state: BrowserState, model_output: AgentOutput, steps: int
            ):
                nonlocal step_counter
                # Take screenshot
                session_path = os.path.join(SCREENSHOT_DIR, session_id)
                os.makedirs(session_path, exist_ok=True)
                filename = f"step_{step_counter}.jpg"
                filepath = os.path.join(session_path, filename)
                await page.screenshot(path=filepath, type="jpeg", quality=80)
                screenshot_url = f"/session_screenshots/{session_id}/{filename}"
                screenshots.append(screenshot_url)
                step_counter += 1

            # Create and run agent
            agent = BrowserAgent(
                task=task,
                llm=llm,
                context=context or {},
                register_new_step_callback=screenshot_callback,
            )

            # If the agent has a start URL, go there
            if hasattr(agent, "start_url") and agent.start_url:
                await page.goto(agent.start_url)
                # Take screenshot after navigation
                session_path = os.path.join(SCREENSHOT_DIR, session_id)
                os.makedirs(session_path, exist_ok=True)
                filename = f"step_{step_counter}.jpg"
                filepath = os.path.join(session_path, filename)
                await page.screenshot(path=filepath, type="jpeg", quality=80)
                screenshots.append(f"/session_screenshots/{session_id}/{filename}")
                step_counter += 1

            # Run the agent
            result = await agent.run()

            # Take final screenshot
            session_path = os.path.join(SCREENSHOT_DIR, session_id)
            os.makedirs(session_path, exist_ok=True)
            filename = f"step_{step_counter}.jpg"
            filepath = os.path.join(session_path, filename)
            await page.screenshot(path=filepath, type="jpeg", quality=80)
            screenshots.append(f"/session_screenshots/{session_id}/{filename}")

            # Close browser
            await browser_context.close()
            await browser.close()

            return {"result": str(result), "screenshot_urls": screenshots}

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
            model="claude-3-5-sonnet-20240620",
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
