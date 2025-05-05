import os
import asyncio
import uuid
import json
from typing import List, Dict, Any, Callable, Optional
from fastapi import WebSocket
from pydantic import BaseModel
import logging
from langchain_anthropic import ChatAnthropic
import anthropic

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AgentType:
    BROWSER = "browser"
    DIRECT_LLM = "direct_llm"

class AgentTask(BaseModel):
    task_id: str
    agent_type: str
    task: str
    context: Dict[str, Any] = {}
    status: str = "pending"
    result: Optional[str] = None
    screenshots: List[Dict[str, Any]] = []
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    error: Optional[str] = None

class MasterAgent:
    """
    Master agent that plans and orchestrates multiple sub-agents concurrently.
    """
    def __init__(
        self, 
        task: str, 
        llm: ChatAnthropic,
        context: Dict[str, Any] = None,
        websocket: Optional[WebSocket] = None
    ):
        self.task = task
        self.llm = llm
        self.context = context or {}
        self.websocket = websocket
        self.session_id = str(uuid.uuid4())
        self.subtasks: List[AgentTask] = []
        
    async def plan_tasks(self) -> List[AgentTask]:
        """
        Use the LLM to analyze the task and break it down into appropriate subtasks.
        """
        logger.info(f"Planning tasks for: {self.task}")
        
        # Prompt the LLM to break down the task
        prompt = f"""
        You are a task planning assistant. Your job is to break down a complex task into smaller subtasks
        that can be executed concurrently. Each subtask should specify if it requires browser automation
        or can be handled directly with an LLM.
        
        Task: {self.task}
        
        Please break this down into 2-5 subtasks. For each subtask, specify:
        1. A clear task description
        2. Whether it requires browser automation ("browser") or can be handled directly by an LLM ("direct_llm")
        
        Format your response as a valid JSON array of objects with the following structure:
        [
            {{
                "agent_type": "browser or direct_llm",
                "task": "detailed task description"
            }}
        ]
        
        Only return the JSON array, nothing else.
        """
        
        try:
            response = await self.llm.ainvoke(prompt)
            task_list = json.loads(response.content[0].text)
            
            # Create AgentTask objects from the planned tasks
            subtasks = []
            for i, task_info in enumerate(task_list):
                task_id = f"{self.session_id}-{i}"
                subtasks.append(AgentTask(
                    task_id=task_id,
                    agent_type=task_info["agent_type"],
                    task=task_info["task"],
                    context=self.context.copy()
                ))
            
            self.subtasks = subtasks
            return subtasks
            
        except Exception as e:
            logger.error(f"Failed to plan tasks: {str(e)}")
            # Fallback: create a single task using the original prompt
            task_id = f"{self.session_id}-0"
            fallback_task = AgentTask(
                task_id=task_id,
                agent_type=AgentType.BROWSER,
                task=self.task,
                context=self.context.copy()
            )
            self.subtasks = [fallback_task]
            return self.subtasks
    
    async def send_update(self, update_type: str, data: Dict[str, Any]):
        """Send an update via WebSocket if available."""
        if self.websocket:
            await self.websocket.send_json({
                "type": update_type,
                **data
            })
    
    async def run_browser_agent(self, task: AgentTask):
        """Execute a browser agent task."""
        from browser_use import Agent
        from playwright.async_api import async_playwright
        
        logger.info(f"Running browser agent for task: {task.task_id}")
        task.status = "running"
        task.started_at = asyncio.get_event_loop().time()
        
        await self.send_update("task_update", {
            "task_id": task.task_id,
            "agent_type": task.agent_type,
            "status": task.status,
            "task": task.task
        })
        
        # Track screenshot steps
        step_counter = 0
        
        async def screenshot_callback(page, browser_state, model_output, steps):
            nonlocal step_counter
            step_counter += 1
            
            screenshot_path = os.path.join("session_screenshots", self.session_id, f"{task.task_id}-step-{step_counter}.jpg")
            os.makedirs(os.path.dirname(screenshot_path), exist_ok=True)
            
            try:
                await page.screenshot(path=screenshot_path, type="jpeg", quality=80)
                screenshot_url = f"/session_screenshots/{self.session_id}/{task.task_id}-step-{step_counter}.jpg"
                
                # Get base64 representation of the screenshot
                screenshot_bytes = await page.screenshot(type="jpeg", quality=80)
                import base64
                screenshot_base64 = base64.b64encode(screenshot_bytes).decode('utf-8')
                
                # Create screenshot entry
                screenshot_entry = {
                    "step": step_counter,
                    "url": screenshot_url,
                    "base64": screenshot_base64,
                    "description": getattr(getattr(model_output, "current_state", None), "memory", ""),
                    "timestamp": asyncio.get_event_loop().time()
                }
                
                task.screenshots.append(screenshot_entry)
                
                # Send update via WebSocket
                await self.send_update("screenshot_update", {
                    "task_id": task.task_id,
                    "screenshot": screenshot_entry
                })
                
            except Exception as e:
                logger.error(f"Error capturing screenshot: {str(e)}")
        
        try:
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True)
                context = await browser.new_context()
                page = await context.new_page()
                
                # Create the browser agent
                browser_agent = Agent(
                    task=task.task,
                    llm=self.llm,
                    context=task.context,
                    register_new_step_callback=screenshot_callback
                )
                
                # Run the browser agent
                result = await browser_agent.run()
                
                # Update task with result
                task.result = str(result)
                task.status = "completed"
                task.completed_at = asyncio.get_event_loop().time()
                
                await self.send_update("task_update", {
                    "task_id": task.task_id,
                    "status": task.status,
                    "result": task.result
                })
                
                return task
                
        except Exception as e:
            error_msg = f"Browser agent error: {str(e)}"
            logger.error(error_msg)
            task.status = "error"
            task.error = error_msg
            task.completed_at = asyncio.get_event_loop().time()
            
            await self.send_update("task_update", {
                "task_id": task.task_id,
                "status": task.status,
                "error": task.error
            })
            
            return task
    
    async def run_direct_llm_agent(self, task: AgentTask):
        """Execute a direct LLM task without browser automation."""
        logger.info(f"Running direct LLM agent for task: {task.task_id}")
        task.status = "running"
        task.started_at = asyncio.get_event_loop().time()
        
        await self.send_update("task_update", {
            "task_id": task.task_id,
            "agent_type": task.agent_type,
            "status": task.status,
            "task": task.task
        })
        
        # Create a tracking step for UI visualization
        step_entry = {
            "step": 1,
            "url": None,
            "base64": None,
            "description": "Processing with Claude",
            "timestamp": asyncio.get_event_loop().time()
        }
        
        task.screenshots.append(step_entry)
        
        await self.send_update("step_update", {
            "task_id": task.task_id,
            "step": step_entry
        })
        
        try:
            # Create Anthropic client for direct call
            client = anthropic.AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
            
            # Prepare the prompt
            prompt = f"""
            You are a helpful AI assistant that provides accurate and concise answers.
            
            Context information: {json.dumps(task.context) if task.context else "{}"}
            
            User task: {task.task}
            
            Please provide a helpful response based on the task and context provided.
            """
            
            # Make direct Claude API call with thinking steps for visualization
            await self.send_update("step_update", {
                "task_id": task.task_id,
                "step": {
                    "step": 2,
                    "description": "Thinking...",
                    "timestamp": asyncio.get_event_loop().time()
                }
            })
            
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
            
            # Update task with result
            task.result = response.content[0].text
            task.status = "completed"
            task.completed_at = asyncio.get_event_loop().time()
            
            # Add final step
            task.screenshots.append({
                "step": 3,
                "url": None,
                "base64": None,
                "description": "Task completed",
                "timestamp": asyncio.get_event_loop().time()
            })
            
            await self.send_update("task_update", {
                "task_id": task.task_id,
                "status": task.status,
                "result": task.result,
                "step": {
                    "step": 3,
                    "description": "Task completed",
                    "timestamp": asyncio.get_event_loop().time()
                }
            })
            
            return task
            
        except Exception as e:
            error_msg = f"Direct LLM agent error: {str(e)}"
            logger.error(error_msg)
            task.status = "error"
            task.error = error_msg
            task.completed_at = asyncio.get_event_loop().time()
            
            await self.send_update("task_update", {
                "task_id": task.task_id,
                "status": task.status,
                "error": task.error
            })
            
            return task
    
    async def synthesize_results(self, completed_tasks: List[AgentTask]) -> str:
        """Combine the results from all subtasks into a coherent final response."""
        logger.info("Synthesizing results from completed tasks")
        
        # Extract results from each task
        task_results = []
        for task in completed_tasks:
            if task.status == "completed" and task.result:
                task_results.append({
                    "task": task.task,
                    "agent_type": task.agent_type,
                    "result": task.result
                })
            elif task.status == "error":
                task_results.append({
                    "task": task.task,
                    "agent_type": task.agent_type,
                    "result": f"Error: {task.error}"
                })
        
        # Prompt the LLM to synthesize the results
        prompt = f"""
        You are a synthesis assistant. You've received the results from multiple parallel tasks and need to 
        synthesize them into a coherent, comprehensive response.
        
        Original task: {self.task}
        
        Results from subtasks:
        {json.dumps(task_results, indent=2)}
        
        Please synthesize these results into a unified, coherent response that fully addresses the original task.
        Focus on insights, connections between the subtasks, and providing a complete answer.
        """
        
        try:
            response = await self.llm.ainvoke(prompt)
            return response.content[0].text
        except Exception as e:
            logger.error(f"Failed to synthesize results: {str(e)}")
            # Fallback: just concatenate the results
            return "\n\n".join([
                f"Results from task: {task['task']}\n{task['result']}"
                for task in task_results
            ])
    
    async def run(self) -> str:
        """
        Run the master agent, which plans tasks, executes them concurrently,
        and synthesizes the results.
        """
        logger.info(f"Master agent starting for task: {self.task}")
        
        # Step 1: Plan the tasks
        await self.send_update("status_update", {
            "status": "planning",
            "message": "Breaking down the task into subtasks..."
        })
        
        subtasks = await self.plan_tasks()
        
        await self.send_update("plan_update", {
            "status": "planned",
            "subtasks": [task.dict() for task in subtasks]
        })
        
        # Step 2: Execute the tasks concurrently
        await self.send_update("status_update", {
            "status": "executing",
            "message": "Executing subtasks concurrently..."
        })
        
        tasks = []
        for task in subtasks:
            if task.agent_type == AgentType.BROWSER:
                tasks.append(self.run_browser_agent(task))
            elif task.agent_type == AgentType.DIRECT_LLM:
                tasks.append(self.run_direct_llm_agent(task))
            else:
                logger.warning(f"Unknown agent type: {task.agent_type}")
        
        completed_tasks = await asyncio.gather(*tasks)
        
        # Step 3: Synthesize the results
        await self.send_update("status_update", {
            "status": "synthesizing",
            "message": "Synthesizing results from all subtasks..."
        })
        
        final_result = await self.synthesize_results(completed_tasks)
        
        # Step 4: Return the final result
        await self.send_update("status_update", {
            "status": "completed",
            "message": "All tasks completed",
            "final_result": final_result
        })
        
        return final_result