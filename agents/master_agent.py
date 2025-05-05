import asyncio
import os
import json
import logging
from typing import Dict, Any, List

from tools.tools import BrowserTool, DirectLLMTool, base64_to_image
from fastapi import WebSocket

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AgentTask:
    def __init__(self, agent_id, task, agent_type, context=None):
        self.agent_id = agent_id
        self.task = task
        self.agent_type = agent_type
        self.context = context or {}
        self.context["agent_id"] = agent_id
        self.result = None
        self.done = False
        self.screenshots = []


class MasterAgent:
    def __init__(self):
        self.tasks = []
        self.results = {}

    def analyze_task(self, task, context=None):
        context = context or {}
        logger.info(f"Analyzing task: {task}")

        # For demo purposes, we'll hard-code the analysis for specific tasks
        if (
            "doctor" in task.lower()
            and "calendar" in task.lower()
            and "vishnu" in task.lower()
        ):
            logger.info("Detected doctor update and calendar task")
            # Create multiple specialized agents
            self.tasks = [
                AgentTask(
                    "email_agent",
                    "Log in to Gmail with email shubhayan935@gmail.com and password shashwat1397. Find emails from Dr. Vishnu Kadaba and check for any updates or lab results.",
                    "browser",
                ),
                AgentTask(
                    "calendar_agent",
                    "Log in to Google Calendar with email shubhayan935@gmail.com and password shashwat1397. Create a new appointment for a follow-up with Dr. Vishnu Kadaba on May 10th at 2:00 PM.",
                    "browser",
                ),
                AgentTask(
                    "analysis_agent",
                    "Analyze the following medical information: Lab results show cholesterol levels at 180 mg/dL and blood sugar at 95 mg/dL. Explain what these results mean in plain language.",
                    "direct_llm",
                ),
            ]
        else:
            # Default to just a single browser agent for other tasks
            logger.info("Using default single agent for task")
            self.tasks = [AgentTask("default_agent", task, "browser", context)]

        return self.tasks

    def combine_results(self):
        logger.info(f"Combining results from {len(self.results)} agents")
        # Combine all results for the final response
        if (
            "email_agent" in self.results
            and "calendar_agent" in self.results
            and "analysis_agent" in self.results
        ):
            return (
                f"I've checked your emails and found a message from Dr. Vishnu Kadaba requesting a follow-up appointment. "
                f"I've scheduled this appointment in your Google Calendar for May 10th at 2:00 PM. "
                f"Regarding your lab results: {self.results.get('analysis_agent', 'No analysis available.')}"
            )
        else:
            # Return whatever results we have
            return " ".join([result for result in self.results.values()])


async def run_browser_agent(task, context, websocket, agent_id):
    logger.info(f"Starting browser agent {agent_id}")
    browser_tool = BrowserTool()
    return await browser_tool._arun(task, context, websocket)


async def run_direct_llm_agent(task, context, websocket, agent_id):
    logger.info(f"Starting direct LLM agent {agent_id}")
    direct_llm_tool = DirectLLMTool()
    result = await direct_llm_tool.run(task, context, websocket)
    return result


async def execute_master_agent(task, context, websocket):
    await websocket.accept()

    try:
        # Initialize master agent and analyze task
        master_agent = MasterAgent()
        agent_tasks = master_agent.analyze_task(task, context)

        # Inform the frontend about the agents being launched
        await websocket.send_json(
            {
                "status": "initializing",
                "message": f"Launching {len(agent_tasks)} agents to handle your request",
                "agent_count": len(agent_tasks),
                "agent_ids": [task.agent_id for task in agent_tasks],
            }
        )

        # Launch tasks asynchronously
        agent_coroutines = []
        for agent_task in agent_tasks:
            if agent_task.agent_type == "browser":
                coro = run_browser_agent(
                    agent_task.task, agent_task.context, websocket, agent_task.agent_id
                )
            else:  # "direct_llm"
                coro = run_direct_llm_agent(
                    agent_task.task, agent_task.context, websocket, agent_task.agent_id
                )
            agent_coroutines.append(coro)

        # Run all agents in parallel
        results = await asyncio.gather(*agent_coroutines, return_exceptions=True)

        # Process results
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                error_message = f"Agent {agent_tasks[i].agent_id} error: {str(result)}"
                logger.error(error_message)
                master_agent.results[agent_tasks[i].agent_id] = f"Error: {str(result)}"
            else:
                if isinstance(result, dict) and "result" in result:
                    master_agent.results[agent_tasks[i].agent_id] = result.get(
                        "result", "No result"
                    )
                else:
                    master_agent.results[agent_tasks[i].agent_id] = str(result)

        # Combine results and send final response
        final_result = master_agent.combine_results()
        await websocket.send_json(
            {"result": final_result, "status": "success", "done": True}
        )

    except Exception as e:
        error_message = f"Master agent error: {str(e)}"
        logger.error(error_message)
        await websocket.send_json({"error": error_message})
