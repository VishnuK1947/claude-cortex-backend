import os
import asyncio
import json
import uuid
import logging
from typing import Dict, List, Any, Optional, TypedDict, Annotated, Union
from dotenv import load_dotenv
from fastapi import WebSocket

from langchain_anthropic import ChatAnthropic
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

from tools.tools import BrowserTool, DirectLLMTool, base64_to_image, SecureClaudeTool

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


# Define state structure
class MultiAgentState(TypedDict):
    # The master task
    master_task: str
    # Context for the task
    context: Dict[str, Any]
    # Subtasks to be executed
    subtasks: List[Dict[str, Any]]
    # Current subtask index
    current_subtask_index: int
    # Results from completed subtasks
    subtask_results: List[Dict[str, Any]]
    # Final combined result
    final_result: Optional[str]
    # Any errors encountered
    errors: List[str]
    # Session ID
    session_id: str
    # WebSocket for live updates (not serialized)
    websocket: Optional[Any]
    # Flag to indicate completion
    is_complete: bool


# Helper functions
def create_empty_multi_agent_state(
    task: str, context: Dict[str, Any], websocket=None
) -> MultiAgentState:
    """Create an initial empty state for the multi-agent system."""
    session_id = str(uuid.uuid4())

    return {
        "master_task": task,
        "context": context,
        "subtasks": [],
        "current_subtask_index": -1,  # Start at -1 to indicate planning phase
        "subtask_results": [],
        "final_result": None,
        "errors": [],
        "session_id": session_id,
        "websocket": websocket,
        "is_complete": False,
    }


# Define the master agent nodes
async def task_planner(state: MultiAgentState) -> MultiAgentState:
    """Plan the execution by breaking down the task into subtasks."""
    try:
        # Use Claude to create a plan
        client = ChatAnthropic(model="claude-3-7-sonnet-20250219")

        # Construct the planning prompt
        planning_prompt = f"""
        # Task Planning
        
        You are a task planning AI that breaks down complex tasks into simpler subtasks that can be executed by specialized agents.
        
        ## Master Task
        {state["master_task"]}
        
        ## Context
        {json.dumps(state["context"], indent=2)}
        
        ## Available Agent Types
        1. Browser Agent: Can interact with websites, fill forms, click buttons, etc.
           Use this for tasks that require web navigation, data collection from websites, etc.
        
        2. Direct LLM Agent: Can answer questions, analyze data, and perform reasoning without browser interaction.
           Use this for tasks that require analysis, summarization, creative writing, etc.
        
        ## Instructions
        Break down the master task into 1-5 subtasks, each assigned to the most appropriate agent type.
        
        Format your response as a JSON array of subtasks, each with:
        - "id": A unique identifier for the subtask (e.g., "subtask_1")
        - "description": A detailed description of what the subtask should accomplish
        - "agent_type": Either "browser" or "direct_llm"
        - "dependencies": List of subtask IDs that must be completed before this one (can be empty)
        
        Example:
        ```json
        [
          {{
            "id": "subtask_1",
            "description": "Search for recent news articles about renewable energy",
            "agent_type": "browser",
            "dependencies": []
          }},
          {{
            "id": "subtask_2",
            "description": "Analyze the findings and summarize key trends",
            "agent_type": "direct_llm",
            "dependencies": ["subtask_1"]
          }}
        ]
        ```
        
        Only respond with the JSON array, no additional text.
        """

        # Get the plan from Claude
        response = await client.ainvoke(planning_prompt)

        # Extract the JSON plan
        plan_text = response.content

        # Clean up the response to ensure it's valid JSON
        if "```json" in plan_text:
            plan_text = plan_text.split("```json")[1].split("```")[0].strip()
        elif "```" in plan_text:
            plan_text = plan_text.split("```")[1].split("```")[0].strip()

        # Parse the plan
        subtasks = json.loads(plan_text)

        # Update state with planned subtasks
        updated_state = state.copy()
        updated_state["subtasks"] = subtasks
        updated_state["current_subtask_index"] = 0  # Start with the first subtask

        # Inform the client about the plan
        if state["websocket"]:
            try:
                await state["websocket"].send_json(
                    {
                        "status": "planning_complete",
                        "message": f"Task broken down into {len(subtasks)} subtasks",
                        "subtasks": subtasks,
                    }
                )
            except Exception as e:
                logger.error(f"WebSocket error during planning: {e}")

        return updated_state

    except Exception as e:
        # Handle errors
        logger.error(f"Error in task planner: {e}")
        updated_state = state.copy()
        updated_state["errors"].append(f"Planning error: {str(e)}")

        # Inform the client about the error
        if state["websocket"]:
            try:
                await state["websocket"].send_json(
                    {"status": "planning_error", "error": str(e)}
                )
            except Exception as ws_e:
                logger.error(f"WebSocket error: {ws_e}")

        return updated_state


async def subtask_executor(state: MultiAgentState) -> MultiAgentState:
    """Execute the current subtask based on its agent type."""
    try:
        # Get the current subtask
        if state["current_subtask_index"] >= len(state["subtasks"]):
            # No more subtasks to execute
            updated_state = state.copy()
            updated_state["is_complete"] = True
            return updated_state

        current_subtask = state["subtasks"][state["current_subtask_index"]]

        # Check if all dependencies are satisfied
        dependencies = current_subtask.get("dependencies", [])
        completed_subtask_ids = [task["id"] for task in state["subtask_results"]]

        if not all(dep in completed_subtask_ids for dep in dependencies):
            # Dependencies not satisfied, move to the next subtask
            updated_state = state.copy()
            updated_state["current_subtask_index"] += 1
            return updated_state

        # Prepare context with results from dependencies
        subtask_context = state["context"].copy()
        for dep_id in dependencies:
            for result in state["subtask_results"]:
                if result["id"] == dep_id:
                    subtask_context[f"result_from_{dep_id}"] = result["result"]

        # Inform the client about the current subtask
        if state["websocket"]:
            try:
                await state["websocket"].send_json(
                    {
                        "status": "executing_subtask",
                        "subtask": current_subtask,
                        "subtask_index": state["current_subtask_index"],
                        "total_subtasks": len(state["subtasks"]),
                    }
                )
            except Exception as e:
                logger.error(f"WebSocket error: {e}")

        # Execute the subtask based on agent type
        subtask_result = None
        if current_subtask["agent_type"] == "browser":
            # Use Browser Tool
            browser_tool = BrowserTool()

            # Create a new WebSocket handler for this subtask
            class SubtaskWebSocketHandler:
                def __init__(self, parent_websocket):
                    self.parent_websocket = parent_websocket
                    self.accepted = False

                async def accept(self):
                    self.accepted = True

                async def send_json(self, data):
                    if self.parent_websocket:
                        # Add subtask information to the data
                        data["subtask_id"] = current_subtask["id"]
                        data["subtask_index"] = state["current_subtask_index"]
                        await self.parent_websocket.send_json(data)

                async def receive_json(self):
                    # This is a dummy method as we don't expect to receive from this handler
                    return {
                        "task": current_subtask["description"],
                        "context": subtask_context,
                    }

            subtask_websocket = SubtaskWebSocketHandler(state["websocket"])
            subtask_result = await browser_tool._arun(
                current_subtask["description"], subtask_context, subtask_websocket
            )

        elif current_subtask["agent_type"] == "direct_llm":
            # Use Direct LLM Tool
            llm_tool = DirectLLMTool()
            subtask_result = await llm_tool.run(
                current_subtask["description"], subtask_context
            )

        # Update state with subtask result
        updated_state = state.copy()
        result_entry = {
            "id": current_subtask["id"],
            "description": current_subtask["description"],
            "agent_type": current_subtask["agent_type"],
            "result": (
                subtask_result.get("result", str(subtask_result))
                if isinstance(subtask_result, dict)
                else str(subtask_result)
            ),
        }
        updated_state["subtask_results"].append(result_entry)
        updated_state["current_subtask_index"] += 1

        # Inform the client about the subtask completion
        if state["websocket"]:
            try:
                await state["websocket"].send_json(
                    {
                        "status": "subtask_complete",
                        "subtask_id": current_subtask["id"],
                        "subtask_index": state["current_subtask_index"] - 1,
                        "result": result_entry["result"],
                    }
                )
            except Exception as e:
                logger.error(f"WebSocket error: {e}")

        return updated_state

    except Exception as e:
        # Handle errors
        logger.error(f"Error in subtask executor: {e}")
        updated_state = state.copy()
        updated_state["errors"].append(
            f"Execution error for subtask {state['current_subtask_index']}: {str(e)}"
        )
        updated_state["current_subtask_index"] += 1  # Move to next subtask

        # Inform the client about the error
        if state["websocket"]:
            try:
                await state["websocket"].send_json(
                    {
                        "status": "subtask_error",
                        "subtask_index": state["current_subtask_index"] - 1,
                        "error": str(e),
                    }
                )
            except Exception as ws_e:
                logger.error(f"WebSocket error: {ws_e}")

        return updated_state


async def result_combiner(state: MultiAgentState) -> MultiAgentState:
    """Combine the results from all subtasks into a final result."""
    try:
        # Use Claude to combine results
        client = ChatAnthropic(model="claude-3-7-sonnet-20250219")

        # Construct the prompt for combining results
        combining_prompt = f"""
        # Result Combination
        
        You are an AI that combines results from multiple subtasks into a coherent final response.
        
        ## Original Master Task
        {state["master_task"]}
        
        ## Subtask Results
        {json.dumps(state["subtask_results"], indent=2)}
        
        ## Instructions
        Create a comprehensive and coherent response that addresses the original master task,
        integrating all the information from the subtask results.
        
        Your response should be well-structured, clear, and directly address what the user asked for
        in the master task. Include all relevant information from the subtasks while avoiding repetition.
        
        Only respond with the final combined result, no additional explanations or meta-commentary.
        """

        # Get the combined result from Claude
        response = await client.ainvoke(combining_prompt)
        combined_result = response.content

        # Update state with final result
        updated_state = state.copy()
        updated_state["final_result"] = combined_result
        updated_state["is_complete"] = True

        # Inform the client about the completed task
        if state["websocket"]:
            try:
                await state["websocket"].send_json(
                    {"status": "task_complete", "result": combined_result, "done": True}
                )
            except Exception as e:
                logger.error(f"WebSocket error: {e}")

        return updated_state

    except Exception as e:
        # Handle errors
        logger.error(f"Error in result combiner: {e}")
        updated_state = state.copy()
        updated_state["errors"].append(f"Result combination error: {str(e)}")
        updated_state["is_complete"] = True

        # Create a simple final result from the available subtask results
        simple_result = "I encountered an error while combining the results, but here's what I found:\n\n"
        for result in state["subtask_results"]:
            simple_result += f"- {result['description']}:\n{result['result']}\n\n"

        updated_state["final_result"] = simple_result

        # Inform the client about the error
        if state["websocket"]:
            try:
                await state["websocket"].send_json(
                    {
                        "status": "combination_error",
                        "error": str(e),
                        "result": simple_result,
                        "done": True,
                    }
                )
            except Exception as ws_e:
                logger.error(f"WebSocket error: {ws_e}")

        return updated_state


def should_continue_execution(state: MultiAgentState) -> str:
    """Determine the next step in the execution flow."""
    # If there are errors and max retries reached, go to result combiner
    if len(state["errors"]) >= 5:
        return "result_combiner"

    # If all subtasks are executed, go to result combiner
    if state["current_subtask_index"] >= len(state["subtasks"]):
        return "result_combiner"

    # Otherwise, continue executing subtasks
    return "subtask_executor"


# Create the LangGraph workflow
def create_multi_agent_graph():
    """Create the LangGraph workflow for the multi-agent system."""
    # Initialize the state graph
    workflow = StateGraph(MultiAgentState)

    # Add nodes to the graph
    workflow.add_node
