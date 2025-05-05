import asyncio
import base64
import os
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from browser_use import Agent
from browser_use.browser.views import BrowserState
from browser_use.agent.views import AgentOutput

# Load environment variables
load_dotenv()

# Initialize LLM
llm = ChatAnthropic(model='claude-3-opus-20240229')  # You can replace with your preferred model

# Helper function to convert base64 string to image
def base64_to_image(base64_string: str, output_filename: str):
    """Convert base64 string to image."""
    if not os.path.exists(os.path.dirname(output_filename)):
        os.makedirs(os.path.dirname(output_filename))
    
    img_data = base64.b64decode(base64_string)
    with open(output_filename, "wb") as f:
        f.write(img_data)
    return output_filename

# Callback function to capture screenshots at each step
def new_step_callback(state: BrowserState, model_output: AgentOutput, steps: int):
    """Capture screenshot after each step."""
    # Create path for storing screenshot
    path = f"./screenshots/{steps}.png"
    
    # Get the latest screenshot from browser state
    last_screenshot = state.screenshot
    
    # Save the screenshot
    img_path = base64_to_image(
        base64_string=str(last_screenshot),
        output_filename=path
    )
    print(f"Screenshot saved: {img_path}")

# Main function
async def main():
    # Initialize the agent with the callback function
    agent = Agent(
        task="Compare openai and anthropic api pricing", # Replace with your specific task
        llm=llm,
        register_new_step_callback=new_step_callback,
    )
    
    # Run the agent
    result = await agent.run()
    
    # Additional processing if needed
    print("Task completed!")

# Run the main function
if __name__ == "__main__":
    asyncio.run(main())
