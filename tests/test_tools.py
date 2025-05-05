import asyncio
from tools.tools import BrowserTool


async def test_browser_tool():
    tool = BrowserTool()
    # Example: searching for OpenAI on Google
    result = await tool._arun("search for OpenAI on Google", context={})
    print(result)


if __name__ == "__main__":
    # Comment out or keep your DirectLLMTool test if you want
    # asyncio.run(test_direct_llm_tool())
    asyncio.run(test_browser_tool())
