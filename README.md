# Claude Cortex Backend

A FastAPI backend with a LangGraph-based multi-agent architecture using Claude's API and browser-use for web automation.

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file in the root directory with your Anthropic API key:
```
ANTHROPIC_API_KEY=your_api_key_here
```

## Running the Backend

Start the server:
```bash
uvicorn main:app --reload
```

The server will run on `http://localhost:8000`

## API Endpoints

### POST /execute_task
Execute a task using the multi-agent system.

Request body:
```json
{
    "task": "Your task description",
    "context": {
        "any": "additional context"
    }
}
```

Response:
```json
{
    "result": "Task execution result",
    "status": "success"
}
```

## Architecture

The system uses:
- FastAPI for the web server
- LangGraph for the multi-agent architecture
- Claude's API for agent intelligence
- browser-use for web automation

The workflow consists of:
1. Master Agent: Analyzes the task and creates a plan
2. Execution Agent: Carries out the plan using available tools
