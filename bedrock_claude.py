import boto3
import os
import json
from typing import Optional

class BedrockClaudeClient:
    """
    Wrapper for interacting with Claude via AWS Bedrock.
    """
    def __init__(self):
        self.region = os.getenv("AWS_REGION_BEDROCK")
        self.client = boto3.client(
            "bedrock-runtime",
            region_name=self.region,
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        )
        # Claude model id for Bedrock (Sonnet, Opus, etc)
        self.model_id = os.getenv("BEDROCK_CLAUDE_MODEL_ID", "anthropic.claude-3-sonnet-20240229-v1:0")

    def chat(self, prompt, max_tokens=4096, temperature=0):
        body = {
            "prompt": prompt,
            "max_tokens_to_sample": max_tokens,
            "temperature": temperature,
        }
        response = self.client.invoke_model(
            modelId=self.model_id,
            body=json.dumps(body),
            contentType="application/json",
            accept="application/json",
        )
        result = json.loads(response["body"].read())
        return result.get("completion", "")
