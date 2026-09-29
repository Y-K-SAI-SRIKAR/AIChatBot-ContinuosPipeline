import json
import os
import boto3
from dotenv import load_dotenv 

load_dotenv()
def main(prompt):
    prompt_Content = prompt
    client = boto3.client("bedrock-runtime", region_name="us-east-1")
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "text": prompt_Content
                }
            ]
        }
    ]
    model_id = os.getenv("MODEL_ID")
    response = client.converse(
        modelId=model_id,
        messages=messages,
        inferenceConfig={
            "maxTokens": 512,
            "temperature": 0.75
        }
    )

    response_text = response["output"]["message"]["content"][0]["text"]
    print("Model response to Prompt: \n")
    return response_text
