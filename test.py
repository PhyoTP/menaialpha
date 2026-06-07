from dotenv import load_dotenv
import os
load_dotenv()
api_key = os.getenv("HCAI_API_KEY")
import requests


response = requests.post(
    "https://ai.hackclub.com/proxy/v1/chat/completions",
    headers={
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    },
    json={
        "model": "qwen/qwen3.7-plus",
        "messages": [
            {"role": "user", "content": "Hello!"}
        ]
    }
)


print(response.status_code, response.json())