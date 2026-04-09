from dotenv import load_dotenv
import os
load_dotenv()
api_key = os.getenv("HCAI_API_KEY")
import requests

response = requests.post(
    "https://ai.hackclub.com/proxy/v1/embeddings",
    headers={"Authorization": f"Bearer {api_key}"},
    json={"model": "qwen/qwen3-embedding-8b", "input": "test"}
)
print(response.status_code, response.json())