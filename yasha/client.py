import sys
import ray
import time
import requests

from openai import OpenAI

# Initialize client (change IP with yours)
client = OpenAI(base_url="http://192.168.2.27:8000/v1", api_key="fake-key")

response = client.embeddings.create(
    model="nomic-embed-text-v1.5",
    input="Text to be embedded"
)

print(response)

# Basic completion
# response = client.chat.completions.create(
#     model="Llama-3.1-8B",
#     messages=[
#         {"role": "user", "content": "how about a good joke?"}],
#     stream=True
# )

# for chunk in response:
#     sys.stdout.write(chunk.choices[0].delta.content)


