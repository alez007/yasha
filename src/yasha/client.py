import sys
import ray
import time
import requests

# r = requests.get("http://localhost:8000/server", stream=True)
# start = time.time()
# r.raise_for_status()
# for i, chunk in enumerate(r.iter_content(chunk_size=None, decode_unicode=True)):
#     print(f"Got result {round(time.time()-start, 1)}s after start: '{chunk}'")
#     if i == 5:
#         print("Client disconnecting")
#         break


r = requests.post("http://127.0.0.1:8000/app1/instruct", json={"input": "how are you today ?"})
# print(r.json())

# from openai import OpenAI

# # Initialize client
# client = OpenAI(base_url="http://127.0.0.1:8000/app2/v1", api_key="fake-key")

# # Basic completion
# response = client.chat.completions.create(
#     model="gemma-3-1b-it",
#     messages=[{"role": "user", "content": "thanks"}],
#     stream=True
# )

# for chunk in response:
#     sys.stdout.write(chunk.choices[0].delta.content)
#     # print(chunk)
#     # print(chunk.choices[0].delta.content)
#     # print("****************")


