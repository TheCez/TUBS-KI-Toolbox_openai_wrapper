from openai import OpenAI
import os

client = OpenAI(base_url="http://localhost:8000/v1", api_key=os.environ["TUBS_API_KEY"])

print("--- Testing SSE Streaming ---")

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Write a short haiku about coding."}],
    stream=True
)

for chunk in response:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)

print("\n\n[Stream Finished cleanly.]")
