from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="86653868c86740a8a1141512a74abd7e")

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