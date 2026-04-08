from openai import OpenAI

# 1. Initialize the official OpenAI client
client = OpenAI(
    api_key="86653868c86740a8a1141512a74abd7e", # Ignored by your wrapper
    base_url="http://localhost:8000/v1" # This is the magic! Point it to your local server
)

print("Sending request to local wrapper...")

# 2. Make a standard streaming request
response = client.chat.completions.create(
    model="gpt-5.1",
    messages=[
        {"role": "user", "content": "Write a haiku about APIs."}
    ],
    stream=True
)

# 3. Print the stream as it arrives
for chunk in response:
    if chunk.choices[0].delta.content is not None:
        print(chunk.choices[0].delta.content, end="", flush=True)

print("\n\nDone!")