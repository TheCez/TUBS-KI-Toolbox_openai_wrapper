from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="86653868c86740a8a1141512a74abd7e")

print("Sending autonomous tool request to local wrapper...")

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "What is the weather like in Tokyo?"}],
    tools=[{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "The city, e.g., Tokyo"}
                },
                "required": ["location"]
            }
        }
    }],
    stream=True
)

for chunk in response:
    delta = chunk.choices[0].delta
    
    # This proves the wrapper successfully parsed the XML back into standard JSON
    if delta.tool_calls:
        tc = delta.tool_calls[0]
        if tc.function.name:
            print(f"\n[Parsed Function Name]: {tc.function.name}")
        if tc.function.arguments:
            print(f"[Parsed Arguments Chunk]: {tc.function.arguments}")
            
    # Fallback just in case it leaks
    elif delta.content:
        print(delta.content, end="", flush=True)

print("\n\nStream connection closed successfully.")