from openai import OpenAI
import os

client = OpenAI(base_url="http://localhost:8000/v1", api_key=os.environ["TUBS_API_KEY"])

print("--- Testing Parallel Agentic Tools (Stream) ---")

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "What is the weather like in Tokyo and Paris right now?"}],
    tools=[{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string"}
                },
                "required": ["location"]
            }
        }
    }],
    stream=True
)

for chunk in response:
    delta = chunk.choices[0].delta
    
    if delta.tool_calls:
        for tool_call in delta.tool_calls:
            idx = tool_call.index
            if tool_call.function.name:
                print(f"\n[Tool {idx} Triggered]: {tool_call.function.name}")
            if tool_call.function.arguments:
                print(tool_call.function.arguments, end="", flush=True)
                
    elif delta.content:
        print(delta.content, end="", flush=True)

print("\n\n[Tool Stream Completed]")
