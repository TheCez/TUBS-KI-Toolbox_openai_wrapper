from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="<ADD_YOUR_API_KEY_HERE>")

print("Sending parallel tool request to local wrapper...")

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "What is the weather like in Tokyo, Paris and Berlin right now?"}],
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
    
    # Iterate through the array of tool calls (crucial for parallel processing)
    if delta.tool_calls:
        for tool_call in delta.tool_calls:
            # The 'index' property tells us which parallel tool this chunk belongs to
            idx = tool_call.index
            
            if tool_call.function.name:
                print(f"\n[Tool {idx} Triggered]: {tool_call.function.name}")
            if tool_call.function.arguments:
                # We print the arguments as they stream in
                print(tool_call.function.arguments, end="", flush=True)
                
    # Fallback for normal text or reasoning
    elif delta.content:
        print(delta.content, end="", flush=True)

print("\n\nStream connection closed successfully.")