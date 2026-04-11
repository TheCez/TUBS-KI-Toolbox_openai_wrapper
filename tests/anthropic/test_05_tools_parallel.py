import traceback
from anthropic import Anthropic

client = Anthropic(base_url="http://localhost:8000/v1", api_key="<ADD_YOUR_API_KEY_HERE>")

print("--- Testing Anthropic Parallel Tool Use (Stream) ---")

try:
    with client.messages.stream(
        model="claude-3-opus-20240229",
        max_tokens=300,
        tools=[{
            "name": "get_weather",
            "description": "Get the current weather for a given location.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City name"}
                },
                "required": ["location"]
            }
        }],
        messages=[{"role": "user", "content": "What is the weather like in Tokyo and Paris right now?"}]
    ) as stream:
        for event in stream:
            event_type = event.type

            if event_type == "content_block_start":
                block = event.content_block
                if block.type == "tool_use":
                    print(f"\n[Tool Triggered]: {block.name} (id: {block.id})")
                elif block.type == "text":
                    pass  # Text block starting

            elif event_type == "content_block_delta":
                delta = event.delta
                if delta.type == "text_delta":
                    print(delta.text, end="", flush=True)
                elif delta.type == "input_json_delta":
                    print(delta.partial_json, end="", flush=True)

            elif event_type == "message_delta":
                print(f"\n\n[Final Stop Reason]: {event.delta.stop_reason}")

    print("\n[Tool Stream Completed]")
except Exception as e:
    print("Test Failed!")
    traceback.print_exc()
