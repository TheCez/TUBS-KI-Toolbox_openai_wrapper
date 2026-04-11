import traceback
import os
from anthropic import Anthropic

client = Anthropic(base_url="http://localhost:8000", api_key=os.environ["TUBS_API_KEY"])

print("--- Testing Anthropic SSE Streaming ---")

try:
    with client.messages.stream(
        model="claude-3-5-sonnet-20241022",
        max_tokens=100,
        messages=[{"role": "user", "content": "Write a short haiku about coding."}]
    ) as stream:
        for text in stream.text_stream:
            print(text, end="", flush=True)

    print("\n\n[Stream Finished cleanly.]")
except Exception as e:
    print("Test Failed!")
    traceback.print_exc()
