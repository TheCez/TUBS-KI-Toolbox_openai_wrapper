import traceback
from anthropic import Anthropic

client = Anthropic(base_url="http://localhost:8000/v1", api_key="<ADD_YOUR_API_KEY_HERE>")

print("--- Testing Anthropic Roles & Stop Sequences ---")

try:
    response = client.messages.create(
        model="claude-3-opus-20240229",
        max_tokens=100,
        system="You are a helpful assistant. Keep answers under 2 sentences.",
        messages=[
            {"role": "user", "content": "Count from 1 to 10. For example: 1, 2, 3, 4..."}
        ],
        stop_sequences=["6", "six"],
        stream=False
    )
    
    print("\n[Response]:\n", response.content[0].text)
    print(f"\n[Finish Reason] (Should be 'stop_sequence'): {response.stop_reason}")
except Exception as e:
    print("Test Failed!")
    traceback.print_exc()
