import traceback
import json
from anthropic import Anthropic

client = Anthropic(base_url="http://localhost:8000/v1", api_key="<ADD_YOUR_API_KEY_HERE>")

print("--- Testing Anthropic JSON Pre-fill Mode ---")

try:
    # We pass a strict JSON format request as a system prompt, appending an opening brace via assistant to force JSON natively
    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=200,
        system="Output exactly and only valid JSON matching this schema: {'name': 'string', 'alias': 'string', 'cyberware': ['string'], 'danger_level': int}. Do not output markdown code blocks or any other text.",
        messages=[
            {"role": "user", "content": "Generate a profile for a fake cyberpunk character."},
            {"role": "assistant", "content": "{\n"}
        ]
    )
    
    raw_content = "{" + response.content[0].text
    print("\n[Raw Response String]:\n", raw_content)

    # If this loads without throwing a json.decoder.JSONDecodeError, natively supported!
    parsed_json = json.loads(raw_content)
    print("\n[Successfully Parsed JSON!]")
except Exception as e:
    print("Test Failed!")
    traceback.print_exc()
