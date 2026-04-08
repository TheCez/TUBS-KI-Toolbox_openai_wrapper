from openai import OpenAI
import json

client = OpenAI(base_url="http://localhost:8000/v1", api_key="86653868c86740a8a1141512a74abd7e")

print("--- Testing JSON Mode & Structured Outputs ---")

# We pass a strict JSON schema that your wrapper will inject into the prompt
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Generate a profile for a fake cyberpunk character."}],
    response_format={
        "type": "json_schema",
        "json_schema": {
            "name": "character_profile",
            "schema": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "alias": {"type": "string"},
                    "cyberware": {"type": "array", "items": {"type": "string"}},
                    "danger_level": {"type": "integer", "description": "1 to 10"}
                },
                "required": ["name", "alias", "cyberware", "danger_level"]
            }
        }
    }
)

raw_content = response.choices[0].message.content
print("\n[Raw Response String]:\n", raw_content)

# If this loads without throwing a json.decoder.JSONDecodeError, your wrapper succeeds!
parsed_json = json.loads(raw_content)
print("\n[Successfully Parsed JSON!]")