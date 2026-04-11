import traceback
import os
from anthropic import Anthropic

client = Anthropic(base_url="http://localhost:8000", api_key=os.environ["TUBS_API_KEY"])

print("--- Testing Anthropic Multimodal Vision ---")

# A tiny 1x1 pixel red dot base64 image
base64_image = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="

try:
    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=100,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What color is this 1x1 pixel image?"},
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": base64_image
                        }
                    }
                ]
            }
        ]
    )

    print("\n[Vision Response]:\n", response.content[0].text)
    print(f"\n[Stop Reason]: {response.stop_reason}")
    print(f"[Usage]: Input={response.usage.input_tokens}, Output={response.usage.output_tokens}")
except Exception as e:
    print("Test Failed!")
    traceback.print_exc()
