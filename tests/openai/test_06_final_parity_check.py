from openai import OpenAI
import os

client = OpenAI(base_url="http://localhost:8000/v1", api_key=os.environ["TUBS_API_KEY"])

print("--- Testing Developer Role & Reasoning Mapping ---")

response = client.chat.completions.create(
    model="o3", # Uses a 2026 reasoning model name
    messages=[
        {"role": "developer", "content": "You are a logical solver. Think step-by-step."},
        {"role": "user", "content": "If I have 3 apples and give 1 away, how many do I have? Wrap your logic in <thought> tags."}
    ],
    stream=False
)

# Check if the developer role was accepted without error
print("\n[Response]:\n", response.choices[0].message.content)
print(f"\n[Finish Reason]: {response.choices[0].finish_reason}")
