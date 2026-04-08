from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="<ADD_YOUR_API_KEY_HERE>")

print("--- Testing Roles & Stop Sequences ---")

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "You are a helpful assistant. Keep answers under 2 sentences."},
        {"role": "user", "content": "Count from 1 to 10. For example: 1, 2, 3, 4..."}
    ],
    # The moment the AI tries to output "6", the wrapper should kill the connection
    stop=["6", "six"], 
    stream=False
)

print("\n[Response]:\n", response.choices[0].message.content)
print(f"\n[Finish Reason] (Should be 'stop'): {response.choices[0].finish_reason}")