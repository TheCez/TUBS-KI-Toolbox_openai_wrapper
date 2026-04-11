import traceback
from anthropic import Anthropic

client = Anthropic(base_url="http://localhost:8000/v1", api_key="<ADD_YOUR_API_KEY_HERE>")

print("--- Testing Anthropic System Prompt & Multi-Turn Parity ---")

try:
    # Test 1: System prompt with structured instructions
    response = client.messages.create(
        model="claude-3-7-sonnet-20250219",
        max_tokens=150,
        system="You are a logical solver. Think step-by-step. Wrap your reasoning in <thought> tags.",
        messages=[
            {"role": "user", "content": "If I have 3 apples and give 1 away, how many do I have?"}
        ],
        stream=False
    )

    print("\n[System Prompt + Reasoning Response]:")
    print(response.content[0].text)
    print(f"\n[Stop Reason]: {response.stop_reason}")
    print(f"[Model Used]: {response.model}")
    print(f"[Usage]: Input={response.usage.input_tokens}, Output={response.usage.output_tokens}")

except Exception as e:
    print("Test 1 (System Prompt) Failed!")
    traceback.print_exc()

print("\n" + "="*50)

try:
    # Test 2: Multi-turn conversation with tool_result role simulation
    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=100,
        messages=[
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "2+2 = 4."},
            {"role": "user", "content": "Now multiply that result by 3."}
        ],
        stream=False
    )

    print("\n[Multi-Turn Response]:")
    print(response.content[0].text)
    print(f"\n[Stop Reason]: {response.stop_reason}")

except Exception as e:
    print("Test 2 (Multi-Turn) Failed!")
    traceback.print_exc()

print("\n" + "="*50)

try:
    # Test 3: Model list endpoint
    import httpx
    r = httpx.get("http://localhost:8000/v1/models")
    models = r.json()
    
    anthropic_models = [m["id"] for m in models["data"] if m.get("owned_by") == "anthropic-shim"]
    tubs_models = [m["id"] for m in models["data"] if m.get("owned_by") == "tu-bs"]
    
    print(f"\n[GET /v1/models] Total: {len(models['data'])} models")
    print(f"  TU-BS native: {len(tubs_models)}")
    print(f"  Anthropic mapped: {len(anthropic_models)}")
    print(f"  Anthropic IDs: {anthropic_models}")

except Exception as e:
    print("Test 3 (Models Endpoint) Failed!")
    traceback.print_exc()

print("\n\n=== ALL ANTHROPIC PARITY TESTS COMPLETE ===")
