# TU Braunschweig KI-Toolbox API Wrapper

This repository provides a robust FastAPI gateway that seamlessly translates incoming standard [OpenAI API](https://platform.openai.com/docs/api-reference) requests into the proprietary format used by the TU Braunschweig KI-Toolbox API.

With this wrapper, any client, agentic framework, or application that expects an OpenAI-compatible endpoint can directly communicate with TU Braunschweig's powerful Cloud and Local language models.

## Features

- **Drop-in OpenAI Replacement:** Fully compliant with standard `/v1/chat/completions` requests.
- **Anthropic Compatibility:** Parallel support for Anthropic `/v1/messages` enabling tools like Claude Code or Langchain to use TU-BS.
- **Model Discovery:** Supports `/v1/models` so standard agents can dynamically query available cloud/local nodes, and perfectly routes Anthropic identifiers via an environment variable mapping (`ANTHROPIC_MODEL_MAP`).
- **Bi-modal Routing:** Automatically routes to external endpoints or on-premise institute endpoints depending on the requested model.
- **Streaming Support:** Real-time generation chunked perfectly into Server-Sent Events (SSE) or Anthropic Message events.
- **Vision Support:** Complete support for base64 encoded multipart image messages in OpenAI and Anthropic format.
- **Extensive Parity:** Context mapping and proper formatting for standard clients.
- **Backpressure Controls:** Optional wrapper-side concurrency and pacing limits to protect the TU-BS upstream from bursty agents such as Claude Code.
- **Thread-Aware Context Compaction:** Reuses TU-BS threads while shrinking each outbound prompt to a controlled working set so clients with large histories can stay inside tight upstream token limits.

## Getting Started

### 1. Launching the Gateway with Docker

The best way to run the API wrapper is via Docker Compose to ensure a consistent environment.

Ensure you have [Docker](https://www.docker.com/) and [Docker Compose](https://docs.docker.com/compose/) installed on your system.

```bash
# Start the docker container using compose
docker-compose up -d --build
```

The gateway will now be running locally on **port 8000**. The OpenAI-compatible API base URL is: `http://localhost:8000/v1`

### Concurrency and Rate-Limit Protection

If an agent sends too many requests in parallel and TU-BS responds with `429 Too Many Requests`, configure the wrapper to queue outbound requests instead of forwarding every burst immediately.

```yaml
environment:
  - TUBS_MAX_CONCURRENT_REQUESTS=1
  - TUBS_MIN_REQUEST_INTERVAL_SECONDS=0.5
```

- `TUBS_MAX_CONCURRENT_REQUESTS` limits how many upstream TU-BS requests may be active at once across the wrapper.
- `TUBS_MIN_REQUEST_INTERVAL_SECONDS` enforces a minimum delay between the start of outbound TU-BS requests.
- For aggressive agents, start with `1` concurrent request and `0.5` to `1.0` seconds spacing.

### Thread-Aware Context Budgeting

If your upstream TU-BS deployment has a hard per-request limit such as `10k` prompt tokens, the wrapper can keep requests small while still preserving longer conversations through TU-BS thread reuse.

```yaml
environment:
  - TUBS_MAX_PROMPT_TOKENS=9000
  - TUBS_THREAD_PROMPT_TOKENS=3000
  - TUBS_KEEP_LAST_TURNS=8
  - TUBS_COMPACT_SUMMARY_CHARS=4000
  - TUBS_THREAD_SUMMARY_CHARS=1200
  - TUBS_APPROX_CHARS_PER_TOKEN=4
```

- `TUBS_MAX_PROMPT_TOKENS` is the approximate prompt budget for requests that do not yet have a cached TU-BS thread.
- `TUBS_THREAD_PROMPT_TOKENS` is the smaller working-set budget used once the wrapper can rely on an existing TU-BS thread.
- `TUBS_KEEP_LAST_TURNS` controls how many recent non-system messages are preserved before older context is compacted.
- `TUBS_COMPACT_SUMMARY_CHARS` controls how much room is available for stateless summary replay.
- `TUBS_THREAD_SUMMARY_CHARS` controls the compact bridge summary size when a TU-BS thread already exists.
- `TUBS_APPROX_CHARS_PER_TOKEN` is the coarse estimator used for budget enforcement.

For a `10k` upstream cap, the shipped defaults are a good starting point:
- first request: up to about `9k` prompt tokens
- follow-up requests on the same TU-BS thread: up to about `3k` prompt tokens plus compacted bridge context

This keeps the latest task intact, avoids blunt truncation, and leans on TU-BS thread history for the older conversation state.

### 2. Python Demo: Connecting via the `openai` module

Once the server is running, use standard tooling without needing any custom code.

```python
from openai import OpenAI

# Initialize the official OpenAI client pointing to the local gateway
client = OpenAI(
    base_url="http://localhost:8000/v1",
    # To get your TUBS KI-Toolbox API key (Bearer token), navigate to:
    # https://ki-toolbox.tu-braunschweig.de/authenticationToken/show
    api_key="your_tubs_bearer_token_here" 
)

# Send a chat completion request
response = client.chat.completions.create(
    model="gpt-4o", # Use one of the supported models below
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello! Can you help me with a task?"}
    ]
)

# Print the generated response
print(response.choices[0].message.content)
```

## Supported Models

The wrapper automatically determines routing (Cloud vs On-Premise) based on the exact model string you supply. The following tables list the exact model string identifiers you must use as the `model` parameter in your API calls.

### Cloud Models (External)

| OpenAi Model Name Parameter |
| :--- |
| `gpt-5.4` |
| `gpt-5.2` |
| `gpt-5.1` |
| `gpt-5` |
| `o4-mini` |
| `o3` |
| `o3-mini` |
| `o1` |
| `gpt-4.1` |
| `gpt-4o` |
| `gpt-4o-mini` |
| `gpt-4-turbo` |
| `gpt-4` |
| `gpt-3.5-turbo-0125` |

### Anthropic Mapping

If you are using Anthropic SDKs (which default ask for `claude-3-opus-20240229` or `claude-3-5-sonnet-latest`), the API wrapper will automatically translate these. By default, it maps Opus to `gpt-5.4`, Sonnet to `gpt-4o` or `o3`, and Haiku to `gpt-4o-mini` or `o4-mini`. 
You can overwrite these defaults precisely in your `docker-compose.yml` via the environment variable `ANTHROPIC_MODEL_MAP`:
```yaml
environment:
  - ANTHROPIC_MODEL_MAP={"claude-3-opus-20240229": "gpt-5.4", "claude-3-5-sonnet-20241022": "gpt-4o"}
```
You can query `GET /v1/models` natively to see the active merged mapping target list!

### Local Models (On-Premise Institute Run)

| OpenAi Model Name Parameter |
| :--- |
| `OpenAI/GPT-OSS-120B` |
| `Qwen/Qwen3-30B-A3B` |
| `Qwen/Qwen2.5-Coder-32B-Instruct` |
| `Microsoft/Phi-4` |
| `mistralai/Mistral-Small-24B-Instruct-2501` |
| `mistralai/Magistral-Small-2509` |

## Acknowledgments / Reference

A big thank you to TU Braunschweig (TUBS) for providing the KI-Toolbox service that powers this functionality.

Special thanks and credit to the original API integrations developed in:
[python-tubskitb](https://git.rz.tu-bs.de/ias/python-tubskitb)
