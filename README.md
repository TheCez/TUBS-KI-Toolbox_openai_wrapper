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
- **Durable Context Layer:** Adds wrapper-owned long-horizon memory with Redis hot state, Postgres + `pgvector` durable recall, and model-facing retrieval tools for better recovery from TU-BS thread loss.

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
  - TUBS_THREAD_PROMPT_TOKENS=9000
  - TUBS_USE_UPSTREAM_THREADS=true
  - TUBS_KEEP_LAST_TURNS=8
  - TUBS_COMPACT_SUMMARY_CHARS=4000
  - TUBS_THREAD_SUMMARY_CHARS=1200
  - TUBS_APPROX_CHARS_PER_TOKEN=4
```

- `TUBS_MAX_PROMPT_TOKENS` is the approximate prompt budget for requests that do not yet have a cached TU-BS thread.
- `TUBS_THREAD_PROMPT_TOKENS` is the prompt budget used once the wrapper can rely on an existing TU-BS thread. In this version, it defaults to the same ceiling as `TUBS_MAX_PROMPT_TOKENS` unless you explicitly lower it.
- `TUBS_USE_UPSTREAM_THREADS` controls whether the wrapper should actively reuse TU-BS threads for follow-up turns. It is `true` by default.
- `TUBS_KEEP_LAST_TURNS` controls how many recent non-system messages are preserved before older context is compacted.
- `TUBS_COMPACT_SUMMARY_CHARS` controls how much room is available for stateless summary replay.
- `TUBS_THREAD_SUMMARY_CHARS` controls the compact bridge summary size when a TU-BS thread already exists.
- `TUBS_APPROX_CHARS_PER_TOKEN` is the coarse estimator used for budget enforcement.

For a `10k` upstream cap, the shipped defaults are a good starting point:
- first request: up to about `9k` prompt tokens
- follow-up requests on the same TU-BS thread: also up to about `9k` prompt tokens by default, unless you deliberately lower `TUBS_THREAD_PROMPT_TOKENS`

The wrapper now also avoids early compaction when the full prompt still fits inside budget. It only starts summarizing older turns once the compiled prompt actually exceeds the configured ceiling.

Practical meaning:
- If the compiled outbound prompt is still under your configured budget, the wrapper now sends the full prompt without compacting older turns.
- Compaction is a true overflow fallback, not the default path.
- If you want compaction to begin around a `10k` upstream ceiling, set `TUBS_MAX_PROMPT_TOKENS` and `TUBS_THREAD_PROMPT_TOKENS` just under that ceiling, for example `9000`.
- The wrapper now also budgets for `customInstructions` overhead before compacting the prompt. That means tool schemas, reasoning directives, and wrapper hints are counted as part of the real upstream request budget instead of being ignored.
- Tool-call instructions are now deliberately compact. The wrapper no longer dumps full tool schemas into `customInstructions`; it keeps only a short XML contract plus a compact required-arguments summary so more real prompt context survives before compaction.
- Reasoning effort and max-output guidance are also emitted in a shorter form to reduce instruction overhead on every request.
- If TU-BS rejects a non-stream request with a conversation-level token-limit error such as `Sie haben das Token limit für dieses Gespräch überschritten.`, the wrapper now treats that as an exhausted upstream thread, drops the cached TU-BS thread for that logical conversation, clears any staged-ingestion progress tied to it, and retries once on a fresh TU-BS thread.

When `TUBS_USE_UPSTREAM_THREADS=true` and a cached TU-BS thread already exists:
- the wrapper sends only the latest live dialogue turn to TU-BS instead of replaying the full prior turn list
- system/developer instructions still stay active through `customInstructions`
- wrapper-owned durable context is used to bootstrap new or rotated TU-BS threads, but steady-state follow-ups stay as small as possible

When `TUBS_USE_UPSTREAM_THREADS=false`:
- the wrapper keeps the previous full-history replay behavior and uses prompt compaction to stay within budget

Useful related knobs:

```yaml
environment:
  - TUBS_INSTRUCTION_TOKEN_RESERVE=600
```

- `TUBS_INSTRUCTION_TOKEN_RESERVE` keeps extra headroom for request metadata and instruction overhead so the final TU-BS request is less likely to cross the true upstream limit after prompt compaction.

### Durable Context Layer

The wrapper can now keep its own app-managed memory instead of relying only on TU-BS thread replay. This gives coding agents a more reliable way to recover earlier goals, failures, file facts, and decisions after long sessions.

```yaml
environment:
  - TUBS_CONTEXT_DATABASE_URL=postgresql://postgres:postgres@postgres:5432/tubs_context
  - TUBS_CONTEXT_HOT_BACKEND=redis
  - TUBS_CONTEXT_HOT_TTL_SECONDS=21600
  - TUBS_CONTEXT_HOT_PREFIX=tubs:context:hot:
  - TUBS_CONTEXT_TOOLS_ENABLED=true
  - TUBS_CONTEXT_TOOL_LOOP_LIMIT=4
  - TUBS_REQUIRED_CONTEXT_RETRIEVALS=1
  - TUBS_CONTEXT_EMBEDDING_DIMENSIONS=64
```

How it works:
- Redis stores a small hot snapshot for each logical wrapper thread: current objective, plan, blockers, recent decisions, and recent failures.
- Postgres + `pgvector` stores durable memory records such as goals, constraints, tool failures, file facts, code summaries, and assistant decisions.
- Non-streaming chat, responses, and Anthropic requests expose wrapper-owned context tools like `search_context`, `get_context_by_ids`, and `get_thread_state` only when the wrapper already has durable state worth retrieving for that logical thread.
- The wrapper resolves those context tool calls locally, then asks the model to continue with the retrieved context, so only relevant history comes back into the prompt.
- In normal requests, these tools are presented as optional retrieval helpers and the model should answer directly when the current prompt already contains enough information.
- In overflow requests, the wrapper switches to a bounded retrieval protocol: it stores the incoming turn first, sends only compact bridge context upstream, and requires at least one wrapper context retrieval before it accepts a final answer or external tool call.
- The wrapper now also maintains a pinned state layer for exact non-semantic thread facts such as user name, assistant identity, bootstrap status, workflow status, and a compact hidden bridge summary. This pinned state is injected on every request, including reused TU-BS thread requests, so exact state does not depend on upstream thread memory alone.
- `search_context` is intended as the semantic RAG-style lookup entry point. The model can then call `get_context_by_ids` for exact records or `get_thread_state` for the current working snapshot.
- The wrapper also injects targeted planner hints from tool results: repair hints for failed edits, and completion hints for successful file writes/edits so agents are more likely to explicitly close related tasks or todos.
- Context retrieval payloads are bounded before they go back into the next model turn. Search results, exact-record fetches, and thread-state responses are truncated and capped so retrieval itself does not become the next source of prompt bloat.
- Wrapper context-tool instructions are also kept short and only added when those tools are actually exposed, so fresh chats do not pay a context-RAG instruction penalty they cannot use.

Useful related knobs:

```yaml
environment:
  - TUBS_CONTEXT_SEARCH_TOP_K_MAX=4
  - TUBS_CONTEXT_SUMMARY_CHARS=280
  - TUBS_CONTEXT_RECORD_CONTENT_CHARS=500
  - TUBS_CONTEXT_GET_BY_IDS_MAX_RECORDS=3
  - TUBS_CONTEXT_THREAD_STATE_RECENT_MESSAGES_MAX=2
```

Important behavior notes:
- Durable context is scoped per logical wrapper thread, not shared globally.
- Non-overflow requests are still ingested after the turn completes, so retrieval stays focused on prior context instead of echoing the active prompt back to the model.
- Overflow requests ingest the current turn before the first upstream TU-BS call so the model can retrieve it back through wrapper tools if compaction or thread slicing would otherwise hide important detail.
- The overflow retrieval loop is bounded by `TUBS_CONTEXT_TOOL_LOOP_LIMIT`, and the wrapper only enforces it when context actually overflowed and durable state exists. If no overflow happens, or if the thread has no retrievable durable state, the wrapper returns the answer normally without entering the loop.
- In overflow mode, the wrapper now also rejects low-information placeholder finals such as bootstrap filler or generic closure text after retrieval. It injects one more targeted retry note instead of accepting that response as the final answer.
- Pinned thread state is kept separate from semantic retrieval memory. Identity/bootstrap/workflow state is stored structurally and injected on every turn, while semantic retrieval stays focused on older facts, file history, failures, and decisions.
- Streaming requests still get the lightweight Redis-backed summary, but wrapper-owned context tools are only resolved on non-streaming requests in this version.
- TU-BS thread memory is now a secondary helper. The primary long-horizon memory layer is the wrapper-owned durable context system.

The default `docker-compose.yml` now starts a `pgvector`-enabled Postgres container and points the API at it automatically. It also runs Redis with append-only persistence and a mounted volume so hot context, staged-ingestion progress, and cached upstream thread IDs survive restarts. If Postgres is unavailable, the wrapper degrades safely to in-memory durable storage while Redis hot snapshots continue to work.

### TU-BS Threads vs Wrapper Memory

The wrapper now uses a layered approach:

- TU-BS threads:
  useful as a lightweight upstream continuity helper
- Wrapper hot context:
  fast Redis snapshot for the current objective, plan, blockers, and recent failures
- Wrapper durable context:
  Postgres + `pgvector` long-horizon memory with retrieval tools

Recommended interpretation:
- TU-BS threads are no longer the source of truth for long conversations.
- They still help with short follow-up continuity and can reduce the amount of replay needed when the upstream behaves well.
- If TU-BS thread replay or thread-side compaction is weak, the wrapper can still recover older state through durable retrieval.
- If a TU-BS thread becomes conversation-exhausted, the wrapper now automatically rotates to a fresh upstream thread for non-stream requests and relies on the wrapper-owned durable context layer to restore the most relevant state.
- Because of that, TU-BS threads are no longer critical, but they are also not useless.

Current recommendation:
- Keep TU-BS threads enabled for now.
- Treat them as a best-effort optimization, not as the foundation of memory.
- Do not lower prompt budgets just because a TU-BS thread exists unless you have measured that it helps.
- If future testing shows TU-BS threads consistently harm output quality or cause confusing context drift, they can be disabled later without losing the new wrapper-owned durable context architecture.

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
