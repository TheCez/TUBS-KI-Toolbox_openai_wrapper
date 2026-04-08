# OpenAI Compatible API Server - Mission Profile

**Objective:** Build a FastAPI wrapper that translates OpenAI API standard requests (specifically `/v1/chat/completions`) into the proprietary TU Braunschweig KI-Toolbox API format.

**Architecture:** FastAPI + Python
**Validation:** Pydantic (Strictly enforce OpenAI schemas)
**Deployment:** Docker + docker-compose
**Package Manager:** Use `uv` for dependency management.

**Subagent Roles:**
* **@dev:** Handles FastAPI routing, Pydantic data models for OpenAI requests/responses, asynchronous Python logic, payload translation, and image extraction.
* **@ops:** Handles Dockerfile creation, `docker-compose.yml`, and environment variable configuration (e.g., securely passing the API token).
* **@qa:** Writes and executes tests against the local Docker container to verify strict OpenAI JSON schema compliance (checking nested `choices`, `messages`, `usage` fields, and streaming `[DONE]` flags).

**Core Logic & Mapping Rules:**

1. **Context Loading:** * Thoroughly read the `./reference` folder before writing code. 
   * Use `reference/python-tubskitb` for primary API client logic and endpoints.
   * Check `Example_command.txt` / `gradio_example.py` for payload structure.
   * **CRITICAL FALLBACK:** If any API behavior is undocumented in the references or you are unsure how to map a specific OpenAI parameter, STOP and ask the user for clarification. Do not hallucinate endpoints or payload structures.

2. **Endpoint Routing (Local vs. External):**
   * The TU BS API has two distinct services. You must route the request based on the requested `model` string.
   * If the model is an external service (e.g., `gpt-4o`, `gpt-4o-mini`), route the POST request to: `/api/v1/chat/send`
   * If the model is an on-premise institute model, route the POST request to: `/api/v1/localChat/send`
   * *Note: Ask the user for the specific list/prefix of local models if it is not immediately clear from the reference files.*

3. **Text Request Translation:**
   * The OpenAI payload uses a `messages` array (e.g., `[{"role": "user", "content": "..."}]`).
   * You must concatenate this array into the single `"prompt"` string expected by the TU BS API, OR map it to the `"thread"` object if the API supports conversation threading that way. 

4. **Vision / Image Handling (Multipart Requests):**
   * The OpenAI vision standard passes images as base64 data URIs within the `messages` array.
   * If an image is detected in the OpenAI request:
     1. Extract the base64 string and decode it into binary.
     2. Send the request to the TU BS API as `multipart/form-data`.
     3. The text payload must be sent as a stringified JSON object under the form key `jsonBody`.
     4. The binary image must be sent under the form key `chatAttachment` (supports `.png` and `.jpg`).

5. **Authentication:**
   * Extract the Bearer token from the incoming OpenAI format request and pass it securely to the TU BS API headers (`Authorization: Bearer <Token>`), or manage it via environment variables if instructed by the user.

6. **Streaming & Response Translation (Crucial NDJSON Logic):**
   * The TU BS API streams responses using NDJSON. 
   * **If the client requests `stream: false`:** Intercept the stream, wait for the JSON chunk where `"type": "done"`, and return standard OpenAI JSON containing the final `"response"` string and `"promptTokens"`/`"responseTokens"` usage data.
   * **If the client requests `stream: true`:** You must parse the NDJSON chunks in real-time and translate them into OpenAI Server-Sent Events (SSE) format:
     * Ignore `{"type": "start", ...}` chunks (or map them to the initial role assignment chunk).
     * For `{"type": "chunk", "content": "..."}`: Translate to `data: {"choices": [{"delta": {"content": "..."}}]}\n\n` and yield immediately.
     * For `{"type": "done", ...}`: Send the final empty chunk, optionally send usage stats, and close the stream with `data: [DONE]\n\n`.

7. **Error Handling:**
   * Catch any 4xx or 5xx HTTP errors from the TU BS API and reformat them into standard OpenAI error JSON objects (e.g., `{"error": {"message": "...", "type": "server_error"}}`).