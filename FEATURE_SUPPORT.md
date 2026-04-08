# OpenAI API Feature Support Checklist

Based on the current state of the FastAPI wrapper codebase (`tubs_cloud_api_wrapper`) and the TU-BS KI-Toolbox API capabilities, here is the evaluation checklist for the OpenAI API features:

### 1. Core Chat Features (`/v1/chat/completions`)
- **[x] System/User/Assistant Roles**: *Fully Supported*. The wrapper cleanly maps `System` level instructions directly into the `customInstructions` property of the `TubsChatRequest`. `User` and `Assistant` roles are seamlessly concatenated into the `prompt` string recursively to support conversational history.
- **[x] Vision (Multimodal)**: *Fully Supported*. The wrapper perfectly parses base64 data URIs out of the nested `image_url` blocks and translates the HTTP request into `multipart/form-data` as required by the backend.
- **[x] Streaming (Server-Sent Events)**: *Fully Supported*. The real-time stream properly catches the local NDJSON response chunks and dynamically translates them into OpenAI-compliant SSE format, including extracting final token usage and appending the final `[DONE]` signal.
- **[ ] Generation Parameters**: *Unsupported / Needs to be ignored gracefully*. The Pydantic model (`ChatCompletionRequest`) safely accepts parameters like `temperature`, `top_p`, `max_tokens`, etc., but they are entirely dropped when constructing the `TubsChatRequest` because the TU-BS API lacks these controls.
- **[x] Stop Sequences**: *Fully Supported (via Truncator)*. By enabling `ENABLE_STOP_TRUNCATION`, the wrapper dynamically enforces stop boundaries in both standard and streaming pipelines. In standard responses, it cleanly strips the string. Mid-stream, it severs the upstream generator loop dynamically mid-yield and injects the `stop` reason to the client safely.

### 2. Advanced Control & Formatting
- **[x] JSON Mode**: *Fully Supported*. The wrapper intercepts `response_format: {"type": "json_object"}` and automatically injects strict JSON formatting instructions into the TU-BS payload using prompt injection.
- **[x] Structured Outputs**: *Fully Supported*. The wrapper parses the incoming `json_schema` from `response_format`, stringifies it, and explicitly instructs the model to adhere perfectly to the schema via prompt injection.
- **[ ] Reproducible Outputs (Seed)**: *Unsupported / Needs to be ignored gracefully*. 
- **[ ] Logprobs**: *Unsupported / Needs to be ignored gracefully*. The TU-BS upstream does not report token probabilistic probabilities.

### 3. Agentic Capabilities
- **[x] Function Calling / Tool Use**: *Fully Supported (via XML Failsafe)*. The wrapper serializes the `tools` array and injects strict XML output requirements using prompt injection. It supports both standard request parsing and real-time streaming:
  - For normal requests, it intercepts the `[DONE]` JSON chunk, parses out the `<tool_call>` XML using regex, reformats it into the OpenAI standard `tool_calls` array, and overrides `finish_reason=tool_calls`.
  - For `stream=True`, an intercepting State Machine buffers the chunks, captures the `<tool_call>` output, reformats the chunks into standard OpenAI SSE `delta: tool_calls`, and overrides `finish_reason=tool_calls`.
- **[x] Parallel Tool Calling**: *Fully Supported*. The wrapper utilizes `re.finditer` to locate and parse multiple consecutive XML tool calls from the model buffer, translating them into a singular `tool_calls` array response mimicking OpenAI parallel logic.
- **[x] Tool Role Backflow**: *Fully Supported*. The translation layer gracefully handles multi-turn `tool` role responses by flattening them cleanly into the prompt string prefixed as "Tool Result: ", feeding critical upstream execution data back to the upstream LLM logic seamlessly.
- **[x] Reasoning Output / Thoughts**: *Fully Supported*. Automatically strips `<thought>...</thought>` tags from standard content and accurately redirects generation chunks to the native OpenAI `reasoning` / `reasoning_content` delta properties.

### 4. Beyond the Chat Endpoint
- **[ ] Embeddings (`/v1/embeddings`)**: *Unsupported / Needs to be ignored gracefully*. We only have the chat router implemented.
- **[ ] Text-to-Speech (`/v1/audio/speech`)**: *Unsupported / Needs to be ignored gracefully*.
- **[ ] Speech-to-Text (`/v1/audio/transcriptions`)**: *Unsupported / Needs to be ignored gracefully*.
- **[ ] Image Generation (`/v1/images/generations`)**: *Unsupported / Needs to be ignored gracefully*.
- **[~] Assistants API / Threads**: *Partially Supported / Needs Work*. The fully stateful OpenAI Assistants ecosystem is unsupported, but there is an interesting crossover: The TU-BS KI-Toolbox API officially accepts a `thread` (string ID) parameter to continue an existing conversation natively! We are currently hardcoding `thread=None` in `app/api/routes/chat.py`, but you could hook this up down the road to support conversational memory.
