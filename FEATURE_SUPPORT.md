# Feature Support Checklist

This document tracks feature parity of the TU-BS KI-Toolbox API Wrapper across both the **OpenAI** and **Anthropic** API interfaces.

---

## OpenAI API (`/v1/chat/completions`)

### Core Chat Features
| Feature | Status | Notes |
|---|---|---|
| System / User / Assistant Roles | ✅ Supported | Mapped to `customInstructions` and `prompt` |
| Developer Role | ✅ Supported | Treated identically to `system` |
| Vision (Multimodal) | ✅ Supported | Base64 data URIs → `multipart/form-data` |
| Streaming (SSE) | ✅ Supported | NDJSON → OpenAI SSE with `[DONE]` |
| Stop Sequences | ✅ Supported | Via `ENABLE_STOP_TRUNCATION` env var |
| Generation Params (`temperature`, `top_p`, etc.) | ⚠️ Accepted | Gracefully ignored — TU-BS API lacks these controls |

### Advanced Control & Formatting
| Feature | Status | Notes |
|---|---|---|
| JSON Mode (`response_format: json_object`) | ✅ Supported | Prompt injection |
| Structured Outputs (`json_schema`) | ✅ Supported | Schema injected into prompt |
| Reproducible Outputs (`seed`) | ❌ Unsupported | Gracefully ignored |
| Logprobs | ❌ Unsupported | TU-BS API does not expose token probabilities |

### Agentic Capabilities
| Feature | Status | Notes |
|---|---|---|
| Function Calling / Tool Use | ✅ Supported | XML-based extraction with escape-tolerant regex |
| Parallel Tool Calling | ✅ Supported | `re.finditer` parses multiple `<tool_call>` XML blocks |
| Tool Role Backflow | ✅ Supported | Flattened as `[Tool Result]: ...` in prompt |
| Reasoning Output (`<thought>` tags) | ✅ Supported | Extracted to `reasoning` / `reasoning_content` fields |

### Model Discovery
| Feature | Status | Notes |
|---|---|---|
| `GET /v1/models` | ✅ Supported | Returns Cloud + Local + Anthropic-mapped models |

---

## Anthropic API (`/v1/messages`)

### Core Chat Features
| Feature | Status | Notes |
|---|---|---|
| System Prompt (string or block array) | ✅ Supported | Mapped to `customInstructions` |
| User / Assistant Roles | ✅ Supported | Compiled to prompt string |
| Vision (base64 image blocks) | ✅ Supported | Native `source.type: base64` → binary extraction |
| Streaming (Message Events) | ✅ Supported | `message_start` → `content_block_delta` → `message_stop` |
| Stop Sequences | ✅ Supported | Via `ENABLE_STOP_TRUNCATION` env var |
| `max_tokens` | ⚠️ Accepted | Gracefully ignored — TU-BS API lacks this control |

### Agentic Capabilities
| Feature | Status | Notes |
|---|---|---|
| Tool Use (`input_schema`) | ✅ Supported | XML-based extraction, same as OpenAI path |
| Parallel Tool Use | ✅ Supported | Multiple `tool_use` content blocks emitted |
| Tool Result Backflow | ✅ Supported | `tool_result` blocks flattened into prompt |

### Model Mapping
| Feature | Status | Notes |
|---|---|---|
| Anthropic model aliases | ✅ Supported | `ANTHROPIC_MODEL_MAP` env var or built-in defaults |

---

## Beyond Chat
| Feature | Status | Notes |
|---|---|---|
| Embeddings (`/v1/embeddings`) | ❌ Unsupported | Not available via TU-BS API |
| Text-to-Speech (`/v1/audio/speech`) | ❌ Unsupported | Not available via TU-BS API |
| Speech-to-Text (`/v1/audio/transcriptions`) | ❌ Unsupported | Not available via TU-BS API |
| Image Generation (`/v1/images/generations`) | ❌ Unsupported | Not available via TU-BS API |
| Assistants API / Threads | ⚠️ Partial | TU-BS supports `thread` param but wrapper hardcodes `null` |
