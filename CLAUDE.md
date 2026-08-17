# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Proxy Server (Backend)
- **Install dependencies**: `pip install -r requirements.txt`
- **Run server (direct)**: `python -m uvicorn app.main:app --host 127.0.0.1 --port 9223`
- **Run server (scripts)**: `Run.bat` (Windows) or `./Run.sh` (Linux/macOS)
- **Run all tests**: `pytest` or `python -m pytest`
- **Run single test file**: `pytest tests/test_billing_cost.py`
- **Run single test case**: `pytest tests/test_billing_cost.py::ComputeCostTests::test_usd_official_price_stored_as_cny`

### Admin Dashboard (Frontend)
- **Install dependencies**: `cd admin-dashboard && npm install` (or `pnpm install`)
- **Run dev server**: `cd admin-dashboard && npm run dev` (or `start-admin.bat` / `./start-admin.sh`)
- **Build**: `cd admin-dashboard && npm run build`

---

## Architecture Overview

UniClaudeProxy is a FastAPI proxy that sits between the Claude Code CLI (or any Anthropic Messages API client) and diverse LLM backends (OpenAI-compatible APIs, Google Gemini native, Anthropic passthrough).

### Request Pipeline (`POST /v1/messages`)
1. **Security**: `LocalOnlyMiddleware` enforces localhost-only access (`127.0.0.1`, `::1`, `localhost`) when `server.local_only` is true.
2. **Route Resolution (`app/config.py`)**: Maps the requested Anthropic model name to a target provider and model. Supports:
   - Single target string: `"claude-sonnet-4-5-20250929": "deepseek/deepseek-chat"`
   - Round-robin list: `["provider_a/model", "provider_b/model"]`
   - Weighted dictionary: `{"provider_a/model": 3, "provider_b/model": 1}`
   - Multiple API keys per provider via round-robin over `api_keys`.
3. **Pre-processing**:
   - `system_replacements`: Replaces identity/system prompt strings per model before sending upstream.
   - `use_react` fallback (`app/react/`): For models lacking native tool calling, strips native tool definitions, embeds XML tool schemas into the system prompt, and parses `<tool_call>` XML responses back to Anthropic `tool_use` blocks.
   - `images` (`app/utils/images.py`): Supports `input_image` (inline base64), `save_and_ref` (save to disk and pass file reference), or `strip` (text-only models).
4. **Provider Dispatch & Conversion (`app/converters/`, `app/providers/`)**:
   - **OpenAI** (`openai_provider.py`): Uses `/v1/chat/completions` or `/v1/responses` (`responses: true`). Converts tool calls, thinking tokens (`<think>` tags or reasoning summaries), and usage cache statistics (`prompt_cache_hit_tokens`).
   - **Gemini** (`gemini_provider.py`): Uses native Google Gemini API (`generateContent` / `streamGenerateContent`). Handles `thoughtSignature` preservation across turns and auto-corrects parameter casing (camelCase to snake_case via `build_tool_param_index`).
   - **Claude** (`anthropic_provider.py`): Direct passthrough to upstream Anthropic-compatible endpoints.
   - **Retry & Timeouts** (`retry_utils.py`): Global first-byte timeouts (streaming/non-streaming) and retry attempts.
5. **Streaming SSE / Response Translation**: Real-time conversion into Anthropic SSE events (`message_start`, `content_block_delta`, `message_delta`, `message_stop`).
6. **Billing & Analytics (`app/billing.py`)**:
   - Records token usage, cache hits/misses, latency, and costs to SQLite database (`logs/billing.db`).
   - Computes costs with peak/off-peak pricing (Beijing time) and converts foreign currencies (e.g. USD) to CNY via `fx_to_cny`.
   - Exposes `/stats`, `/api/stats`, and `/api/stats/recent` endpoints.

---

## Configuration Architecture

Configuration is split across three files to separate committable defaults, secrets, and pricing data:

| File | Purpose | Committed to Git |
|---|---|---|
| `global.json` | Shared server settings, upstream timeouts/retries, billing DB retention | Yes |
| `config.json` | Providers, API keys, model route mappings, disabled routes, route-to-price bindings (`price_bindings`) | No (gitignored) |
| `prices.json` | Model price tables, FX conversion rates | Optional |
| `config.example.json` | Template for single-route configuration | Yes |
| `config.example.loadbalance.json` | Template for multi-key / weighted load balancing | Yes |

`app/watcher.py` automatically hot-reloads `config.json` and `global.json` on modification without restarting the server.
