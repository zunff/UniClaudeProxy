# 计费统计 + 缓存命中检测（先做 ds）

## Context（为什么做）

当前 UniClaudeProxy **没有任何计费/用量统计**：`usage` 只是从上游响应解析后透传给客户端，仅有一条 `logger.info` 的 per-request 日志，无聚合、无持久化。

更关键的问题：ds 走 OpenAI 路径，而 [openai\_to\_anthropic.py:150-151](file:///c:/Users/User/Documents/code/UniClaudeProxy/app/converters/openai_to_anthropic.py#L150) 把 `cache_creation_input_tokens` / `cache_read_input_tokens` **硬编码为 0**，DeepSeek 上游返回的 `usage.prompt_cache_hit_tokens` / `prompt_cache_miss_tokens` 被**直接丢弃**。所以"请求是否命中缓存"这个能力现在根本不存在，必须先把这层解析补上。

目标：

1. 解析并透出 ds（及所有 OpenAI 兼容 provider）的缓存命中 token 数；
2. 记录全部 provider 的 token 用量，对 ds 按国内官方价（CNY）算成本；
3. 新增 `GET /stats` 端点查看总量/命中率/最近记录，同时写 `logs/billing.jsonl` 持久化。

定价依据：DeepSeek 官方国内价（CNY/百万 token）。8/17 起启用峰谷定价，因此价目表支持峰谷自动切换（按北京时间）。

## 设计概览

新增 `app/billing.py` 模块（记录器 + 内存聚合 + JSONL 持久化 + 成本计算），在 `main.py` 的响应最终化点注入记录调用；修复 converter 让缓存字段在流式/非流式都正确填充；config.json 增加 `billing` 段；新增 `/stats` 端点。

记录范围：**全部 provider 记 token 用量**，但只有配置了价目表的路由（当前仅 `beyondpower/ds`）会算成本，其余 cost=null。基础设施通用，后续给其它 provider 加价目表即可扩展。

## 修改清单

### 1. 新建 `app/billing.py`

核心组件：

* `BillingConfig`：从 config 的 `billing` 段读取（enabled、log\_file、prices 表）。

* `record(route, anthropic_model, usage, is_stream, success, latency_ms)`：

  * 组装一条记录 `{ts, provider_name, model_id, anthropic_model, is_stream, success, input_tokens, output_tokens, cache_read_tokens, cache_miss_tokens, cost, currency}`；

  * 追加写到 `logs/billing.jsonl`（`encoding="utf-8"`，每行一个 JSON，`with open(..., "a")` 即可，足够快）；

  * 更新内存聚合 `_stats`（总请求数、各类 token 累计、成本累计、按 `provider/model` 分桶、命中请求数/命中 token 数）；

  * 用 `threading.Lock` 保护写操作。

* `_compute_cost(provider_name, model_id, input_tokens, output_tokens, cache_read_tokens)`：

  * 查价目表 `prices["{provider}/{model}"]`；找不到 → cost=None；

  * 缓存命中部分按 `input_cached` 单价，未命中部分 `(input - cache_read)` 按 `input` 单价，输出按 `output` 单价；

  * 支持峰谷：价目表可给 `peak`/`offpeak` 两套，按北京时间（UTC+8）小时判断（高峰 9-12、14-18）；只给一套则不分时。

* `get_stats()`：返回内存聚合快照（总量 + 分模型 + 命中率 + 最近 N 条）。

* `reset_stats()`：清内存聚合（可选，给 `DELETE /stats` 用；不影响 jsonl 文件）。

成本公式：

```
cache_miss = max(input_tokens - cache_read_tokens, 0)
cost = cache_read_tokens/1e6 * price_cached + cache_miss/1e6 * price_input + output_tokens/1e6 * price_output
```

### 2. `app/converters/openai_to_anthropic.py` — 补齐缓存字段解析

**非流式** `from_openai_chat_response`（约 [line 137-153](file:///c:/Users/User/Documents/code/UniClaudeProxy/app/converters/openai_to_anthropic.py#L137)）：

* 从 `usage` 读 DeepSeek 字段：`prompt_cache_hit_tokens`、`prompt_cache_miss_tokens`；同时兼容 OpenAI 风格 `prompt_tokens_details.cached_tokens`（取较大者作为 cache\_read）。

* `cache_read_input_tokens = hit_tokens`，`cache_creation_input_tokens = 0`（DeepSeek 无独立"写入"语义，保持 0 准确）。

* `from_openai_responses_response`（约 line 233-247）同步处理。

**流式** `stream_openai_chat_to_anthropic`：

* 增加 `input_tokens`、`cache_read_tokens` 局部变量；

* 在 usage chunk 处（约 [line 499-501](file:///c:/Users/User/Documents/code/UniClaudeProxy/app/converters/openai_to_anthropic.py#L499)）除 `completion_tokens` 外，再读 `prompt_tokens`、`prompt_cache_hit_tokens`；

* 扩展 `_build_message_delta_event`（[line 411](file:///c:/Users/User/Documents/code/UniClaudeProxy/app/converters/openai_to_anthropic.py#L411)）签名增加 `input_tokens=0, cache_read_tokens=0, cache_creation_tokens=0`，写入 `usage` 字段——这样 main.py 流式包装器和 force-stream 路径都能从 `message_delta` 读到完整 usage（force-stream 路径 [main.py:503-507](file:///c:/Users/User/Documents/code/UniClaudeProxy/app/main.py#L503) 已经在读 `u["input_tokens"]`，天然兼容）。

说明：cache 解析对所有 OpenAI 兼容 provider 生效（缺字段默认 0，无副作用），ds 直接受益。

### 3. `app/main.py` — 注入计费记录钩子

在三个响应最终化点调用 `billing.record(...)`：

* **非流式** `_handle_non_streaming`：在 [line 611-618](file:///c:/Users/User/Documents/code/UniClaudeProxy/app/main.py#L611) 的 `logger.info("Response ...")` 处，扩展日志加 `cache_read` / `cache_hit` 字段，并调用 `billing.record(route, anthropic_model, anthropic_response["usage"], is_stream=False, success=True, latency_ms=...)`。错误分支（502/500）也记录 `success=False`。

* **流式** `_handle_streaming` 的 `event_generator`（[line 638-681](file:///c:/Users/User/Documents/code/UniClaudeProxy/app/main.py#L638)）：包装 converter 迭代，解析每个 SSE event 的 `message_start`/`message_delta` 累积 usage；在生成器 `finally` 块里调用 `billing.record(...)`（`is_stream=True`）。异常分支记 `success=False`。

* **force-stream** `_handle_force_stream_non_streaming`（[line 532](file:///c:/Users/User/Documents/code/UniClaudeProxy/app/main.py#L532)）：usage 已在手，直接 `billing.record(...)`。

latency：在 handler 入口记 `time.perf_counter()`，最终化时算差值。

新增端点（放在 `/health` 附近，[line 684](file:///c:/Users/User/Documents/code/UniClaudeProxy/app/main.py#L684)）：

```python
@app.get("/stats")
async def stats(): return billing.get_stats()

@app.delete("/stats")
async def reset_stats(): billing.reset_stats(); return {"reset": True}
```

（受现有 `LocalOnlyMiddleware` 保护，仅本地可访问。）

### 4. `config.json` — 新增 `billing` 段

```json
"billing": {
  "enabled": true,
  "log_file": "logs/billing.jsonl",
  "prices": {
    "beyondpower/ds": {
      "currency": "CNY",
      "model": "deepseek-v4-pro",
      "peak": { "input": 9.0, "input_cached": 0.30, "output": 27.0 },
      "offpeak": { "input": 4.5, "input_cached": 0.15, "output": 13.5 },
      "peak_hours": [[9,12],[14,18]]
    }
  }
}
```

默认值用 DeepSeek V4-pro 8/17 起的峰谷官方价（CNY/百万 token）。`peak_hours` 按北京时间。若实际网关服务的是 V4-flash 或价格不同，用户改这一段即可。未列出的 provider 仅记 token、不算成本。

### 5. `app/config.py` — 解析 billing 段

新增 `BillingPriceConfig` / `BillingConfig` Pydantic 模型 + 在 `load_config` 里挂到 `cfg.billing`（可选字段，缺省 `enabled=False`，保证向后兼容）。`ResolvedRoute` 无需改动（已有 `provider_name`/`model_id`）。

## 默认定价说明（DeepSeek 国内官方价，CNY/百万 token）

| 模型       | 时段             | 输入(未命中) | 输入(命中) | 输出    |
| -------- | -------------- | ------- | ------ | ----- |
| V4-pro   | 高峰(9-12,14-18) | ¥9      | ¥0.30  | ¥27   |
| V4-pro   | 闲时             | ¥4.5    | ¥0.15  | ¥13.5 |
| V4-flash | 高峰             | ¥3      | ¥0.10  | ¥9    |
| V4-flash | 闲时             | ¥1.5    | ¥0.05  | ¥4.5  |

默认配 V4-pro 峰谷价。`beyondpower/ds` 实际对应哪个 DeepSeek 模型需用户确认（可在 config 改 model 名与价目）。

## `/stats` 返回示例

```json
{
  "totals": { "requests": 128, "success": 125, "input_tokens": 234100, "output_tokens": 58200,
              "cache_read_tokens": 187300, "cache_miss_tokens": 46800, "cost": 3.42, "currency": "CNY" },
  "cache": { "hit_requests": 96, "hit_rate": 0.75, "cached_token_ratio": 0.80 },
  "by_model": { "beyondpower/ds": { "requests": 128, "cost": 3.42, "input_tokens": 234100, ... } },
  "recent": [ { "ts": "...", "provider": "beyondpower", "model": "ds", "input_tokens": 1820,
                "cache_read_tokens": 1500, "output_tokens": 420, "cost": 0.027, "stream": true } ]
}
```

## 验证

1. **启动**：`python -m app.main`（或现有 Run.sh），确认无导入错误，`/health` 200。
2. **非流式 ds 请求**：用 Claude Code 或 curl 向 `/v1/messages` 发一个 `model=claude-sonnet-5` 的非流式请求。

   * 检查 `logs/billing.jsonl` 新增一条记录，`cache_read_tokens` 反映上游 `prompt_cache_hit_tokens`（连发两次同样请求，第二次应看到命中）。

   * `GET /stats` 返回的 `totals`/`by_model["beyondpower/ds"]` 数字递增，`cache.hit_rate` 合理。

   * 主日志 `Response [provider=beyondpower]` 行包含 `cache_read` / `cache_hit` 字段。
3. **流式 ds 请求**：发 `stream=true` 请求，确认流式结束后 `billing.jsonl` 也有一条记录且 `input_tokens`/`cache_read_tokens` 非 0（验证流式 usage 捕获修复生效）。
4. **成本校验**：手算一笔 `cache_read/1e6 * 0.15 + miss/1e6 * 4.5 + output/1e6 * 13.5`（闲时）与 `/stats` 的 cost 一致；高峰时段发请求验证切到 peak 价。
5. **其他 provider**：发一个 `claude-opus-5`（opencode）请求，确认 jsonl 有记录但 `cost=null`（无价目表），`/stats` by\_model 含该 provider。
6. **向后兼容**：`billing.enabled=false` 或缺该段时，所有计费逻辑静默跳过，不影响现有请求。

