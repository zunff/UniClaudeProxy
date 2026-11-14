# 网页端 Provider（web providers）

UniClaudeProxy 除了 OpenAI / Gemini / Anthropic 三种「正经 API」外，还有一类
provider：**没有公开 API 的对话网站**。这类站点只提供网页聊天框，我们按人的方式
使用它——把 ReAct 请求压成一条消息发进聊天框，再把站点自己的流式响应取回来。

原来只有 `deepseek_web_provider.py` 一个单站点实现，现在抽成了一层通用抽象：

```
app/providers/web/
├── base.py        # 站点无关的公共件：prompt 压平、会话复用、</tool_call> 截断、Anthropic SSE
├── engine.py      # 浏览器传输（Playwright 驱动真实页面）
├── driver.js      # 注入页面的脚本：包 fetch/XHR，把站点自己的流式响应转给 Python
├── http.py        # 直连传输（用 token 直接 POST 站点接口，不需要浏览器）
├── api.py         # 对外的 Anthropic 请求/响应管线（两种传输共用）
├── registry.py    # 站点注册表 + 管理接口用到的状态/登录/账号逻辑
├── capture.py     # 命令行抓包/调试工具（加站点时用）
└── sites/
    ├── deepseek.py  # deepseek-web：浏览器传输 + DSML 工具调用兜底
    ├── hailuo.py    # hailuo-web：浏览器传输 + MiniMax Agent 事件流
    └── custom.py    # web-http：协议写在 config.json 里，无需写代码
```

旧的 `app/providers/deepseek_web_provider.py` 仍然保留，但只是一个转发壳子，
所有名字（`send_streaming`、`get_status`、`DeepSeekStreamParser`、`_states`…）都
指向新包，老调用方和测试不用改。

---

## 两种传输

| 传输 | 何时用 | 代价 | 代表站点 |
|---|---|---|---|
| `browser` | 站点请求要 JS 签名、要 cookie、要过风控 | 要开一个常驻浏览器，慢一些 | `deepseek-web`、`hailuo-web` |
| `http` | 站点接口只认一个 bearer token / api_key | 没有浏览器，和普通 API 一样快 | `web-http` |

判定标准很简单：**能不能只凭一个 token 重放站点前端的那条请求**。能，就走
`http`（省一个浏览器）；不能（要 `x-uskey` 签名、要 `msToken` 指纹），就走
`browser`，让页面自己发请求，我们只旁听。

两种传输共用同一条管线：

```
/v1/messages → app.react 压平(use_react 自动打开) → 站点一手回合
             → 站点原生工具调用标记归一化 → </tool_call> 截断
             → 极简 Anthropic SSE（一个 text block）
```

---

## 现有站点

| `provider_type` | 传输 | 状态 |
|---|---|---|
| `deepseek-web` | browser | 已在用（多账号、会话续聊、验证码等待、DSML 兜底） |
| `hailuo-web` | browser | 海螺AI / MiniMax Agent（agent.minimax.cn）。输入框是 TipTap 的 `contenteditable`（不是 textarea），答案在 `POST /minimax-cloud/api/v1/session/<sid>/message` 返回的 SSE 里，帧形如 `{"type":6,"agent_message_chunk":{"msg_content":…}}`，最后一帧 `agent_message` 会重发全文，解析器按 `msg_id` 去重 |
| `web-http` | http | 完全由 `config.json` 描述，用来接任意「小众站点 / 自建系统」 |

---

## 配置

### deepseek-web（浏览器）

```json
"deepseek-web": {
  "provider_type": "deepseek-web",
  "base_url": "https://chat.deepseek.com",
  "api_keys": ["alice", "bob"],
  "active_account": "alice",
  "models": { "deepseek-web": { "name": "DeepSeek Web", "use_react": true } }
}
```

`api_keys` 里的每一项是**账号 id**（不是密钥）：每个账号一个独立的浏览器 profile，
必须分别在那个窗口里登录。`active_account` 决定由谁服务请求，请求不会在账号间轮询
（否则会在会话中途换身份）。

环境变量：`DEEPSEEK_WEB_PROFILE_DIR` / `_CHANNEL` / `_HEADLESS` / `_THINKING` /
`_MIN_INTERVAL`，含义见 `sites/deepseek.py` 顶部。

### hailuo-web（浏览器）

```json
"hailuo-web": {
  "provider_type": "hailuo-web",
  "base_url": "https://agent.minimaxi.com",
  "api_keys": ["default"],
  "models": { "hailuo": { "name": "海螺AI（网页端）", "use_react": true } }
}
```

不用 token：登录态在浏览器 profile 里，面板点「打开登录窗口」在那个窗口登录一次即可。
几个实测结论（省得下次再踩）：

| 事 | 值 |
|---|---|
| 聊天地址 | `https://agent.minimax.cn`（`agent.minimaxi.com` 会 301 过去；`hailuoai.com` 已经是视频/生图产品，没有聊天） |
| 输入框 | `div.ProseMirror[contenteditable="true"]`（TipTap），**不是** `textarea` |
| 发送 | 聚焦后 `fill()` + `Enter`，回车即发（不用点按钮） |
| 建会话 | `POST /minimax-cloud/api/v1/agent/<id>/session` |
| 答案流 | `POST https://agent-stream.minimax.cn/minimax-cloud/api/v1/session/<sid>/message`，`text/event-stream` |
| 帧 | `data:{"type":6,"agent_message_chunk":{"msg_id",  "msg_content", "chunk_index", "finish"}}`；结尾还有一帧 `{"type":2,"agent_message"}` 带全文和 `usage` |
| 结束 | `agent_message_chunk.finish=true` 或 `{"type":"session.finish"}` |
| 干扰项 | 页面加载时就挂着一条 `text/event-stream  /api/v1/events` 的**事件总线**（会话/标题/权限事件），跟答案无关；答案只在上面那条 message 流里。别把 `stream_paths` 设成 `*`，否则一堆 analytics JSON 会灌进解析器 |

`stream_paths` 写作 `("/minimax-cloud/api/v1/session",)` —— **不要带尾斜杠**，
`driver.js` 的匹配是 `path === p || path.startsWith(p + '/') || path.endsWith(p)`，
写成 `/session/` 会拼成 `/session//` 永远匹配不上（这个坑踩过）。

### web-http（任意站点，零代码）

```json
"niche-web": {
  "provider_type": "web-http",
  "base_url": "https://chat.example.com",
  "api_key": "站点前端发出去的 token",
  "headers": { "X-App-Id": "web" },
  "web": {
    "url": "https://chat.example.com/api/chat",
    "method": "POST",
    "headers": { "X-Version": "1.0" },
    "body": {
      "model": "{{model}}",
      "stream": true,
      "messages": "{{messages}}"
    },
    "stream": {
      "frames": "sse",
      "text": "choices.0.delta.content",
      "thinking": "choices.0.delta.reasoning_content",
      "done_on": "choices.0.finish_reason",
      "chat_id": "id"
    }
  },
  "models": { "their-model": { "name": "小众站点模型", "upstream_model_id": "their-model" } }
}
```

`web` 字段：

| 字段 | 说明 |
|---|---|
| `url` / `method` | 目标接口；`url` 里可以用模板变量 |
| `headers` | 追加/覆盖请求头（默认已带 `Authorization: Bearer <token>`） |
| `body` | 请求体模板。`{{prompt}}` 压平后的整段对话、`{{messages}}` OpenAI 风格消息数组、`{{system}}` / `{{model}}` / `{{api_key}}` / `{{chat_id}}` / `{{stream}}` |
| `stream.frames` | `sse`（默认，`data:` 帧）、`jsonl`（一行一个 JSON）、`json`（非流式，整段 JSON） |
| `stream.text` | 正文的 JSON 路径，如 `choices.0.delta.content` |
| `stream.thinking` | 思考/推理文本路径（会被丢弃，只用来触发首字节计时） |
| `stream.done_on` | 该路径取到非空值即认为流结束 |
| `stream.chat_id` | 站点侧会话 id 路径，用于续聊 |
| `stream.error` | 错误文本路径，命中即抛错 |

JSON 路径支持 `a.b.0.c` 或 `["a","b",0,"c"]`。

> 顺带一提：如果你的「小众站点」其实是 **OpenAI 兼容**的（很多小厂和自建系统都是），
> 根本不需要 web provider —— 直接用现有的 `provider_type: "openai"` 配 `base_url`，
> 工具调用、缓存计费、流式转换全都有。

---

## 管理接口与面板

| 接口 | 说明 |
|---|---|
| `GET /api/web/sites` | 列出已配置的网页端 provider |
| `GET /api/web/{provider}/status` | 每个账号/ token 的登录或配置状态 |
| `POST /api/web/{provider}/login` | 打开某个账号的浏览器窗口（http 站点返回取 token 的说明） |
| `POST /api/web/{provider}/accounts` | 添加账号 id 或 token |
| `DELETE /api/web/{provider}/accounts?account=…` | 删除 |
| `PUT /api/web/{provider}/active-account` | 切换当前账号/token |

旧的 `/api/deepseek-web/*` 全部保留为别名。管理面板左侧「网页端」页面会按
`transport` 自动切换：浏览器站点显示「打开登录窗口」，直连站点显示 token 提示并
把 token 打码显示。

---

## 加一个新站点

### 路线 A：能只凭 token 重放请求（首选，零代码）

1. 登录站点，devtools 找到那条聊天请求：URL、method、headers、body；
2. 在 `config.json` 里加一个 `provider_type: "web-http"` 的 provider，把上面写进
   `web` 块，body 里把用户消息换成 `{{prompt}}` 或 `{{messages}}`；
3. 用 `python -m app.providers.web.capture <provider> --prompt "hi" --raw` 跑一次，
   对着 `--raw` 打出来的原始帧确认 `stream.*` 路径；
4. 加一条 `models` 路由映射，收工。

### 路线 B：能只凭 token 重放，但协议怪（写个小模块）

照 `sites/custom.py`：实现 `new_parser(route)`（把原始流变成 `("text", delta)` /
`("thinking", delta)`，继承 `SseJsonParser` 或 `JsonLineParser`）和
`build_http_request(...)`（返回 `(method, url, headers, body)`），最后在
`registry.py` 的 `SITE_LIST` / `RUNTIMES` 里登记。

### 路线 C：必须让页面自己发请求（浏览器传输）

> 现成的例子：`deepseek-web`、`hailuo-web`。它们的登录页都有验证码，
> 所以不要走登录流程——把日常浏览器里的 cookie 粘进面板的「用 cookie 免登录」框，
> proxy 打开页面时就是登录态。

照 `sites/deepseek.py`：填 `composer`（输入框选择器）、`stream_paths`（要旁听的
请求路径）、`probe_js`（判断是否已登录 + 输入框是否就绪）、`stop_js`（停止生成）、
`login_path`、`chat_url_marker`，再实现 `new_parser()`（以及需要时的 `new_normalizer()`
处理站点原生工具调用标记）。**发现**（新站点/解析器不匹配时）：

```bash
# 把 stream_paths 临时设成 ["*"]，或者直接看 --raw 的原始帧
python -m app.providers.web.capture <site> --prompt "hello" --raw 2>frames.log
```

`capture` 的其它用法：

```bash
python -m app.providers.web.capture --list                 # 已知站点
python -m app.providers.web.capture deepseek-web --login   # 登录一次，存进 profile
python -m app.providers.web.capture deepseek-web --prompt "hi"
python -m app.providers.deepseek_web_provider --login      # 老入口，等价
```

---

## 免费 / 小众的「直接 API Key」源

先给结论：**国内小厂绝大多数都提供了 OpenAI 兼容接口**，接这些不需要逆向网页版、不违反
ToS、也不会封号（最多被限速）。上面那张「官网 → 网页版」的辛苦活，优先度应该排在它们后面。

### 额度最大的三家

在管理后台「模型与上游配置 → 新建 Provider」里填 `provider_type: openai` + 下面的
`base_url` + 你的 key 即可，不需要命令行。

| 站点 | 额度（2026-11 口径） | base_url | key 页面 |
|---|---|---|---|
| 智谱 BigModel | 赠送 **2000 万 tokens 永久**；`glm-4.5-flash` 本身长期免费 | `https://open.bigmodel.cn/api/paas/v4` | open.bigmodel.cn/usercenter/apikeys |
| 硅基流动 | 注册 **2000 万 tokens 等价额度，永久** | `https://api.siliconflow.cn/v1` | cloud.siliconflow.cn/account/ak |
| 魔搭 ModelScope | **每天 2000 次**，每日刷新（需绑定阿里云+实名，仅体验/非商用） | `https://api-inference.modelscope.cn/v1` | modelscope.cn/my/myaccesstoken |

### 其它小厂

| 站点 | base_url | 备注 |
|---|---|---|
| 小米 MiMo | `https://api.xiaomimimo.com/v1` | OpenAI 兼容；另有 Anthropic 兼容端点（见下） |
| MiniMax（海螺AI） | `https://api.minimax.io/v1`（国内 `https://api.minimax.chat/v1`） | 海螺就是它的网页版 |
| 阶跃星辰（跃问） | `https://api.stepfun.com/v1` | `step-3.7-flash` 很便宜 |
| 百川智能（百小应） | `https://api.baichuan-ai.com/v1` | 注册送额度 |
| 商汤日日新 | `https://api.sensenova.cn/compatible-mode/v1` | 官方兼容模式 |
| 零一万物 | `https://api.lingyiwanwu.com/v1` | 官方兼容 |
| Pollinations | `https://text.pollinations.ai/openai` | 完全免 key，兜底用 |
| GitHub Models | `https://models.github.ai/inference` | 用 GitHub PAT 当 key |

`config.example.json` 里 `minimax` / `stepfun` / `baichuan` / `siliconflow` / `zhipu` /
`modelscope` / `mimo` 都是可直接改 key 的样板。

#### 小米 MiMo（两个端点，按需选）

| 用途 | base_url | provider_type |
|---|---|---|
| OpenAI 兼容（走本项目的转换层，支持多 key 轮询、缓存计费） | `https://api.xiaomimimo.com/v1` | `openai` |
| Anthropic 原生（直通，保留 thinking / 工具调用原样） | `https://api.xiaomimimo.com/anthropic/v1` | `claude` |

Anthropic 端点要自己带鉴权头（项目的直通模式不会替你加 `Authorization`）：

```json
"mimo-native": {
  "provider_type": "claude",
  "base_url": "https://api.xiaomimimo.com/anthropic/v1",
  "headers": { "x-api-key": "your-xiaomimimo-api-key", "anthropic-version": "2023-06-01" },
  "models": { "mimo-v2.6-pro": { "name": "小米 MiMo V2.6 Pro（原生）" } }
}
```

> ⚠️ 买的是 **Token Plan 套餐**（`tp-` 开头的 key）的话，官方条款写明额度**只能用在编程
> 工具里**，禁止拿去当自动化脚本 / 应用后端。要让 proxy 调它，请用按量计费（`sk-` 那类）
> 的 key，并自己算好价格。
>
> 模型名：`mimo-v2.6-pro` / `mimo-v2.6-flash`（V2.5 系列已于 2026-10-21 下线）。
> 价格按量计费、以控制台为准；`mimo-v2.6-flash` 的输出单价约 $0.28/M、pro 约 $0.87/M
> （第三方口径，未含输入与缓存价）。

**只有网页版、没有开放 API 的小站**（360 纳米AI、秘塔搜索、讯飞星火等）才需要走
`web-http`；其中讯飞星火用 WebSocket，`web-http` 覆盖不了，得单独写传输层。

### 什么时候才值得逆网页版

- 这个站点**只有网页版**、或官方 API 已经不给免费额度了（例如 DeepSeek 网页版比 API 便宜的那段时间）；
- 你想用的是网页端**独有**的能力（联网搜索、深度研究、附件解析）；
- 只是学习逆向本身。

判断顺序：**官方免费 API > 官方付费 API > 网页版**。前两者不会封号，只是限速。

### 风控相关的几个环境变量

| 变量 | 默认 | 作用 |
|---|---|---|
| `<SITE>_MIN_INTERVAL` | 3 | 同一账号两回合之间的最小间隔（秒）；**http 与 browser 传输都生效**，建议小号用 8–15 |
| `<SITE>_FAILURE_THRESHOLD` | 2 | 连续几次「硬失败」（验证码 / 429 / 403）后冷却该账号 |
| `<SITE>_COOLDOWN_MINUTES` | 30 | 冷却时长；冷却期间该账号的请求**立即失败**，不会继续打上游 |
| `<SITE>_MAX_TURNS_PER_DAY` | 未启用 | 每日回合上限（预留，尚未实现） |

`<SITE>` 是站点前缀：`DEEPSEEK_WEB_*`、`HAILUO_WEB_*`。

冷却状态会出现在 `GET /api/web/{provider}/status` 的 `cooldown_seconds_left` 字段里，
面板上也能看到。

---

## 注意

- 这类站点没有服务条款层面的「编程调用」许可，**账号风险自负**；不要拿主账号跑。
- token / cookie 等同于账号密码，别提交进 git（`config.json` 已在 `.gitignore` 里）。
- 网页端模型不会真的遵守 `</tool_call>` 停止序列，所以答案是**边流边截断**的；工具
  调用只能一轮一个（ReAct 的限制）。
- 站点改版会让解析器失效。`capture --raw` 打出来的帧是排查的第一手材料；多数失效
  只是 JSON 路径变了，`deepseek` 之外通常几分钟能修好。
