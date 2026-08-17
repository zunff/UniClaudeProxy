import json
import threading
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, Field


class ModelConfig(BaseModel):
    """Configuration for a specific model within a provider.

    Attributes:
        name: str - Human-readable model name.
        upstream_model_id: str | None - Override model ID sent to provider (when config key differs).
        responses: bool - Whether to use the /v1/responses endpoint.
        use_react: bool - Whether the model supports ReAct-style tool use.
        inject_context: bool - Inject system prompt and tool summary into input messages.
        force_stream: bool - Provider always returns SSE; consume internally for non-stream.
        upstream_system: bool - Provider forces its own system prompt and tools.
        tool_mapping: dict[str, str] - Map upstream tool names to Claude Code tool names.
        reasoning: dict[str, Any] - Reasoning config e.g. {"effort": "high", "summary": "auto"}.
        truncation: str | None - Truncation strategy ("auto", "disabled", or None).
        text: dict[str, str] - Text config e.g. {"verbosity": "low"}.
        max_output_tokens: int | None - Max output tokens override for this model.
        parallel_tool_calls: bool | None - Enable parallel tool calls.
        image_mode: str - How to handle images: "input_image", "save_and_ref", or "strip".
        image_dir: str | None - Directory to save images when using save_and_ref mode.
        system_replacements: dict[str, str] - String replacements applied to system prompt (key=target, value=replacement).
        extra_body: dict[str, Any] - Extra parameters to include in the request body.
        omit_tool_choice: bool - Omit tool_choice for upstreams that reject it.
    """

    name: str
    upstream_model_id: Optional[str] = None
    responses: bool = False
    use_react: bool = False
    inject_context: bool = False
    force_stream: bool = False
    upstream_system: bool = False
    tool_mapping: dict[str, str] = Field(default_factory=dict)
    reasoning: dict[str, Any] = Field(default_factory=dict)
    truncation: Optional[str] = None
    text: dict[str, str] = Field(default_factory=dict)
    max_output_tokens: Optional[int] = None
    parallel_tool_calls: Optional[bool] = None
    image_mode: str = "input_image"
    image_dir: Optional[str] = None
    system_replacements: dict[str, str] = Field(default_factory=dict)
    extra_body: dict[str, Any] = Field(default_factory=dict)
    omit_tool_choice: bool = False


class ProviderConfig(BaseModel):
    """Configuration for a provider (OpenAI, Gemini, etc.).

    Attributes:
        provider_type: str - The type of provider (openai, claude, gemini).
        api_key: str - API key for authentication.
        base_url: str - Base URL for the provider's API.
        headers: dict[str, str] - Extra headers to include in requests.
        models: dict[str, ModelConfig] - Available models for this provider.
    """

    provider_type: str
    api_key: str = ""
    api_keys: list[str] = Field(default_factory=list)
    base_url: str
    headers: dict[str, str] = Field(default_factory=dict)
    models: dict[str, ModelConfig] = Field(default_factory=dict)

    def resolved_api_keys(self) -> list[str]:
        """Return the list of API keys, falling back to single api_key for backward compat."""
        if self.api_keys:
            return self.api_keys
        if self.api_key:
            return [self.api_key]
        return []


class ServerConfig(BaseModel):
    """Server binding configuration.

    Attributes:
        host: str - Host address to bind to.
        port: int - Port number to listen on.
        local_only: bool - Restrict access to localhost connections only.
    """

    host: str = "127.0.0.1"
    port: int = 9223
    local_only: bool = True


class UpstreamStreamTimeoutConfig(BaseModel):
    """First-byte timeout config for streaming upstream requests."""

    first_byte_timeout_ms: int = 8000


class UpstreamNonStreamTimeoutConfig(BaseModel):
    """First-byte timeout config for non-streaming upstream requests."""

    first_byte_timeout_ms: int = 20000


class UpstreamRetryConfig(BaseModel):
    """Retry strategy for upstream requests."""

    max_attempts: int = 3
    interval_ms: int = 1000
    total_timeout_ms: int = 45000


class UpstreamConfig(BaseModel):
    """Global upstream timeout and retry configuration."""

    enabled: bool = True
    stream: UpstreamStreamTimeoutConfig = Field(default_factory=UpstreamStreamTimeoutConfig)
    non_stream: UpstreamNonStreamTimeoutConfig = Field(default_factory=UpstreamNonStreamTimeoutConfig)
    retry: UpstreamRetryConfig = Field(default_factory=UpstreamRetryConfig)
    disabled_routes: list[str] = Field(default_factory=list)


class BillingConfig(BaseModel):
    """Billing / usage statistics configuration.

    Attributes:
        enabled: bool - Master switch. When False, all billing logic is skipped.
        db_file: str - SQLite database path for per-request billing records.
        log_file: str - (Deprecated) legacy JSONL path. Kept only so old
            configs don't break; new records go to SQLite. On first run,
            existing JSONL is migrated into the database and renamed to
            `<log_file>.migrated`.
        prices: dict - Named price tables (key = price name, e.g.
            "deepseek-v4-flash"). Each entry may carry peak/offpeak tiers
            (per 1M tokens, in the table's own currency) plus peak_hours
            (Beijing time). Official USD list prices stay in USD; billing
            converts to CNY via fx_to_cny at record time.
        price_bindings: dict - Maps "provider/model_id" route keys to price
            table names. Multiple routes can share the same price table.
            Routes without a binding record token usage but cost=None.
        fx_to_cny: dict - FX rates keyed by ISO currency (e.g. {"USD": 7.2})
            used to convert non-CNY unit prices into CNY for stored costs
            and dashboard totals.
        retention_days: int - Auto-cleanup retention window in days. Records
            older than this are deleted by the background scheduler. Set to
            0 to disable auto-cleanup (keep forever).
        cleanup_hour: int - Beijing-time hour (0-23) at which the daily
            cleanup job runs. Default 3 (3 AM).
    """

    enabled: bool = False
    db_file: str = "logs/billing.db"
    log_file: str = "logs/billing.jsonl"
    prices: dict[str, Any] = Field(default_factory=dict)
    price_bindings: dict[str, str] = Field(default_factory=dict)
    fx_to_cny: dict[str, float] = Field(default_factory=lambda: {"USD": 7.2})
    retention_days: int = 90
    cleanup_hour: int = 3


class AppConfig(BaseModel):
    """Root application configuration.

    Attributes:
        server: ServerConfig - Server binding settings.
        models: dict[str, str | list[str] | dict[str, int]] - Mapping of Anthropic model names to provider/model identifiers.
        providers: dict[str, ProviderConfig] - Provider configurations keyed by provider name.
    """

    server: ServerConfig = Field(default_factory=ServerConfig)
    models: dict[str, str | list[str] | dict[str, int]] = Field(default_factory=dict)
    providers: dict[str, ProviderConfig] = Field(default_factory=dict)
    upstream: UpstreamConfig = Field(default_factory=UpstreamConfig)
    billing: BillingConfig = Field(default_factory=BillingConfig)
    price_bindings: dict[str, str] = Field(default_factory=dict)


class ResolvedRoute:
    """Resolved routing information for a request.

    Attributes:
        provider_name: str - Name of the provider in config.
        provider: ProviderConfig - Full provider configuration.
        provider_type: str - The provider type (openai, gemini, claude).
        model_id: str - The model identifier to send to the provider.
        model_config: ModelConfig - Model-specific configuration flags.
        use_responses: bool - Whether to use the /v1/responses endpoint.
        endpoint_url: str - Fully resolved endpoint URL (OpenAI providers only).
    """

    def __init__(
        self,
        provider_name: str,
        provider: ProviderConfig,
        model_id: str,
        model_config: ModelConfig,
        upstream_config: UpstreamConfig,
    ):
        """Initialize a ResolvedRoute.

        Args:
            provider_name: str - Name of the provider in config.
            provider: ProviderConfig - Full provider configuration.
            model_id: str - The model identifier to send to the provider.
            model_config: ModelConfig - Model-specific configuration flags.
        """
        self.provider_name = provider_name
        self.provider = provider
        self.provider_type = provider.provider_type
        self.model_id = model_config.upstream_model_id or model_id
        self.model_config = model_config
        self.use_responses = model_config.responses
        self.inject_context = model_config.inject_context
        self.force_stream = model_config.force_stream
        self.upstream_system = model_config.upstream_system
        self.tool_mapping = model_config.tool_mapping
        self.reasoning = model_config.reasoning
        self.truncation = model_config.truncation
        self.text = model_config.text
        self.max_output_tokens = model_config.max_output_tokens
        self.parallel_tool_calls = model_config.parallel_tool_calls
        self.image_mode = model_config.image_mode
        self.image_dir = model_config.image_dir
        self.extra_body = model_config.extra_body
        self.omit_tool_choice = model_config.omit_tool_choice
        self.upstream_config = upstream_config

        base = provider.base_url.rstrip("/")
        if self.provider_type == "claude":
            self.endpoint_url = f"{base}/messages"
        elif self.provider_type == "gemini":
            self.endpoint_url = base
        elif self.use_responses:
            self.endpoint_url = f"{base}/responses"
        else:
            self.endpoint_url = f"{base}/chat/completions"

    def build_headers(self) -> dict[str, str]:
        """Build the full set of headers for the outgoing request.

        Uses round-robin selection across configured api_keys for load balancing.

        Returns:
            dict[str, str] - Merged headers with authorization.
        """
        keys = self.provider.resolved_api_keys()
        if keys:
            with _counter_lock:
                idx = _key_counters.get(self.provider_name, 0)
                selected_key = keys[idx % len(keys)]
                _key_counters[self.provider_name] = idx + 1
        else:
            selected_key = ""

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {selected_key}",
            "User-Agent": "codex_cli_rs/0.47.0 (Windows 10.0.19043; x86_64)",
            "originator": "codex_cli_rs",
        }
        headers.update(self.provider.headers)
        return headers


_config: Optional[AppConfig] = None
_config_path: Optional[str] = None
_key_counters: dict[str, int] = {}
_model_counters: dict[str, int] = {}
_counter_lock = threading.Lock()

_GLOBAL_UPSTREAM_KEYS = ("enabled", "stream", "non_stream", "retry")
_GLOBAL_BILLING_KEYS = ("enabled", "db_file", "log_file", "retention_days", "cleanup_hour")


def config_path() -> str:
    """Return the resolved path to config.json.

    Returns:
        str - Absolute path to the config file.
    """
    return _config_path or str(Path(__file__).parent.parent / "config.json")


def global_path(cfg_path: Optional[str] = None) -> str:
    """Return the path to global.json (sibling of config.json)."""
    return str(Path(cfg_path or config_path()).with_name("global.json"))


def prices_path() -> str:
    """Return the resolved path to prices.json (separate from config.json)."""
    return str(Path(config_path()).parent / "prices.json")


def _load_json_file(path: str) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {}
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {}


def _overlay_dict(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    """Shallow-merge overlay onto base; nested dicts are merged one level."""
    out = dict(base)
    for k, v in overlay.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            merged = dict(out[k])
            merged.update(v)
            out[k] = merged
        else:
            out[k] = v
    return out


def merge_config_files(
    global_raw: dict[str, Any],
    local_raw: dict[str, Any],
) -> dict[str, Any]:
    """Merge commitable global.json with local config.json.

    global.json owns server / billing infra / upstream timeouts.
    config.json owns models, providers, price_bindings, and upstream.disabled_routes.
    Leftover global keys still present in config.json overlay for backward compat.
    """
    g_up = dict(global_raw.get("upstream") or {})
    l_up = dict(local_raw.get("upstream") or {})
    disabled = l_up.pop("disabled_routes", None)
    g_up.pop("disabled_routes", None)
    upstream = _overlay_dict(g_up, l_up)
    if disabled is not None:
        upstream["disabled_routes"] = disabled
    else:
        upstream.setdefault("disabled_routes", [])

    server = _overlay_dict(
        dict(global_raw.get("server") or {}),
        dict(local_raw.get("server") or {}),
    )
    billing = _overlay_dict(
        dict(global_raw.get("billing") or {}),
        dict(local_raw.get("billing") or {}),
    )

    price_bindings = local_raw.get("price_bindings")
    if price_bindings is None:
        price_bindings = (local_raw.get("billing") or {}).get("price_bindings") or {}
    billing["price_bindings"] = dict(price_bindings)

    merged = dict(local_raw)
    merged["server"] = server
    merged["upstream"] = upstream
    merged["billing"] = billing
    merged["models"] = local_raw.get("models") or {}
    merged["providers"] = local_raw.get("providers") or {}
    merged["price_bindings"] = dict(price_bindings)
    return merged


def split_config_files(merged: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Split a merged config dict back into (global.json, config.json) payloads."""
    up = merged.get("upstream") or {}
    billing = merged.get("billing") or {}
    global_out = {
        "server": merged.get("server") or {},
        "upstream": {k: up[k] for k in _GLOBAL_UPSTREAM_KEYS if k in up},
        "billing": {k: billing[k] for k in _GLOBAL_BILLING_KEYS if k in billing},
    }
    price_bindings = merged.get("price_bindings")
    if price_bindings is None:
        price_bindings = billing.get("price_bindings") or {}
    local_out = {
        "upstream": {"disabled_routes": up.get("disabled_routes") or []},
        "models": merged.get("models") or {},
        "providers": merged.get("providers") or {},
        "price_bindings": price_bindings,
    }
    return global_out, local_out


def _load_prices_file() -> dict[str, Any]:
    """Load prices.json if it exists, return prices / bindings / fx_to_cny."""
    p = Path(prices_path())
    if not p.exists():
        return {"prices": {}, "price_bindings": {}, "fx_to_cny": {}}
    try:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        fx = data.get("fx_to_cny") or {}
        return {
            "prices": data.get("prices", {}),
            "price_bindings": data.get("price_bindings", {}),
            "fx_to_cny": {str(k).upper(): float(v) for k, v in fx.items() if v is not None},
        }
    except (json.JSONDecodeError, OSError, TypeError, ValueError):
        return {"prices": {}, "price_bindings": {}, "fx_to_cny": {}}


def load_config(path: Optional[str] = None) -> AppConfig:
    """Load and parse the application configuration from disk.

    Args:
        path: Optional[str] - Path to config.json. Defaults to project root.

    Returns:
        AppConfig - Parsed application configuration.
    """
    global _config, _config_path
    if _config is not None:
        return _config

    if path is None:
        path = str(Path(__file__).parent.parent / "config.json")

    _config_path = path

    with open(path, "r", encoding="utf-8") as f:
        local_raw = json.load(f)
    if not isinstance(local_raw, dict):
        raise ValueError(f"config.json must be an object: {path}")
    global_raw = _load_json_file(global_path(path))
    raw = merge_config_files(global_raw, local_raw)

    # Merge prices from separate prices.json into billing config
    prices_data = _load_prices_file()
    billing = raw.get("billing", {})
    billing["prices"] = prices_data["prices"]
    if not billing.get("price_bindings") and prices_data.get("price_bindings"):
        billing["price_bindings"] = prices_data["price_bindings"]
        raw["price_bindings"] = prices_data["price_bindings"]
    if prices_data.get("fx_to_cny"):
        billing["fx_to_cny"] = prices_data["fx_to_cny"]
    raw["billing"] = billing

    _config = AppConfig(**raw)
    return _config


def reload_config() -> AppConfig:
    """Force-reload configuration from disk, replacing the cached config.

    Returns:
        AppConfig - Freshly parsed application configuration.
    """
    global _config
    _config = None
    return load_config(_config_path)


def resolve_route(anthropic_model: str) -> ResolvedRoute:
    """Resolve an Anthropic model name to a provider route.

    Supports both single mapping (string) and multiple mappings (list).
    When a list is provided, uses round-robin to select one.
    Supports weighted load balancing by using dictionary format: {"provider/model": weight}

    Args:
        anthropic_model: str - The model name from the incoming Anthropic request.

    Returns:
        ResolvedRoute - Resolved routing information.

    Raises:
        ValueError: If the model or provider is not found in config.
    """
    cfg = load_config()

    mapping = cfg.models.get(anthropic_model)
    if mapping is None:
        raise ValueError(f"Model '{anthropic_model}' not found in config models mapping")

    selected = ""
    # Handle weighted load balancing
    if isinstance(mapping, dict):
        # Create weighted candidates list
        weighted_candidates = []
        for candidate, weight in mapping.items():
            weighted_candidates.extend([candidate] * weight)
        
        # Round-robin selection across weighted candidates
        with _counter_lock:
            idx = _model_counters.get(anthropic_model, 0)
            selected = weighted_candidates[idx % len(weighted_candidates)]
            _model_counters[anthropic_model] = idx + 1
    # Handle traditional list format
    elif isinstance(mapping, list):
        # Round-robin selection across candidates
        with _counter_lock:
            idx = _model_counters.get(anthropic_model, 0)
            selected = mapping[idx % len(mapping)]
            _model_counters[anthropic_model] = idx + 1
    # Handle single string format
    else:
        selected = mapping

    parts = selected.split("/", 1)
    if len(parts) != 2:
        raise ValueError(f"Invalid model mapping format: '{selected}'. Expected 'provider/model'")

    provider_name, model_id = parts

    provider = cfg.providers.get(provider_name)
    if provider is None:
        raise ValueError(f"Provider '{provider_name}' not found in config providers")

    model_cfg = provider.models.get(model_id)
    if model_cfg is None:
        raise ValueError(f"Model '{model_id}' not found in provider '{provider_name}' models")

    return ResolvedRoute(
        provider_name=provider_name,
        provider=provider,
        model_id=model_id,
        model_config=model_cfg,
        upstream_config=cfg.upstream,
    )
