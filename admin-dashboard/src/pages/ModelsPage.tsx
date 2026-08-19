import { useEffect, useMemo, useState } from "react";
import { toast } from "sonner";
import {
  Check,
  ChevronDown,
  ChevronRight,
  Eye,
  EyeOff,
  Globe,
  Key,
  Layers,
  Link2,
  Pencil,
  Plus,
  RefreshCw,
  Save,
  Search,
  Server,
  Settings,
  Settings2,
  Sliders,
  Trash2,
  Zap,
} from "lucide-react";
import { useAdmin, type RawAppConfig } from "@/store/admin";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Input, Label } from "@/components/ui/input";
import { SelectField } from "@/components/ui/select";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { cn } from "@/lib/utils";

function ConfigField({
  label,
  value,
  onChange,
  type = "text",
  disabled,
}: {
  label: string;
  value: string | number | boolean;
  onChange: (v: string) => void;
  type?: "text" | "number" | "password" | "checkbox";
  disabled?: boolean;
}) {
  if (type === "checkbox") {
    return (
      <label className="flex items-center gap-2.5 cursor-pointer select-none">
        <input
          type="checkbox"
          checked={!!value}
          onChange={(e) => onChange(String(e.target.checked))}
          disabled={disabled}
          className="w-4 h-4 rounded border-slate-700 bg-slate-900 text-cyan-500 focus:ring-cyan-500 focus:ring-offset-0"
        />
        <span className="text-xs font-medium text-slate-300">
          {label}
        </span>
      </label>
    );
  }
  return (
    <div>
      <Label>{label}</Label>
      <Input
        type={type}
        value={String(value)}
        onChange={(e) => onChange(e.target.value)}
        disabled={disabled}
      />
    </div>
  );
}

function ServerConfigCard({
  config,
  onSave,
}: {
  config: RawAppConfig;
  onSave: (next: RawAppConfig) => Promise<boolean>;
}) {
  const [draft, setDraft] = useState<RawAppConfig>(config);
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    setDraft(config);
  }, [config]);

  const server = draft.server ?? {};
  const setServer = (k: string, v: string) =>
    setDraft({ ...draft, server: { ...server, [k]: v } });

  const save = async () => {
    setSaving(true);
    await onSave(draft);
    setSaving(false);
  };

  const dirty = JSON.stringify(draft.server) !== JSON.stringify(config.server);

  return (
    <Card className="border-cyan-500/20 hover:border-cyan-500/40">
      <CardHeader>
        <div className="flex items-center justify-between">
          <CardTitle className="text-base flex items-center gap-2 text-cyan-300">
            <Server className="w-4 h-4 text-cyan-400" />
            <span>网关监听配置</span>
          </CardTitle>
          <span className="px-2 py-0.5 rounded text-[10px] font-mono border border-cyan-500/30 bg-cyan-500/10 text-cyan-400 font-semibold">
            global.json
          </span>
        </div>
        <CardDescription>本地 HTTP 端口与安全限制</CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-3.5">
          <ConfigField
            label="监听地址 (HOST)"
            value={server.host ?? "127.0.0.1"}
            onChange={(v) => setServer("host", v)}
          />
          <ConfigField
            label="监听端口 (PORT)"
            value={server.port ?? 9223}
            onChange={(v) => setServer("port", v)}
          />
          <div className="flex items-end pb-2">
            <ConfigField
              label="local_only 仅限本地"
              type="checkbox"
              value={server.local_only ?? true}
              onChange={(v) => setServer("local_only", v)}
            />
          </div>
        </div>
        {dirty && (
          <div className="flex justify-end pt-2 border-t border-brand-borderSubtle">
            <Button variant="primary" size="sm" onClick={save} disabled={saving}>
              <Save className="w-3.5 h-3.5" />
              {saving ? "保存中..." : "保存服务器配置"}
            </Button>
          </div>
        )}
      </CardContent>
    </Card>
  );
}

function UpstreamConfigCard({
  config,
  onSave,
}: {
  config: RawAppConfig;
  onSave: (next: RawAppConfig) => Promise<boolean>;
}) {
  const [draft, setDraft] = useState<RawAppConfig>(config);
  const [saving, setSaving] = useState(false);
  const [expanded, setExpanded] = useState(false);

  useEffect(() => {
    setDraft(config);
  }, [config]);

  const up = (draft.upstream ?? {}) as Record<string, any>;
  const setUp = (k: string, v: any) =>
    setDraft({ ...draft, upstream: { ...up, [k]: v } });

  const save = async () => {
    setSaving(true);
    await onSave(draft);
    setSaving(false);
  };

  const dirty = JSON.stringify(draft.upstream) !== JSON.stringify(config.upstream);

  return (
    <Card className="border-amber-500/20 hover:border-amber-500/40">
      <CardHeader>
        <div className="flex items-center justify-between">
          <CardTitle className="text-base flex items-center gap-2 text-amber-300">
            <Zap className="w-4 h-4 text-amber-400" />
            <span>上游策略与重试配置</span>
          </CardTitle>
          <span className="px-2 py-0.5 rounded text-[10px] font-mono border border-amber-500/30 bg-amber-500/10 text-amber-400 font-semibold">
            global.json
          </span>
        </div>
        <CardDescription>首字节超时、重试策略与异常熔断</CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="flex items-center justify-between pb-1">
          <ConfigField
            label="启用上游多路由自动故障转移 (Failover)"
            type="checkbox"
            value={up.enabled ?? false}
            onChange={(v) => setUp("enabled", v === "true")}
          />
        </div>
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-3.5">
          <ConfigField
            label="流式首字节超时 (ms)"
            type="number"
            value={up.stream?.first_byte_timeout_ms ?? 30000}
            onChange={(v) =>
              setUp("stream", { ...up.stream, first_byte_timeout_ms: Number(v) })
            }
          />
          <ConfigField
            label="非流式首字节超时 (ms)"
            type="number"
            value={up.non_stream?.first_byte_timeout_ms ?? 60000}
            onChange={(v) =>
              setUp("non_stream", { ...up.non_stream, first_byte_timeout_ms: Number(v) })
            }
          />
          <ConfigField
            label="最大重试次数"
            type="number"
            value={up.retry?.max_attempts ?? 2}
            onChange={(v) =>
              setUp("retry", { ...up.retry, max_attempts: Number(v) })
            }
          />
        </div>

        <div className="pt-2 border-t border-brand-borderSubtle">
          <button
            className="flex items-center gap-1.5 text-xs text-amber-400/90 hover:text-amber-300 transition-colors font-medium"
            onClick={() => setExpanded(!expanded)}
          >
            {expanded ? (
              <ChevronDown className="w-3.5 h-3.5" />
            ) : (
              <ChevronRight className="w-3.5 h-3.5" />
            )}
            <span>查看/编辑禁用黑名单路由 ({up.disabled_routes?.length ?? 0} 条)</span>
          </button>
          {expanded && (
            <div className="mt-2">
              <textarea
                className="w-full h-24 rounded-lg border border-brand-borderSubtle bg-slate-950 p-3 text-xs font-mono text-slate-200 focus-visible:outline-none focus-visible:border-amber-500"
                placeholder="每行一个 route_key，例如 beyondpower/ds"
                value={(up.disabled_routes ?? []).join("\n")}
                onChange={(e) =>
                  setUp(
                    "disabled_routes",
                    e.target.value
                      .split("\n")
                      .map((s) => s.trim())
                      .filter(Boolean)
                  )
                }
              />
            </div>
          )}
        </div>

        {dirty && (
          <div className="flex justify-end pt-2 border-t border-brand-borderSubtle">
            <Button variant="primary" size="sm" onClick={save} disabled={saving}>
              <Save className="w-3.5 h-3.5" />
              {saving ? "保存中..." : "保存重试配置"}
            </Button>
          </div>
        )}
      </CardContent>
    </Card>
  );
}

function ProxyConfigCard({
  config,
  onSave,
}: {
  config: RawAppConfig;
  onSave: (next: RawAppConfig) => Promise<boolean>;
}) {
  const [draft, setDraft] = useState<RawAppConfig>(config);
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    setDraft(config);
  }, [config]);

  const proxy = draft.proxy ?? {};
  const setProxy = (k: string, v: any) =>
    setDraft({ ...draft, proxy: { ...proxy, [k]: v } });

  const save = async () => {
    setSaving(true);
    await onSave(draft);
    setSaving(false);
  };

  const dirty = JSON.stringify(draft.proxy) !== JSON.stringify(config.proxy);

  return (
    <Card className="border-fuchsia-500/20 hover:border-fuchsia-500/40">
      <CardHeader>
        <div className="flex items-center justify-between">
          <CardTitle className="text-base flex items-center gap-2 text-fuchsia-300">
            <Globe className="w-4 h-4 text-fuchsia-400" />
            <span>出口代理配置</span>
          </CardTitle>
          <span className="px-2 py-0.5 rounded text-[10px] font-mono border border-fuchsia-500/30 bg-fuchsia-500/10 text-fuchsia-400 font-semibold">
            global.json
          </span>
        </div>
        <CardDescription>所有上游 LLM 请求经代理转发，直连或走代理可热切换</CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <ConfigField
          label="启用代理 (enabled)，关闭即直连"
          type="checkbox"
          value={proxy.enabled ?? false}
          onChange={(v) => setProxy("enabled", v === "true")}
        />
        <ConfigField
          label="代理地址 (URL)"
          value={proxy.url ?? ""}
          onChange={(v) => setProxy("url", v)}
        />
        <p className="text-[11px] text-slate-500 leading-relaxed">
          支持 http:// 或 https://，例如 <code className="font-mono text-fuchsia-400/80">http://127.0.0.1:7890</code>。
          修改保存后 watcher 热重载，下一次请求自动生效，无需重启。
        </p>
        {dirty && (
          <div className="flex justify-end pt-2 border-t border-brand-borderSubtle">
            <Button variant="primary" size="sm" onClick={save} disabled={saving}>
              <Save className="w-3.5 h-3.5" />
              {saving ? "保存中..." : "保存代理配置"}
            </Button>
          </div>
        )}
      </CardContent>
    </Card>
  );
}

function ProviderEditDialog({
  initialName,
  initialProvider,
  existingNames,
  onSave,
  onClose,
}: {
  initialName?: string;
  initialProvider?: any;
  existingNames: string[];
  onSave: (name: string, provider: any) => Promise<boolean>;
  onClose: () => void;
}) {
  const isNew = !initialName;
  const [name, setName] = useState(initialName ?? "");
  const [providerType, setProviderType] = useState(
    initialProvider?.provider_type ?? "openai"
  );
  const [apiKey, setApiKey] = useState(initialProvider?.api_key ?? "");
  const [baseUrl, setBaseUrl] = useState(initialProvider?.base_url ?? "");

  const submit = async () => {
    if (!name.trim()) {
      toast.error("Provider 标识不能为空");
      return;
    }
    if (isNew && existingNames.includes(name.trim())) {
      toast.error("Provider 标识已存在");
      return;
    }
    const provider: any = {
      provider_type: providerType,
      api_key: apiKey,
      base_url: baseUrl,
      models: initialProvider?.models ?? {},
    };
    if (initialProvider?.headers) provider.headers = initialProvider.headers;
    const ok = await onSave(name.trim(), provider);
    if (ok) onClose();
  };

  return (
    <Dialog open onOpenChange={(o) => !o && onClose()}>
      <DialogContent className="max-w-md">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <Server className="w-5 h-5 text-cyan-400" />
            {isNew ? "新建 Provider 上游服务" : `编辑 Provider · ${initialName}`}
          </DialogTitle>
          <DialogDescription>
            配置上游服务的 Base URL、密钥凭证与协议类型
          </DialogDescription>
        </DialogHeader>
        <div className="space-y-4 font-mono text-xs">
          <div>
            <Label>Provider 唯一标识 (ID)</Label>
            <Input
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="beyondpower / gemini"
              disabled={!isNew}
            />
          </div>
          <div>
            <Label>协议类型</Label>
            <SelectField
              value={providerType}
              onValueChange={setProviderType}
              placeholder="选择协议类型"
              options={[
                { value: "openai", label: "OpenAI 兼容协议 (Chat / Completions)" },
                { value: "gemini", label: "Google Gemini 协议" },
                { value: "claude", label: "Anthropic Claude 直通协议" },
              ]}
            />
          </div>
          <div>
            <Label>API Key (密钥凭证)</Label>
            <Input
              type="password"
              value={apiKey}
              onChange={(e) => setApiKey(e.target.value)}
              placeholder="sk-..."
            />
          </div>
          <div>
            <Label>Base URL (服务网关地址)</Label>
            <Input
              value={baseUrl}
              onChange={(e) => setBaseUrl(e.target.value)}
              placeholder="https://api.openai.com/v1"
            />
          </div>
        </div>
        <DialogFooter>
          <Button variant="ghost" onClick={onClose} type="button">
            取消
          </Button>
          <Button variant="primary" onClick={submit} type="button">
            <Save className="w-4 h-4" />
            {isNew ? "创建" : "保存"}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}

// --- Comprehensive Model Config Edit Dialog ---
function ModelConfigDialog({
  providerName,
  initialModelId,
  initialConfig,
  existingModelIds,
  onSave,
  onClose,
}: {
  providerName: string;
  initialModelId?: string;
  initialConfig?: Record<string, any>;
  existingModelIds: string[];
  onSave: (modelId: string, modelCfg: Record<string, any>) => Promise<boolean>;
  onClose: () => void;
}) {
  const isNew = !initialModelId;
  const [modelId, setModelId] = useState(initialModelId ?? "");
  const [name, setName] = useState(initialConfig?.name ?? (initialModelId || ""));
  const [upstreamModelId, setUpstreamModelId] = useState(
    initialConfig?.upstream_model_id ?? ""
  );

  // Switch toggles
  const [responses, setResponses] = useState(Boolean(initialConfig?.responses));
  const [useReact, setUseReact] = useState(Boolean(initialConfig?.use_react));
  const [forceStream, setForceStream] = useState(Boolean(initialConfig?.force_stream));
  const [injectContext, setInjectContext] = useState(Boolean(initialConfig?.inject_context));
  const [upstreamSystem, setUpstreamSystem] = useState(Boolean(initialConfig?.upstream_system));
  const [omitToolChoice, setOmitToolChoice] = useState(Boolean(initialConfig?.omit_tool_choice));

  // Advanced settings
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [maxOutputTokens, setMaxOutputTokens] = useState(
    initialConfig?.max_output_tokens ? String(initialConfig.max_output_tokens) : ""
  );
  const [imageMode, setImageMode] = useState(initialConfig?.image_mode ?? "input_image");

  // JSON strings for dict objects
  const [toolMappingStr, setToolMappingStr] = useState(
    initialConfig?.tool_mapping ? JSON.stringify(initialConfig.tool_mapping, null, 2) : ""
  );
  const [systemReplacementsStr, setSystemReplacementsStr] = useState(
    initialConfig?.system_replacements
      ? JSON.stringify(initialConfig.system_replacements, null, 2)
      : ""
  );
  const [extraBodyStr, setExtraBodyStr] = useState(
    initialConfig?.extra_body ? JSON.stringify(initialConfig.extra_body, null, 2) : ""
  );

  const submit = async () => {
    const trimmedId = modelId.trim();
    if (!trimmedId) {
      toast.error("模型 ID 不能为空");
      return;
    }
    if (isNew && existingModelIds.includes(trimmedId)) {
      toast.error("该模型 ID 在此 Provider 下已存在");
      return;
    }

    let parsedToolMapping = {};
    if (toolMappingStr.trim()) {
      try {
        parsedToolMapping = JSON.parse(toolMappingStr);
      } catch (e: any) {
        toast.error("tool_mapping JSON 格式错误", { description: e.message });
        return;
      }
    }

    let parsedSystemReplacements = {};
    if (systemReplacementsStr.trim()) {
      try {
        parsedSystemReplacements = JSON.parse(systemReplacementsStr);
      } catch (e: any) {
        toast.error("system_replacements JSON 格式错误", { description: e.message });
        return;
      }
    }

    let parsedExtraBody = {};
    if (extraBodyStr.trim()) {
      try {
        parsedExtraBody = JSON.parse(extraBodyStr);
      } catch (e: any) {
        toast.error("extra_body JSON 格式错误", { description: e.message });
        return;
      }
    }

    const cfg: Record<string, any> = {
      name: name.trim() || trimmedId,
      responses,
      use_react: useReact,
      force_stream: forceStream,
      inject_context: injectContext,
      upstream_system: upstreamSystem,
      omit_tool_choice: omitToolChoice,
      image_mode: imageMode,
    };

    if (upstreamModelId.trim()) cfg.upstream_model_id = upstreamModelId.trim();
    if (maxOutputTokens.trim()) cfg.max_output_tokens = Number(maxOutputTokens);
    if (Object.keys(parsedToolMapping).length > 0) cfg.tool_mapping = parsedToolMapping;
    if (Object.keys(parsedSystemReplacements).length > 0)
      cfg.system_replacements = parsedSystemReplacements;
    if (Object.keys(parsedExtraBody).length > 0) cfg.extra_body = parsedExtraBody;

    const ok = await onSave(trimmedId, cfg);
    if (ok) onClose();
  };

  return (
    <Dialog open onOpenChange={(o) => !o && onClose()}>
      <DialogContent className="max-w-xl max-h-[85vh] overflow-y-auto scrollbar-thin">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <Sliders className="w-5 h-5 text-cyan-400" />
            {isNew ? `添加模型 · ${providerName}` : `配置模型参数 · ${providerName}/${initialModelId}`}
          </DialogTitle>
          <DialogDescription>
            自定义端点协议、工具调用模式（Responses / ReAct）及提示词规则
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-4 font-mono text-xs">
          {/* Base Info */}
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
            <div>
              <Label>模型 ID (路由标识)</Label>
              <Input
                value={modelId}
                onChange={(e) => setModelId(e.target.value)}
                placeholder="ds / glm-4.7-flash"
                disabled={!isNew}
              />
            </div>
            <div>
              <Label>上游服务模型名 (name)</Label>
              <Input
                value={name}
                onChange={(e) => setName(e.target.value)}
                placeholder="留空则与模型 ID 相同"
              />
            </div>
          </div>

          <div>
            <Label>上游模型覆写 ID (upstream_model_id, 可选)</Label>
            <Input
              value={upstreamModelId}
              onChange={(e) => setUpstreamModelId(e.target.value)}
              placeholder="覆盖发送给上游 API 实际使用的 model 字段"
            />
          </div>

          {/* Protocols & Switches */}
          <div className="p-3.5 rounded-lg border border-brand-borderSubtle bg-slate-950/60 space-y-3">
            <div className="text-xs font-semibold text-slate-300 flex items-center justify-between border-b border-brand-borderSubtle pb-1.5">
              <span>协议端点与工具调用开关</span>
              <span className="text-[10px] text-cyan-400 font-normal">PROTOCOL SWITCHES</span>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-2 gap-2.5">
              <label className="flex items-start gap-2 cursor-pointer p-1.5 rounded hover:bg-slate-900 transition-colors">
                <input
                  type="checkbox"
                  checked={responses}
                  onChange={(e) => setResponses(e.target.checked)}
                  className="mt-0.5 w-4 h-4 rounded border-slate-700 bg-slate-900 text-cyan-500"
                />
                <div>
                  <div className="text-xs font-semibold text-purple-300">responses</div>
                  <div className="text-[10px] text-slate-400">开启 /v1/responses 端点</div>
                </div>
              </label>

              <label className="flex items-start gap-2 cursor-pointer p-1.5 rounded hover:bg-slate-900 transition-colors">
                <input
                  type="checkbox"
                  checked={useReact}
                  onChange={(e) => setUseReact(e.target.checked)}
                  className="mt-0.5 w-4 h-4 rounded border-slate-700 bg-slate-900 text-cyan-500"
                />
                <div>
                  <div className="text-xs font-semibold text-amber-300">use_react</div>
                  <div className="text-[10px] text-slate-400">启用 ReAct XML 工具调用</div>
                </div>
              </label>

              <label className="flex items-start gap-2 cursor-pointer p-1.5 rounded hover:bg-slate-900 transition-colors">
                <input
                  type="checkbox"
                  checked={forceStream}
                  onChange={(e) => setForceStream(e.target.checked)}
                  className="mt-0.5 w-4 h-4 rounded border-slate-700 bg-slate-900 text-cyan-500"
                />
                <div>
                  <div className="text-xs font-semibold text-cyan-300">force_stream</div>
                  <div className="text-[10px] text-slate-400">强制转换上游 SSE 流式输出</div>
                </div>
              </label>

              <label className="flex items-start gap-2 cursor-pointer p-1.5 rounded hover:bg-slate-900 transition-colors">
                <input
                  type="checkbox"
                  checked={injectContext}
                  onChange={(e) => setInjectContext(e.target.checked)}
                  className="mt-0.5 w-4 h-4 rounded border-slate-700 bg-slate-900 text-cyan-500"
                />
                <div>
                  <div className="text-xs font-semibold text-emerald-300">inject_context</div>
                  <div className="text-[10px] text-slate-400">注入 System Prompt 提示词</div>
                </div>
              </label>

              <label className="flex items-start gap-2 cursor-pointer p-1.5 rounded hover:bg-slate-900 transition-colors">
                <input
                  type="checkbox"
                  checked={omitToolChoice}
                  onChange={(e) => setOmitToolChoice(e.target.checked)}
                  className="mt-0.5 w-4 h-4 rounded border-slate-700 bg-slate-900 text-cyan-500"
                />
                <div>
                  <div className="text-xs font-semibold text-slate-200">omit_tool_choice</div>
                  <div className="text-[10px] text-slate-400">省略 tool_choice 参数</div>
                </div>
              </label>

              <label className="flex items-start gap-2 cursor-pointer p-1.5 rounded hover:bg-slate-900 transition-colors">
                <input
                  type="checkbox"
                  checked={upstreamSystem}
                  onChange={(e) => setUpstreamSystem(e.target.checked)}
                  className="mt-0.5 w-4 h-4 rounded border-slate-700 bg-slate-900 text-cyan-500"
                />
                <div>
                  <div className="text-xs font-semibold text-slate-200">upstream_system</div>
                  <div className="text-[10px] text-slate-400">强制使用上游原生系统提示</div>
                </div>
              </label>
            </div>
          </div>

          {/* Advanced Collapsible Section */}
          <div className="pt-1">
            <button
              type="button"
              onClick={() => setShowAdvanced(!showAdvanced)}
              className="flex items-center gap-1.5 text-xs text-slate-400 hover:text-cyan-400 transition-colors"
            >
              {showAdvanced ? (
                <ChevronDown className="w-3.5 h-3.5" />
              ) : (
                <ChevronRight className="w-3.5 h-3.5" />
              )}
              <span>高级参数与 JSON 规则配置 {showAdvanced ? "（点击收起）" : "（点击展开）"}</span>
            </button>

            {showAdvanced && (
              <div className="mt-3 space-y-3 p-3 rounded-lg border border-brand-borderSubtle bg-slate-950/60">
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                  <div>
                    <Label>最大输出 Token (max_output_tokens)</Label>
                    <Input
                      type="number"
                      value={maxOutputTokens}
                      onChange={(e) => setMaxOutputTokens(e.target.value)}
                      placeholder="如 8192，留空按默认"
                    />
                  </div>
                  <div>
                    <Label>图片模式 (image_mode)</Label>
                    <SelectField
                      value={imageMode}
                      onValueChange={setImageMode}
                      options={[
                        { value: "input_image", label: "input_image (原生传图)" },
                        { value: "save_and_ref", label: "save_and_ref (存盘引用)" },
                        { value: "strip", label: "strip (剥离图片)" },
                      ]}
                    />
                  </div>
                </div>

                <div>
                  <Label>工具重命名映射 (tool_mapping JSON)</Label>
                  <textarea
                    className="w-full h-16 rounded-lg border border-brand-borderSubtle bg-slate-900 p-2 text-xs font-mono text-slate-200 focus-visible:outline-none focus-visible:border-cyan-500"
                    placeholder='例如: {"bash": "execute_command"}'
                    value={toolMappingStr}
                    onChange={(e) => setToolMappingStr(e.target.value)}
                  />
                </div>

                <div>
                  <Label>系统提示词替换规则 (system_replacements JSON)</Label>
                  <textarea
                    className="w-full h-16 rounded-lg border border-brand-borderSubtle bg-slate-900 p-2 text-xs font-mono text-slate-200 focus-visible:outline-none focus-visible:border-cyan-500"
                    placeholder='例如: {"Claude": "AI Assistant"}'
                    value={systemReplacementsStr}
                    onChange={(e) => setSystemReplacementsStr(e.target.value)}
                  />
                </div>

                <div>
                  <Label>请求体附加透传参数 (extra_body JSON)</Label>
                  <textarea
                    className="w-full h-16 rounded-lg border border-brand-borderSubtle bg-slate-900 p-2 text-xs font-mono text-slate-200 focus-visible:outline-none focus-visible:border-cyan-500"
                    placeholder='例如: {"temperature": 0.7}'
                    value={extraBodyStr}
                    onChange={(e) => setExtraBodyStr(e.target.value)}
                  />
                </div>
              </div>
            )}
          </div>
        </div>

        <DialogFooter>
          <Button variant="ghost" onClick={onClose} type="button">
            取消
          </Button>
          <Button variant="primary" onClick={submit} type="button">
            <Save className="w-4 h-4" />
            {isNew ? "添加并保存模型" : "保存模型配置"}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}

function ProviderCard({
  name,
  provider,
  priceNames,
  priceBindings,
  onBindPrice,
  onEditProvider,
  onDeleteProvider,
  onAddModel,
  onEditModel,
  onDeleteModel,
}: {
  name: string;
  provider: any;
  priceNames: string[];
  priceBindings: Record<string, string>;
  onBindPrice: (routeKey: string) => void;
  onEditProvider: (name: string) => void;
  onDeleteProvider: (name: string) => void;
  onAddModel: (providerName: string) => void;
  onEditModel: (providerName: string, modelId: string, cfg: any) => void;
  onDeleteModel: (providerName: string, modelId: string) => void;
}) {
  const [showKey, setShowKey] = useState(false);
  const models = provider.models ?? {};
  const apiKey = provider.api_key ?? "";
  const type = provider.provider_type ?? "openai";

  const typeConfig: Record<string, { badge: string; label: string }> = {
    openai: {
      badge: "border-emerald-500/40 bg-emerald-500/10 text-emerald-400",
      label: "OpenAI",
    },
    gemini: {
      badge: "border-cyan-500/40 bg-cyan-500/10 text-cyan-400",
      label: "Gemini",
    },
    claude: {
      badge: "border-purple-500/40 bg-purple-500/10 text-purple-400",
      label: "Claude",
    },
  };

  const tc = typeConfig[type] || {
    badge: "border-slate-700 bg-slate-800 text-slate-300",
    label: type,
  };

  return (
    <Card className="hover:border-slate-700">
      <CardHeader>
        <div className="flex items-center justify-between gap-2">
          <div className="flex items-center gap-2.5 min-w-0">
            <span className="font-mono text-base font-bold text-cyan-300 truncate">
              {name}
            </span>
            <span
              className={cn(
                "text-[10px] font-mono font-semibold px-2 py-0.5 rounded border whitespace-nowrap",
                tc.badge
              )}
            >
              {tc.label}
            </span>
          </div>

          <div className="flex items-center gap-1">
            <Button
              variant="ghost"
              size="icon"
              className="h-7 w-7 text-slate-400 hover:text-white"
              onClick={() => onEditProvider(name)}
              title="编辑 Provider 上游"
            >
              <Pencil className="w-3.5 h-3.5" />
            </Button>
            <Button
              variant="ghost"
              size="icon"
              className="h-7 w-7 text-slate-400 hover:text-rose-400"
              onClick={() => {
                if (
                  confirm(`确定删除 Provider ${name}？其下所有模型条目将一并删除。`)
                )
                  onDeleteProvider(name);
              }}
              title="删除 Provider"
            >
              <Trash2 className="w-3.5 h-3.5" />
            </Button>
          </div>
        </div>

        <CardDescription className="mt-2 space-y-1.5 font-mono text-xs">
          <div className="flex items-center gap-2 text-slate-400 truncate">
            <Globe className="w-3.5 h-3.5 text-blue-400 shrink-0" />
            <span className="truncate text-slate-300">{provider.base_url || "未设置 Base URL"}</span>
          </div>
          {apiKey && (
            <div className="flex items-center gap-2 text-slate-400">
              <Key className="w-3.5 h-3.5 text-amber-400 shrink-0" />
              <span className="text-slate-300">
                {showKey ? apiKey : apiKey.slice(0, 8) + "••••••••"}
              </span>
              <button
                onClick={() => setShowKey(!showKey)}
                className="text-slate-500 hover:text-slate-300 ml-1"
                title={showKey ? "隐藏" : "显示"}
              >
                {showKey ? <EyeOff className="w-3 h-3" /> : <Eye className="w-3 h-3" />}
              </button>
            </div>
          )}
        </CardDescription>
      </CardHeader>

      <CardContent>
        <div className="space-y-2">
          <div className="flex items-center justify-between text-xs font-mono text-slate-400">
            <span>挂载模型 ({Object.keys(models).length})</span>
            <span>配置与价格表</span>
          </div>

          {Object.keys(models).length === 0 ? (
            <div className="text-xs font-mono text-slate-500 py-3 text-center rounded border border-dashed border-brand-borderSubtle">
              暂无挂载模型
            </div>
          ) : (
            Object.entries(models).map(([modelId, mcfg]) => {
              const routeKey = `${name}/${modelId}`;
              const boundPrice = priceBindings[routeKey];
              const m = (mcfg || {}) as Record<string, any>;
              const hasResponses = Boolean(m.responses);
              const hasReact = Boolean(m.use_react);
              const hasForceStream = Boolean(m.force_stream);

              return (
                <div
                  key={modelId}
                  className="flex items-center justify-between p-2.5 rounded-lg bg-slate-950/70 border border-brand-borderSubtle hover:border-slate-700 transition-colors gap-2"
                >
                  <div className="min-w-0 flex-1">
                    <div className="flex items-center gap-1.5 flex-wrap">
                      <span className="text-xs font-mono font-bold text-slate-100 truncate">
                        {modelId}
                      </span>
                      {hasResponses && (
                        <span className="text-[9px] font-mono px-1 py-0.2 rounded bg-purple-500/20 text-purple-300 border border-purple-500/30">
                          RESPONSES
                        </span>
                      )}
                      {hasReact && (
                        <span className="text-[9px] font-mono px-1 py-0.2 rounded bg-amber-500/20 text-amber-300 border border-amber-500/30">
                          REACT
                        </span>
                      )}
                      {hasForceStream && (
                        <span className="text-[9px] font-mono px-1 py-0.2 rounded bg-cyan-500/20 text-cyan-300 border border-cyan-500/30">
                          STREAM
                        </span>
                      )}
                    </div>
                    <div className="mt-1 flex items-center gap-1.5">
                      {boundPrice ? (
                        <span className="text-[10px] font-mono font-semibold px-1.5 py-0.2 rounded border border-emerald-500/40 bg-emerald-500/10 text-emerald-300 truncate">
                          {boundPrice}
                        </span>
                      ) : (
                        <span className="text-[10px] font-mono font-semibold px-1.5 py-0.2 rounded border border-rose-500/40 bg-rose-500/10 text-rose-300">
                          未绑价
                        </span>
                      )}
                    </div>
                  </div>

                  <div className="flex items-center gap-1 shrink-0">
                    <Button
                      variant="ghost"
                      size="icon"
                      className="h-6 w-6 text-slate-400 hover:text-cyan-300"
                      onClick={() => onEditModel(name, modelId, mcfg)}
                      title="配置模型参数 (如 responses / use_react)"
                    >
                      <Sliders className="w-3.5 h-3.5" />
                    </Button>
                    <Button
                      variant={boundPrice ? "secondary" : "default"}
                      size="sm"
                      className="h-6 text-xs font-mono px-2"
                      onClick={() => onBindPrice(routeKey)}
                      title="关联计费价格表"
                    >
                      <Link2 className="w-3 h-3" />
                      {boundPrice ? "换绑" : "绑价"}
                    </Button>
                    <Button
                      variant="ghost"
                      size="icon"
                      className="h-6 w-6 text-slate-400 hover:text-rose-400"
                      onClick={() => {
                        if (confirm(`确定删除模型 ${modelId}？`))
                          onDeleteModel(name, modelId);
                      }}
                      title="删除模型"
                    >
                      <Trash2 className="w-3 h-3" />
                    </Button>
                  </div>
                </div>
              );
            })
          )}

          <Button
            variant="ghost"
            size="sm"
            className="w-full border border-dashed border-brand-borderSubtle text-xs font-mono text-cyan-400 hover:text-cyan-300 hover:bg-cyan-500/10 mt-1"
            onClick={() => onAddModel(name)}
          >
            <Plus className="w-3.5 h-3.5" />
            添加模型挂载
          </Button>
        </div>
      </CardContent>
    </Card>
  );
}

function BindPriceDialog({
  routeKey,
  priceNames,
  currentBinding,
  onSetBinding,
  onUnbind,
  onClose,
}: {
  routeKey: string;
  priceNames: string[];
  currentBinding?: string;
  onSetBinding: (routeKey: string, priceName: string) => Promise<boolean>;
  onUnbind: (routeKey: string) => Promise<boolean>;
  onClose: () => void;
}) {
  const [selected, setSelected] = useState(currentBinding ?? "");

  const submit = async () => {
    if (!selected) return;
    const ok = await onSetBinding(routeKey, selected);
    if (ok) onClose();
  };

  const unbind = async () => {
    const ok = await onUnbind(routeKey);
    if (ok) onClose();
  };

  return (
    <Dialog open onOpenChange={(o) => !o && onClose()}>
      <DialogContent className="max-w-md">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <Link2 className="w-5 h-5 text-cyan-400" />
            绑定价格表 · {routeKey}
          </DialogTitle>
          <DialogDescription>
            关联价格表用于 Token 成本自动计算
          </DialogDescription>
        </DialogHeader>
        <div className="space-y-4">
          <div>
            <Label>选择价格表</Label>
            <SelectField
              value={selected}
              onValueChange={setSelected}
              placeholder="— 选择价格表 —"
              options={priceNames}
            />
          </div>
          {priceNames.length === 0 && (
            <div className="text-xs text-amber-400 p-2.5 rounded border border-amber-500/30 bg-amber-500/10 font-mono">
              暂无可用价格表，请先到「价格表」页面创建。
            </div>
          )}
        </div>
        <DialogFooter>
          {currentBinding && (
            <Button
              variant="ghost"
              onClick={unbind}
              type="button"
              className="mr-auto text-rose-400 hover:text-rose-300"
            >
              解除当前绑定
            </Button>
          )}
          <Button variant="ghost" onClick={onClose} type="button">
            取消
          </Button>
          <Button
            variant="primary"
            onClick={submit}
            disabled={!selected}
            type="button"
          >
            <Save className="w-4 h-4" />
            确认绑定
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}

export default function ModelsPage() {
  const {
    config,
    pricesResp,
    fetchPrices,
    setBinding,
    deleteBinding,
    saveConfig,
    loading,
  } = useAdmin();
  const [filter, setFilter] = useState("");
  const [bindTarget, setBindTarget] = useState<string | null>(null);
  const [providerEdit, setProviderEdit] = useState<string | "new" | null>(null);

  // Model edit target: { providerName, modelId?, config? }
  const [modelDialogTarget, setModelDialogTarget] = useState<{
    providerName: string;
    modelId?: string;
    config?: Record<string, any>;
  } | null>(null);

  useEffect(() => {
    fetchPrices();
  }, [fetchPrices]);

  const providersMap = (config?.providers ??
    pricesResp?.providers ??
    {}) as Record<string, any>;

  const providers = useMemo(() => {
    return Object.entries(providersMap)
      .filter(
        ([name, p]) =>
          !filter.trim() ||
          name.toLowerCase().includes(filter.toLowerCase()) ||
          Object.keys(p.models ?? {}).some((m) =>
            m.toLowerCase().includes(filter.toLowerCase())
          )
      )
      .map(([name, p]) => ({ name, provider: p }));
  }, [providersMap, filter]);

  const priceNames = Object.keys(pricesResp?.prices ?? {});
  const priceBindings = pricesResp?.price_bindings ?? {};

  const saveProvider = async (name: string, provider: any) => {
    const next = { ...(config ?? ({} as RawAppConfig)) };
    next.providers = { ...(next.providers ?? {}) };
    next.providers[name] = provider;
    return saveConfig(next);
  };

  const deleteProvider = async (name: string) => {
    const next = { ...(config ?? ({} as RawAppConfig)) };
    next.providers = { ...(next.providers ?? {}) };
    delete next.providers[name];
    return saveConfig(next);
  };

  const saveModelConfig = async (
    providerName: string,
    modelId: string,
    modelCfg: Record<string, any>
  ) => {
    const next = { ...(config ?? ({} as RawAppConfig)) };
    next.providers = { ...(next.providers ?? {}) };
    const p = { ...next.providers[providerName] };
    p.models = { ...(p.models ?? {}) };
    p.models[modelId] = modelCfg;
    next.providers[providerName] = p;
    return saveConfig(next);
  };

  const deleteModel = async (providerName: string, modelId: string) => {
    const next = { ...(config ?? ({} as RawAppConfig)) };
    next.providers = { ...(next.providers ?? {}) };
    const p = { ...next.providers[providerName] };
    p.models = { ...(p.models ?? {}) };
    delete p.models[modelId];
    next.providers[providerName] = p;
    return saveConfig(next);
  };

  return (
    <div className="space-y-6 w-full">
      {/* Header */}
      <div className="flex items-end justify-between gap-4 flex-wrap">
        <div>
          <div className="flex items-center gap-2.5">
            <h1 className="text-2xl font-bold text-white tracking-tight">
              模型与上游配置
            </h1>
            <span className="px-2 py-0.5 rounded text-[11px] font-mono border border-blue-500/30 bg-blue-500/10 text-blue-400 font-semibold">
              PROVIDERS
            </span>
          </div>
          <p className="text-xs text-slate-400 mt-1">
            Provider 凭证、模型挂载与配置选项（如 responses 端点、use_react、提示词替换）。
          </p>
        </div>

        <div className="flex items-center gap-3">
          <div className="relative w-72">
            <Search className="w-4 h-4 absolute left-3 top-2.5 text-slate-500" />
            <Input
              className="pl-9 h-9"
              placeholder="搜索 Provider / 模型"
              value={filter}
              onChange={(e) => setFilter(e.target.value)}
            />
          </div>
          <Button
            variant="secondary"
            size="icon"
            onClick={fetchPrices}
            disabled={loading}
            className="h-9 w-9"
          >
            <RefreshCw className={cn("w-3.5 h-3.5 text-cyan-400", loading && "animate-spin")} />
          </Button>
          <Button variant="primary" size="sm" onClick={() => setProviderEdit("new")}>
            <Plus className="w-4 h-4" />
            新建 Provider
          </Button>
        </div>
      </div>

      {/* Global Config Cards */}
      <div className="grid grid-cols-1 xl:grid-cols-3 gap-5">
        <ServerConfigCard
          config={config ?? ({} as RawAppConfig)}
          onSave={saveConfig}
        />
        <UpstreamConfigCard
          config={config ?? ({} as RawAppConfig)}
          onSave={saveConfig}
        />
        <ProxyConfigCard
          config={config ?? ({} as RawAppConfig)}
          onSave={saveConfig}
        />
      </div>

      {/* Provider List */}
      <div className="space-y-3">
        <div className="flex items-center justify-between text-xs text-slate-400 font-mono px-1">
          <div className="flex items-center gap-1.5">
            <Settings2 className="w-4 h-4 text-cyan-400" />
            <span className="font-semibold text-slate-300">已配置的 PROVIDERS ({providers.length})</span>
          </div>
        </div>

        {/* 4-column responsive grid on wide screen */}
        <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 2xl:grid-cols-4 gap-5">
          {providers.length === 0 ? (
            <Card className="col-span-full">
              <CardContent className="py-12 text-center text-slate-500 font-mono text-xs">
                未找到匹配的 Provider 服务
              </CardContent>
            </Card>
          ) : (
            providers.map(({ name, provider }) => (
              <ProviderCard
                key={name}
                name={name}
                provider={provider}
                priceNames={priceNames}
                priceBindings={priceBindings}
                onBindPrice={(routeKey) => setBindTarget(routeKey)}
                onEditProvider={(n) => setProviderEdit(n)}
                onDeleteProvider={deleteProvider}
                onAddModel={(pn) => setModelDialogTarget({ providerName: pn })}
                onEditModel={(pn, mid, cfg) =>
                  setModelDialogTarget({ providerName: pn, modelId: mid, config: cfg })
                }
                onDeleteModel={deleteModel}
              />
            ))
          )}
        </div>
      </div>

      {/* Dialogs */}
      {bindTarget && (
        <BindPriceDialog
          routeKey={bindTarget}
          priceNames={priceNames}
          currentBinding={priceBindings[bindTarget]}
          onSetBinding={setBinding}
          onUnbind={deleteBinding}
          onClose={() => setBindTarget(null)}
        />
      )}

      {providerEdit && (
        <ProviderEditDialog
          initialName={providerEdit === "new" ? undefined : providerEdit}
          initialProvider={
            providerEdit === "new" ? undefined : providersMap[providerEdit]
          }
          existingNames={Object.keys(providersMap)}
          onSave={saveProvider}
          onClose={() => setProviderEdit(null)}
        />
      )}

      {modelDialogTarget && (
        <ModelConfigDialog
          providerName={modelDialogTarget.providerName}
          initialModelId={modelDialogTarget.modelId}
          initialConfig={modelDialogTarget.config}
          existingModelIds={Object.keys(
            providersMap[modelDialogTarget.providerName]?.models ?? {}
          )}
          onSave={(mid, cfg) =>
            saveModelConfig(modelDialogTarget.providerName, mid, cfg)
          }
          onClose={() => setModelDialogTarget(null)}
        />
      )}
    </div>
  );
}
