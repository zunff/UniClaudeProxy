import { useEffect, useMemo, useState } from "react";
import { toast } from "sonner";
import {
  ChevronDown,
  ChevronRight,
  Eye,
  EyeOff,
  Key,
  Link2,
  Pencil,
  Plus,
  RefreshCw,
  Save,
  Search,
  Server,
  Settings2,
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
      <label className="flex items-center gap-2 cursor-pointer">
        <input
          type="checkbox"
          checked={!!value}
          onChange={(e) => onChange(String(e.target.checked))}
          disabled={disabled}
          className="w-4 h-4 rounded border-brand-borderSubtle bg-slate-950/40 accent-brand-cyan"
        />
        <span className="text-sm text-slate-300">{label}</span>
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
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Server className="w-5 h-5 text-brand-cyan" />
          服务器配置
        </CardTitle>
        <CardDescription>本地监听地址与端口</CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="grid grid-cols-3 gap-4">
          <ConfigField
            label="Host"
            value={server.host ?? ""}
            onChange={(v) => setServer("host", v)}
          />
          <ConfigField
            label="Port"
            value={server.port ?? ""}
            onChange={(v) => setServer("port", v)}
          />
          <div className="flex items-end pb-1">
            <ConfigField
              label="local_only"
              type="checkbox"
              value={server.local_only ?? false}
              onChange={(v) => setServer("local_only", v)}
            />
          </div>
        </div>
        {dirty && (
          <div className="flex justify-end">
            <Button variant="primary" size="sm" onClick={save} disabled={saving}>
              <Save className="w-3.5 h-3.5" />
              {saving ? "保存中..." : "保存"}
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
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Zap className="w-5 h-5 text-brand-amber" />
          上游 / 重试配置
        </CardTitle>
        <CardDescription>超时、重试与禁用路由</CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="flex items-end pb-1">
          <ConfigField
            label="启用上游故障转移（多路由自动切换）"
            type="checkbox"
            value={up.enabled ?? false}
            onChange={(v) => setUp("enabled", v === "true")}
          />
        </div>
        <div className="grid grid-cols-2 gap-4">
          <ConfigField
            label="流式首字节超时 (ms)"
            type="number"
            value={up.stream?.first_byte_timeout_ms ?? ""}
            onChange={(v) =>
              setUp("stream", { ...up.stream, first_byte_timeout_ms: Number(v) })
            }
          />
          <ConfigField
            label="非流式首字节超时 (ms)"
            type="number"
            value={up.non_stream?.first_byte_timeout_ms ?? ""}
            onChange={(v) =>
              setUp("non_stream", { ...up.non_stream, first_byte_timeout_ms: Number(v) })
            }
          />
          <ConfigField
            label="重试最大次数"
            type="number"
            value={up.retry?.max_attempts ?? ""}
            onChange={(v) =>
              setUp("retry", { ...up.retry, max_attempts: Number(v) })
            }
          />
          <ConfigField
            label="重试间隔 (ms)"
            type="number"
            value={up.retry?.interval_ms ?? ""}
            onChange={(v) =>
              setUp("retry", { ...up.retry, interval_ms: Number(v) })
            }
          />
          <ConfigField
            label="重试总超时 (ms)"
            type="number"
            value={up.retry?.total_timeout_ms ?? ""}
            onChange={(v) =>
              setUp("retry", { ...up.retry, total_timeout_ms: Number(v) })
            }
          />
        </div>
        <div>
          <button
            className="flex items-center gap-1.5 text-xs text-slate-400 hover:text-slate-200"
            onClick={() => setExpanded(!expanded)}
          >
            {expanded ? (
              <ChevronDown className="w-3.5 h-3.5" />
            ) : (
              <ChevronRight className="w-3.5 h-3.5" />
            )}
            禁用路由 ({up.disabled_routes?.length ?? 0} 条)
          </button>
          {expanded && (
            <textarea
              className="mt-2 w-full h-24 rounded-md border border-brand-borderSubtle bg-slate-950/40 px-3 py-2 text-xs font-mono text-slate-300 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-brand-cyan/50"
              value={(up.disabled_routes ?? []).join("\n")}
              onChange={(e) =>
                setUp(
                  "disabled_routes",
                  e.target.value
                    .split("\n")
                    .map((s) => s.trim())
                    .filter(Boolean),
                )
              }
            />
          )}
        </div>
        {dirty && (
          <div className="flex justify-end">
            <Button variant="primary" size="sm" onClick={save} disabled={saving}>
              <Save className="w-3.5 h-3.5" />
              {saving ? "保存中..." : "保存"}
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
    initialProvider?.provider_type ?? "openai",
  );
  const [apiKey, setApiKey] = useState(initialProvider?.api_key ?? "");
  const [baseUrl, setBaseUrl] = useState(initialProvider?.base_url ?? "");

  const submit = async () => {
    if (!name.trim()) {
      toast.error("Provider 名称不能为空");
      return;
    }
    if (isNew && existingNames.includes(name.trim())) {
      toast.error("Provider 名称已存在");
      return;
    }
    const provider: any = {
      provider_type: providerType,
      api_key: apiKey,
      base_url: baseUrl,
      models: initialProvider?.models ?? {},
    };
    // Preserve extra fields like headers
    if (initialProvider?.headers) provider.headers = initialProvider.headers;
    const ok = await onSave(name.trim(), provider);
    if (ok) onClose();
  };

  return (
    <Dialog open onOpenChange={(o) => !o && onClose()}>
      <DialogContent className="max-w-md">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <Server className="w-5 h-5 text-brand-cyan" />
            {isNew ? "新建 Provider" : `编辑 ${initialName}`}
          </DialogTitle>
        </DialogHeader>
        <div className="space-y-4">
          <div>
            <Label>Provider 名称（唯一标识）</Label>
            <Input
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="beyondpower"
              disabled={!isNew}
            />
          </div>
          <div>
            <Label>Provider 类型</Label>
            <SelectField
              value={providerType}
              onValueChange={setProviderType}
              placeholder="选择 provider 类型"
              options={["openai", "gemini", "anthropic"]}
            />
          </div>
          <div>
            <Label>API Key</Label>
            <Input
              type="password"
              value={apiKey}
              onChange={(e) => setApiKey(e.target.value)}
              placeholder="sk-..."
            />
          </div>
          <div>
            <Label>Base URL</Label>
            <Input
              value={baseUrl}
              onChange={(e) => setBaseUrl(e.target.value)}
              placeholder="https://api.example.com/v1"
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

function ModelAddDialog({
  providerName,
  existingModelIds,
  onSave,
  onClose,
}: {
  providerName: string;
  existingModelIds: string[];
  onSave: (modelId: string, modelCfg: Record<string, any>) => Promise<boolean>;
  onClose: () => void;
}) {
  const [modelId, setModelId] = useState("");
  const [upstreamName, setUpstreamName] = useState("");

  const submit = async () => {
    if (!modelId.trim()) {
      toast.error("模型 ID 不能为空");
      return;
    }
    if (existingModelIds.includes(modelId.trim())) {
      toast.error("模型 ID 已存在");
      return;
    }
    const cfg: Record<string, any> = { name: upstreamName.trim() || modelId.trim() };
    const ok = await onSave(modelId.trim(), cfg);
    if (ok) onClose();
  };

  return (
    <Dialog open onOpenChange={(o) => !o && onClose()}>
      <DialogContent className="max-w-md">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <Plus className="w-5 h-5 text-brand-cyan" />
            添加模型 · {providerName}
          </DialogTitle>
        </DialogHeader>
        <div className="space-y-4">
          <div>
            <Label>模型 ID（路由用）</Label>
            <Input
              value={modelId}
              onChange={(e) => setModelId(e.target.value)}
              placeholder="ds / glm-4.7-flash / ..."
            />
          </div>
          <div>
            <Label>上游模型名（name）</Label>
            <Input
              value={upstreamName}
              onChange={(e) => setUpstreamName(e.target.value)}
              placeholder="留空则与 ID 相同"
            />
          </div>
        </div>
        <DialogFooter>
          <Button variant="ghost" onClick={onClose} type="button">
            取消
          </Button>
          <Button variant="primary" onClick={submit} disabled={!modelId.trim()} type="button">
            <Plus className="w-4 h-4" />
            添加
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
  onDeleteModel: (providerName: string, modelId: string) => void;
}) {
  const [showKey, setShowKey] = useState(false);
  const models = provider.models ?? {};
  const apiKey = provider.api_key ?? "";

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center justify-between gap-2">
          <span className="font-mono text-base text-brand-cyan">{name}</span>
          <div className="flex items-center gap-1">
            <span className="text-xs px-2 py-0.5 rounded-md border border-brand-borderSubtle bg-brand-panel2 text-slate-400 whitespace-nowrap">
              {provider.provider_type ?? "unknown"}
            </span>
            <Button variant="ghost" size="icon" onClick={() => onEditProvider(name)}>
              <Pencil className="w-3.5 h-3.5" />
            </Button>
            <Button
              variant="ghost"
              size="icon"
              onClick={() => {
                if (confirm(`确定删除 Provider ${name}？其下所有模型将一并删除。`))
                  onDeleteProvider(name);
              }}
            >
              <Trash2 className="w-3.5 h-3.5 text-rose-400" />
            </Button>
          </div>
        </CardTitle>
        <CardDescription className="space-y-1">
          <div className="flex items-center gap-2 min-w-0">
            <span className="text-xs text-slate-500 shrink-0">base_url:</span>
            <code className="text-sm text-slate-300 truncate">{provider.base_url ?? "—"}</code>
          </div>
          {apiKey && (
            <div className="flex items-center gap-2">
              <Key className="w-3.5 h-3.5 text-slate-500" />
              <code className="text-sm text-slate-400">
                {showKey ? apiKey : apiKey.slice(0, 8) + "••••••••"}
              </code>
              <button
                onClick={() => setShowKey(!showKey)}
                className="text-slate-500 hover:text-slate-300"
              >
                {showKey ? <EyeOff className="w-3 h-3" /> : <Eye className="w-3 h-3" />}
              </button>
            </div>
          )}
        </CardDescription>
      </CardHeader>
      <CardContent>
        <div className="space-y-2">
          {Object.keys(models).length === 0 ? (
            <div className="text-xs text-slate-500 py-2">无模型</div>
          ) : (
            Object.entries(models).map(([modelId, _]) => {
              const routeKey = `${name}/${modelId}`;
              const boundPrice = priceBindings[routeKey];
              return (
                <div
                  key={modelId}
                  className="flex items-center justify-between p-2.5 rounded-lg bg-slate-900/40 border border-brand-borderSubtle"
                >
                  <div className="flex items-center gap-2 min-w-0">
                    <span className="text-sm font-mono text-slate-200">{modelId}</span>
                    {boundPrice ? (
                      <span className="text-[10px] px-1.5 py-0.5 rounded border border-brand-green/40 bg-brand-green/10 text-brand-green truncate">
                        {boundPrice}
                      </span>
                    ) : (
                      <span className="text-[10px] px-1.5 py-0.5 rounded border border-rose-500/30 bg-rose-500/10 text-rose-300">
                        无价格表
                      </span>
                    )}
                  </div>
                  <div className="flex items-center gap-1">
                    <Button
                      variant={boundPrice ? "ghost" : "secondary"}
                      size="sm"
                      onClick={() => onBindPrice(routeKey)}
                    >
                      <Link2 className="w-3 h-3" />
                      {boundPrice ? "更换" : "绑定"}
                    </Button>
                    <Button
                      variant="ghost"
                      size="icon"
                      onClick={() => {
                        if (confirm(`确定删除模型 ${modelId}？`))
                          onDeleteModel(name, modelId);
                      }}
                    >
                      <Trash2 className="w-3.5 h-3.5 text-rose-400" />
                    </Button>
                  </div>
                </div>
              );
            })
          )}
          <Button
            variant="ghost"
            size="sm"
            className="w-full border border-dashed border-brand-borderSubtle"
            onClick={() => onAddModel(name)}
          >
            <Plus className="w-3.5 h-3.5" />
            添加模型
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
            <Link2 className="w-5 h-5 text-brand-cyan" />
            绑定价格表 · {routeKey}
          </DialogTitle>
        </DialogHeader>
        <div className="space-y-3">
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
            <div className="text-xs text-slate-500">
              暂无价格表，请先到「价格表」页面创建。
            </div>
          )}
        </div>
        <DialogFooter>
          {currentBinding && (
            <Button variant="ghost" onClick={unbind} type="button" className="mr-auto text-rose-400">
              解绑
            </Button>
          )}
          <Button variant="ghost" onClick={onClose} type="button">
            取消
          </Button>
          <Button variant="primary" onClick={submit} disabled={!selected} type="button">
            <Save className="w-4 h-4" />
            绑定
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}

export default function ModelsPage() {
  const { config, pricesResp, fetchPrices, setBinding, deleteBinding, saveConfig, loading } =
    useAdmin();
  const [filter, setFilter] = useState("");
  const [bindTarget, setBindTarget] = useState<string | null>(null);
  const [providerEdit, setProviderEdit] = useState<string | "new" | null>(null);
  const [modelAddTarget, setModelAddTarget] = useState<string | null>(null);

  useEffect(() => {
    fetchPrices();
  }, [fetchPrices]);

  const providersMap = (config?.providers ?? pricesResp?.providers ?? {}) as Record<
    string,
    any
  >;

  const providers = useMemo(() => {
    return Object.entries(providersMap)
      .filter(
        ([name, p]) =>
          !filter.trim() ||
          name.toLowerCase().includes(filter.toLowerCase()) ||
          Object.keys(p.models ?? {}).some((m) =>
            m.toLowerCase().includes(filter.toLowerCase()),
          ),
      )
      .map(([name, p]) => ({ name, provider: p }));
  }, [providersMap, filter]);

  const priceNames = Object.keys(pricesResp?.prices ?? {});
  const priceBindings = pricesResp?.price_bindings ?? {};

  // Provider CRUD via saveConfig
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

  // Model CRUD via saveConfig
  const addModel = async (providerName: string, modelId: string, modelCfg: Record<string, any>) => {
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
    <div className="space-y-6">
      <div className="flex items-end justify-between gap-4 flex-wrap">
        <div>
          <h1 className="text-2xl font-bold text-white tracking-wide glow-text">
            模型配置
          </h1>
          <p className="text-sm text-slate-400 mt-1">
            Provider 列表、模型定义与服务器参数。模型相关写入{" "}
            <code className="text-brand-cyan">config.json</code>
            ，超时/计费等写入{" "}
            <code className="text-brand-cyan">global.json</code>。
          </p>
        </div>
        <div className="flex items-center gap-3">
          <div className="relative w-72">
            <Search className="w-4 h-4 absolute left-3 top-2.5 text-slate-500" />
            <Input
              className="pl-9"
              placeholder="搜索 Provider / 模型"
              value={filter}
              onChange={(e) => setFilter(e.target.value)}
            />
          </div>
          <Button variant="ghost" size="icon" onClick={fetchPrices} disabled={loading}>
            <RefreshCw className={cn("w-4 h-4", loading && "animate-spin")} />
          </Button>
        </div>
      </div>

      <div className="grid grid-cols-1 xl:grid-cols-2 gap-5">
        <ServerConfigCard config={config ?? ({} as RawAppConfig)} onSave={saveConfig} />
        <UpstreamConfigCard config={config ?? ({} as RawAppConfig)} onSave={saveConfig} />
      </div>

      <div className="flex items-center justify-between gap-2 text-sm text-slate-400">
        <div className="flex items-center gap-2">
          <Settings2 className="w-4 h-4" />
          <span>Providers ({providers.length})</span>
        </div>
        <Button variant="primary" size="sm" onClick={() => setProviderEdit("new")}>
          <Plus className="w-4 h-4" />
          新建 Provider
        </Button>
      </div>

      <div className="grid grid-cols-1 xl:grid-cols-2 2xl:grid-cols-3 gap-5">
        {providers.length === 0 ? (
          <Card className="xl:col-span-2 2xl:col-span-3">
            <CardContent className="py-10 text-center text-slate-400">
              暂无 Provider
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
              onAddModel={(pn) => setModelAddTarget(pn)}
              onDeleteModel={deleteModel}
            />
          ))
        )}
      </div>

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

      {modelAddTarget && (
        <ModelAddDialog
          providerName={modelAddTarget}
          existingModelIds={Object.keys(providersMap[modelAddTarget]?.models ?? {})}
          onSave={(modelId, cfg) => addModel(modelAddTarget, modelId, cfg)}
          onClose={() => setModelAddTarget(null)}
        />
      )}
    </div>
  );
}
