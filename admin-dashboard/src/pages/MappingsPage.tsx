import { useEffect, useMemo, useState } from "react";
import {
  ArrowRight,
  GitBranch,
  Layers,
  Plus,
  RefreshCw,
  Save,
  Search,
  Trash2,
  X,
} from "lucide-react";
import {
  useAdmin,
  type ModelRoute,
} from "@/store/admin";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Input, Label } from "@/components/ui/input";
import {
  Dialog,
  DialogContent,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { cn } from "@/lib/utils";

type RouteMode = "single" | "list" | "weighted";

function getRouteMode(r: ModelRoute | undefined): RouteMode {
  if (!r) return "single";
  if (typeof r === "string") return "single";
  if (Array.isArray(r)) return "list";
  return "weighted";
}

function getRouteList(r: ModelRoute | undefined): string[] {
  if (!r) return [];
  if (typeof r === "string") return [r];
  if (Array.isArray(r)) return [...r];
  return Object.keys(r);
}

function getRouteWeights(r: ModelRoute | undefined): Record<string, number> {
  if (!r || typeof r === "string" || Array.isArray(r)) return {};
  return r as Record<string, number>;
}

function summarizeRoute(r: ModelRoute | undefined): string {
  if (!r) return "—";
  if (typeof r === "string") return r;
  if (Array.isArray(r)) return r.join(", ");
  return Object.entries(r)
    .map(([k, w]) => `${k}(w:${w})`)
    .join(", ");
}

function MappingEditDialog({
  claudeModel,
  initialRoutes,
  availableRoutes,
  onSave,
  onClose,
}: {
  claudeModel: string;
  initialRoutes: ModelRoute | undefined;
  availableRoutes: string[];
  onSave: (claudeModel: string, routes: ModelRoute) => Promise<boolean>;
  onClose: () => void;
}) {
  const initialMode = getRouteMode(initialRoutes);
  const [mode, setMode] = useState<RouteMode>(initialMode);
  const [selected, setSelected] = useState<string>(
    typeof initialRoutes === "string" ? initialRoutes : "",
  );
  const [list, setList] = useState<string[]>(
    Array.isArray(initialRoutes) ? [...initialRoutes] : [],
  );
  const [weights, setWeights] = useState<Record<string, number>>(
    getRouteWeights(initialRoutes),
  );
  const [pickRoute, setPickRoute] = useState("");

  const addRoute = (r: string) => {
    if (!r) return;
    if (mode === "single") {
      setSelected(r);
    } else if (mode === "list") {
      if (!list.includes(r)) setList([...list, r]);
    } else {
      if (!(r in weights)) setWeights({ ...weights, [r]: 1 });
    }
    setPickRoute("");
  };

  const removeRoute = (r: string) => {
    if (mode === "list") setList(list.filter((x) => x !== r));
    else if (mode === "weighted") {
      const next = { ...weights };
      delete next[r];
      setWeights(next);
    }
  };

  const submit = async () => {
    let routes: ModelRoute;
    if (mode === "single") {
      if (!selected) return;
      routes = selected;
    } else if (mode === "list") {
      if (list.length === 0) return;
      routes = list;
    } else {
      if (Object.keys(weights).length === 0) return;
      routes = weights;
    }
    const ok = await onSave(claudeModel, routes);
    if (ok) onClose();
  };

  const currentRoutes =
    mode === "single"
      ? selected
        ? [selected]
        : []
      : mode === "list"
        ? list
        : Object.keys(weights);

  const unusedRoutes = availableRoutes.filter((r) => !currentRoutes.includes(r));

  return (
    <Dialog open onOpenChange={(o) => !o && onClose()}>
      <DialogContent className="max-w-xl">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <GitBranch className="w-5 h-5 text-brand-violet" />
            编辑映射 · {claudeModel}
          </DialogTitle>
        </DialogHeader>
        <div className="space-y-4">
          <div>
            <Label>路由模式</Label>
            <div className="flex gap-2 mt-1">
              {(
                [
                  { v: "single", label: "单一路由" },
                  { v: "list", label: "负载均衡" },
                  { v: "weighted", label: "加权路由" },
                ] as { v: RouteMode; label: string }[]
              ).map((opt) => (
                <button
                  key={opt.v}
                  onClick={() => setMode(opt.v)}
                  className={cn(
                    "px-3 py-1.5 rounded-md text-xs border transition-colors",
                    mode === opt.v
                      ? "border-brand-cyan/40 bg-brand-cyan/10 text-brand-cyan"
                      : "border-brand-borderSubtle bg-slate-950/40 text-slate-400 hover:text-slate-200",
                  )}
                >
                  {opt.label}
                </button>
              ))}
            </div>
          </div>

          <div>
            <Label>已选路由</Label>
            <div className="mt-1 space-y-1.5 min-h-[2rem]">
              {currentRoutes.length === 0 ? (
                <div className="text-xs text-slate-500 py-2">尚未选择路由</div>
              ) : (
                currentRoutes.map((r) => (
                  <div
                    key={r}
                    className="flex items-center justify-between p-2 rounded-md bg-slate-900/40 border border-brand-borderSubtle"
                  >
                    <span className="text-sm font-mono text-slate-200">{r}</span>
                    {mode === "weighted" && (
                      <input
                        type="number"
                        value={weights[r] ?? 1}
                        onChange={(e) =>
                          setWeights({ ...weights, [r]: Number(e.target.value) })
                        }
                        className="w-16 h-7 rounded border border-brand-borderSubtle bg-slate-950/40 px-2 text-xs text-slate-200 text-right"
                      />
                    )}
                    {mode !== "single" && (
                      <button
                        onClick={() => removeRoute(r)}
                        className="text-slate-500 hover:text-rose-400"
                      >
                        <X className="w-3.5 h-3.5" />
                      </button>
                    )}
                  </div>
                ))
              )}
            </div>
          </div>

          {unusedRoutes.length > 0 && (
            <div>
              <Label>添加路由</Label>
              <div className="flex gap-2 mt-1">
                <select
                  className="flex h-9 flex-1 rounded-md border border-brand-borderSubtle bg-slate-950/40 px-3 py-1 text-sm text-slate-100"
                  value={pickRoute}
                  onChange={(e) => setPickRoute(e.target.value)}
                >
                  <option value="">— 选择 provider/model_id —</option>
                  {unusedRoutes.map((r) => (
                    <option key={r} value={r}>
                      {r}
                    </option>
                  ))}
                </select>
                <Button
                  variant="secondary"
                  size="sm"
                  onClick={() => addRoute(pickRoute)}
                  disabled={!pickRoute}
                >
                  <Plus className="w-3.5 h-3.5" />
                  添加
                </Button>
              </div>
            </div>
          )}
        </div>
        <DialogFooter>
          <Button variant="ghost" onClick={onClose} type="button">
            取消
          </Button>
          <Button variant="primary" onClick={submit} type="button">
            <Save className="w-4 h-4" />
            保存映射
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}

function NewMappingDialog({
  existingModels,
  availableRoutes,
  onSave,
  onClose,
}: {
  existingModels: string[];
  availableRoutes: string[];
  onSave: (claudeModel: string, routes: ModelRoute) => Promise<boolean>;
  onClose: () => void;
}) {
  const [name, setName] = useState("");
  const [route, setRoute] = useState("");

  const submit = async () => {
    if (!name.trim() || !route) return;
    if (existingModels.includes(name.trim())) {
      return;
    }
    const ok = await onSave(name.trim(), route);
    if (ok) onClose();
  };

  return (
    <Dialog open onOpenChange={(o) => !o && onClose()}>
      <DialogContent className="max-w-md">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <Plus className="w-5 h-5 text-brand-cyan" />
            新建 Claude 模型映射
          </DialogTitle>
        </DialogHeader>
        <div className="space-y-4">
          <div>
            <Label>Claude 模型名</Label>
            <Input
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="claude-sonnet-5"
            />
          </div>
          <div>
            <Label>路由到</Label>
            <select
              className="flex h-9 w-full rounded-md border border-brand-borderSubtle bg-slate-950/40 px-3 py-1 text-sm text-slate-100"
              value={route}
              onChange={(e) => setRoute(e.target.value)}
            >
              <option value="">— 选择 provider/model_id —</option>
              {availableRoutes.map((r) => (
                <option key={r} value={r}>
                  {r}
                </option>
              ))}
            </select>
          </div>
        </div>
        <DialogFooter>
          <Button variant="ghost" onClick={onClose} type="button">
            取消
          </Button>
          <Button
            variant="primary"
            onClick={submit}
            type="button"
            disabled={!name.trim() || !route}
          >
            <Plus className="w-4 h-4" />
            创建
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}

export default function MappingsPage() {
  const {
    config,
    pricesResp,
    setModelMapping,
    deleteModelMapping,
    fetchPrices,
    loading,
  } = useAdmin();
  const [filter, setFilter] = useState("");
  const [editing, setEditing] = useState<string | null>(null);
  const [creating, setCreating] = useState(false);

  useEffect(() => {
    fetchPrices();
  }, [fetchPrices]);

  const models = useMemo(() => {
    const raw = (config?.models ??
      (pricesResp?.models as any) ??
      {}) as Record<string, ModelRoute>;
    return Object.entries(raw)
      .filter(
        ([name, _]) =>
          !filter.trim() || name.toLowerCase().includes(filter.toLowerCase()),
      )
      .map(([name, routes]) => ({ name, routes }));
  }, [config, pricesResp, filter]);

  const availableRoutes = useMemo(() => {
    const providers = (config?.providers ??
      pricesResp?.providers ??
      {}) as Record<string, any>;
    const routes: string[] = [];
    for (const [pname, p] of Object.entries(providers)) {
      for (const modelId of Object.keys(p.models ?? {})) {
        routes.push(`${pname}/${modelId}`);
      }
    }
    return routes;
  }, [config, pricesResp]);

  const existingModels = Object.keys(config?.models ?? {});

  return (
    <div className="space-y-6">
      <div className="flex items-end justify-between gap-4 flex-wrap">
        <div>
          <h1 className="text-2xl font-bold text-white tracking-wide glow-text">
            模型映射
          </h1>
          <p className="text-sm text-slate-400 mt-1">
            把 Claude 模型名映射到后端 Provider 路由。修改{" "}
            <code className="text-brand-cyan">config.models</code>。
          </p>
        </div>
        <div className="flex items-center gap-3">
          <div className="relative w-72">
            <Search className="w-4 h-4 absolute left-3 top-2.5 text-slate-500" />
            <Input
              className="pl-9"
              placeholder="搜索 Claude 模型"
              value={filter}
              onChange={(e) => setFilter(e.target.value)}
            />
          </div>
          <Button variant="ghost" size="icon" onClick={fetchPrices} disabled={loading}>
            <RefreshCw className={cn("w-4 h-4", loading && "animate-spin")} />
          </Button>
          <Button variant="primary" onClick={() => setCreating(true)}>
            <Plus className="w-4 h-4" />
            新建映射
          </Button>
        </div>
      </div>

      <div className="grid grid-cols-1 xl:grid-cols-2 gap-5">
        {models.length === 0 ? (
          <Card className="xl:col-span-2">
            <CardContent className="py-14 text-center text-slate-400">
              <Layers className="mx-auto w-10 h-10 text-brand-violet/60 mb-3" />
              <div className="text-white font-semibold mb-1">暂无模型映射</div>
              <div>点击右上角「新建映射」创建第一条 Claude 模型路由。</div>
            </CardContent>
          </Card>
        ) : (
          models.map((m) => {
            const routes = getRouteList(m.routes);
            const mode = getRouteMode(m.routes);

            return (
              <Card key={m.name}>
                <CardHeader>
                  <CardTitle className="flex items-center justify-between gap-2">
                    <span className="font-mono text-base text-white">{m.name}</span>
                    <div className="flex items-center gap-1">
                      <Button
                        variant="ghost"
                        size="sm"
                        onClick={() => setEditing(m.name)}
                      >
                        <GitBranch className="w-3.5 h-3.5" />
                        编辑
                      </Button>
                      <Button
                        variant="ghost"
                        size="icon"
                        onClick={async () => {
                          if (confirm(`确定删除映射 ${m.name}？`))
                            await deleteModelMapping(m.name);
                        }}
                      >
                        <Trash2 className="w-3.5 h-3.5 text-rose-400" />
                      </Button>
                    </div>
                  </CardTitle>
                  <CardDescription>
                    {routes.length} 条路由 ·{" "}
                    {mode === "single"
                      ? "单一路由"
                      : mode === "list"
                        ? "负载均衡"
                        : "加权路由"}
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="space-y-2">
                    {routes.map((r) => {
                      const weight = getRouteWeights(m.routes)[r];
                      return (
                        <div
                          key={r}
                          className="flex items-center gap-3 p-2.5 rounded-lg bg-slate-900/40 border border-brand-borderSubtle"
                        >
                          <ArrowRight className="w-3.5 h-3.5 text-slate-500 shrink-0" />
                          <span className="text-sm font-mono text-slate-200 flex-1">
                            {r}
                          </span>
                          {weight !== undefined && (
                            <span className="text-xs px-1.5 py-0.5 rounded border border-brand-violet/30 bg-brand-violet/10 text-brand-violet whitespace-nowrap">
                              w:{weight}
                            </span>
                          )}
                        </div>
                      );
                    })}
                  </div>
                </CardContent>
              </Card>
            );
          })
        )}
      </div>

      {editing && (
        <MappingEditDialog
          claudeModel={editing}
          initialRoutes={config?.models?.[editing] as ModelRoute | undefined}
          availableRoutes={availableRoutes}
          onSave={setModelMapping}
          onClose={() => setEditing(null)}
        />
      )}

      {creating && (
        <NewMappingDialog
          existingModels={existingModels}
          availableRoutes={availableRoutes}
          onSave={setModelMapping}
          onClose={() => setCreating(false)}
        />
      )}
    </div>
  );
}
