import { useEffect, useMemo, useState } from "react";
import { toast } from "sonner";
import {
  ArrowRight,
  GitBranch,
  Layers,
  Plus,
  RefreshCw,
  Route,
  Save,
  Search,
  Trash2,
  X,
} from "lucide-react";
import { useAdmin, type ModelRoute } from "@/store/admin";
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
    typeof initialRoutes === "string" ? initialRoutes : ""
  );
  const [list, setList] = useState<string[]>(
    Array.isArray(initialRoutes) ? [...initialRoutes] : []
  );
  const [weights, setWeights] = useState<Record<string, number>>(
    getRouteWeights(initialRoutes)
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

  const unusedRoutes = availableRoutes.filter(
    (r) => !currentRoutes.includes(r)
  );

  return (
    <Dialog open onOpenChange={(o) => !o && onClose()}>
      <DialogContent className="max-w-lg">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <GitBranch className="w-5 h-5 text-purple-400" />
            编辑模型映射 · {claudeModel}
          </DialogTitle>
          <DialogDescription>
            配置直通单路由、多上游负载均衡或权重轮询
          </DialogDescription>
        </DialogHeader>
        <div className="space-y-4 font-mono">
          <div>
            <Label>路由模式</Label>
            <div className="grid grid-cols-3 gap-2 mt-1">
              {(
                [
                  { v: "single", label: "单一路由" },
                  { v: "list", label: "负载均衡 (轮询)" },
                  { v: "weighted", label: "按权重分流" },
                ] as { v: RouteMode; label: string }[]
              ).map((opt) => (
                <button
                  key={opt.v}
                  onClick={() => setMode(opt.v)}
                  type="button"
                  className={cn(
                    "p-2 rounded-lg text-xs border text-center transition-all",
                    mode === opt.v
                      ? "border-purple-500 bg-purple-500/15 text-purple-300 font-bold"
                      : "border-brand-borderSubtle bg-slate-950/60 text-slate-400 hover:text-slate-200"
                  )}
                >
                  {opt.label}
                </button>
              ))}
            </div>
          </div>

          <div>
            <Label>已选路由</Label>
            <div className="mt-1 space-y-1.5 min-h-[2.5rem]">
              {mode === "single" ? (
                <SelectField
                  className="font-mono"
                  value={selected}
                  onValueChange={setSelected}
                  placeholder="— 选择 provider/model_id —"
                  options={availableRoutes}
                />
              ) : (
                <>
                  {currentRoutes.length === 0 ? (
                    <div className="text-xs text-slate-500 py-3 text-center rounded border border-dashed border-brand-borderSubtle">
                      尚未选择任何路由
                    </div>
                  ) : (
                    currentRoutes.map((r) => (
                      <div
                        key={r}
                        className="flex items-center justify-between p-2 rounded-lg bg-slate-950/60 border border-brand-borderSubtle"
                      >
                        <span className="text-xs font-mono font-bold text-slate-200">
                          {r}
                        </span>
                        <div className="flex items-center gap-2">
                          {mode === "weighted" && (
                            <div className="flex items-center gap-1">
                              <span className="text-[11px] text-slate-400">权重:</span>
                              <input
                                type="number"
                                min="1"
                                value={weights[r] ?? 1}
                                onChange={(e) =>
                                  setWeights({
                                    ...weights,
                                    [r]: Math.max(1, Number(e.target.value)),
                                  })
                                }
                                className="w-12 h-6 rounded border border-brand-borderSubtle bg-slate-900 px-1 text-xs text-cyan-400 font-bold text-center"
                              />
                            </div>
                          )}
                          <button
                            onClick={() => removeRoute(r)}
                            className="text-slate-500 hover:text-rose-400 p-1"
                          >
                            <X className="w-3.5 h-3.5" />
                          </button>
                        </div>
                      </div>
                    ))
                  )}
                </>
              )}
            </div>
          </div>

          {mode !== "single" && unusedRoutes.length > 0 && (
            <div>
              <Label>添加路由</Label>
              <div className="flex gap-2 mt-1">
                <SelectField
                  className="flex-1 font-mono"
                  value={pickRoute}
                  onValueChange={setPickRoute}
                  placeholder="— 选择待添加路由 —"
                  options={unusedRoutes}
                />
                <Button
                  variant="secondary"
                  size="default"
                  onClick={() => addRoute(pickRoute)}
                  disabled={!pickRoute}
                  className="text-xs"
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
      toast.error("该 Claude 模型名已存在映射");
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
            <Plus className="w-5 h-5 text-purple-400" />
            新建 Claude 模型映射
          </DialogTitle>
          <DialogDescription>
            将客户端请求的 Claude 模型映射至后端的上游 Provider 路由
          </DialogDescription>
        </DialogHeader>
        <div className="space-y-4">
          <div>
            <Label>Claude 模型名 (客户端发送)</Label>
            <Input
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="claude-sonnet-5"
            />
          </div>
          <div>
            <Label>目标路由</Label>
            <SelectField
              className="font-mono"
              value={route}
              onValueChange={setRoute}
              placeholder="— 选择 provider/model_id —"
              options={availableRoutes}
            />
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
            确认创建
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
          !filter.trim() || name.toLowerCase().includes(filter.toLowerCase())
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
    <div className="space-y-6 w-full">
      {/* Header */}
      <div className="flex items-end justify-between gap-4 flex-wrap">
        <div>
          <div className="flex items-center gap-2.5">
            <h1 className="text-2xl font-bold text-white tracking-tight">
              模型映射
            </h1>
            <span className="px-2 py-0.5 rounded text-[11px] font-mono border border-purple-500/30 bg-purple-500/10 text-purple-400 font-semibold">
              MAPPINGS
            </span>
          </div>
          <p className="text-xs text-slate-400 mt-1">
            将客户端请求的 Claude 模型映射至后端的上游 Provider 路由。
          </p>
        </div>

        <div className="flex items-center gap-3">
          <div className="relative w-64">
            <Search className="w-4 h-4 absolute left-3 top-2.5 text-slate-500" />
            <Input
              className="pl-9 h-9"
              placeholder="搜索 Claude 模型"
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
          <Button variant="primary" size="sm" onClick={() => setCreating(true)}>
            <Plus className="w-4 h-4" />
            新建映射
          </Button>
        </div>
      </div>

      {/* Mappings Grid - 4 columns on wide screen */}
      <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 2xl:grid-cols-4 gap-5">
        {models.length === 0 ? (
          <Card className="col-span-full">
            <CardContent className="py-12 text-center text-slate-500 font-mono text-xs">
              <Route className="mx-auto w-8 h-8 text-slate-600 mb-2" />
              <div>暂无模型映射</div>
            </CardContent>
          </Card>
        ) : (
          models.map((m) => {
            const routes = getRouteList(m.routes);
            const mode = getRouteMode(m.routes);

            return (
              <Card key={m.name} className="hover:border-slate-700">
                <CardHeader>
                  <div className="flex items-center justify-between gap-3">
                    <div className="min-w-0">
                      <span className="font-mono text-base font-bold text-purple-300 truncate block">
                        {m.name}
                      </span>
                      <div className="flex items-center gap-2 mt-1 text-xs text-slate-400 font-mono">
                        <span
                          className={cn(
                            "text-[10px] px-1.5 py-0.2 rounded border font-medium",
                            mode === "single"
                              ? "border-cyan-500/30 bg-cyan-500/10 text-cyan-300"
                              : mode === "list"
                              ? "border-emerald-500/30 bg-emerald-500/10 text-emerald-300"
                              : "border-amber-500/30 bg-amber-500/10 text-amber-300"
                          )}
                        >
                          {mode === "single"
                            ? "单一直通"
                            : mode === "list"
                            ? "负载均衡"
                            : "加权分流"}
                        </span>
                        <span>{routes.length} 条路由</span>
                      </div>
                    </div>

                    <div className="flex items-center gap-1">
                      <Button
                        variant="secondary"
                        size="sm"
                        className="h-7 text-xs font-mono px-2.5"
                        onClick={() => setEditing(m.name)}
                      >
                        <GitBranch className="w-3 h-3" />
                        编辑
                      </Button>
                      <Button
                        variant="ghost"
                        size="icon"
                        className="h-7 w-7 text-slate-400 hover:text-rose-400"
                        onClick={async () => {
                          if (confirm(`确定删除映射 ${m.name}？`))
                            await deleteModelMapping(m.name);
                        }}
                      >
                        <Trash2 className="w-3.5 h-3.5" />
                      </Button>
                    </div>
                  </div>
                </CardHeader>
                <CardContent>
                  <div className="space-y-1.5">
                    {routes.map((r) => {
                      const weight = getRouteWeights(m.routes)[r];
                      return (
                        <div
                          key={r}
                          className="flex items-center gap-2.5 p-2 rounded-lg bg-slate-950/70 border border-brand-borderSubtle font-mono"
                        >
                          <ArrowRight className="w-3.5 h-3.5 text-cyan-400 shrink-0" />
                          <span className="text-xs font-semibold text-slate-200 flex-1 truncate">
                            {r}
                          </span>
                          {weight !== undefined && (
                            <span className="text-[11px] font-mono font-bold px-1.5 py-0.2 rounded border border-purple-500/40 bg-purple-500/15 text-purple-300 whitespace-nowrap">
                              权重: {weight}
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

      {/* Edit Dialog */}
      {editing && (
        <MappingEditDialog
          claudeModel={editing}
          initialRoutes={config?.models?.[editing] as ModelRoute | undefined}
          availableRoutes={availableRoutes}
          onSave={setModelMapping}
          onClose={() => setEditing(null)}
        />
      )}

      {/* New Dialog */}
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
