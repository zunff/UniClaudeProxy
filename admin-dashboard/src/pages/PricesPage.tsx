import { useEffect, useMemo, useState } from "react";
import { toast } from "sonner";
import {
  Banknote,
  Layers,
  Link2,
  Plus,
  RefreshCw,
  Search,
  Sparkles,
  Trash2,
  Unlink,
} from "lucide-react";
import { useAdmin, type PriceTableEntry } from "@/store/admin";
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
  DialogTrigger,
} from "@/components/ui/dialog";
import { cn, formatMoney } from "@/lib/utils";

function normalizeEntry(entry?: PriceTableEntry | null) {
  if (!entry) {
    return {
      currency: "CNY",
      displayName: "",
      peakHours: "",
      peak: { input: 0, cached: 0, output: 0 },
      offpeak: { input: 0, cached: 0, output: 0 },
    };
  }
  const currency = entry.currency || "CNY";
  const displayName = entry.model || "";
  const pH = (entry.peak_hours || []).map((r: any) => `${r[0]}-${r[1]}`).join(", ");
  const peak = entry.peak || {
    input: entry.input || 0,
    input_cached: entry.input_cached || entry.input || 0,
    output: entry.output || 0,
  };
  const offpeak = entry.offpeak || peak;
  return {
    currency,
    displayName,
    peakHours: pH,
    peak: {
      input: Number(peak.input) || 0,
      cached: Number(peak.input_cached ?? peak.input) || 0,
      output: Number(peak.output) || 0,
    },
    offpeak: {
      input: Number(offpeak.input) || 0,
      cached: Number(offpeak.input_cached ?? offpeak.input) || 0,
      output: Number(offpeak.output) || 0,
    },
  };
}

function PriceForm({
  initialName,
  initialEntry,
  existingNames,
  onSubmit,
  onCancel,
}: {
  initialName: string;
  initialEntry: PriceTableEntry | null;
  existingNames: string[];
  onSubmit: (name: string, entry: PriceTableEntry) => void;
  onCancel: () => void;
}) {
  const [name, setName] = useState(initialName);
  const norm = normalizeEntry(initialEntry);
  const [currency, setCurrency] = useState(norm.currency);
  const [displayName, setDisplayName] = useState(norm.displayName);
  const [peakHours, setPeakHours] = useState(norm.peakHours || "9-12, 14-18");
  const [peak, setPeak] = useState({
    input: norm.peak.input,
    cached: norm.peak.cached,
    output: norm.peak.output,
  });
  const [offpeak, setOffpeak] = useState({
    input: norm.offpeak.input,
    cached: norm.offpeak.cached,
    output: norm.offpeak.output,
  });
  const isNew = !initialName;

  const submit = () => {
    if (!name.trim()) {
      toast.error("名称不能为空", { description: "请输入价格表名称" });
      return;
    }
    if (isNew && existingNames.includes(name.trim())) {
      toast.error("名称已存在", { description: "请换个名字或改用编辑。" });
      return;
    }
    const ph = peakHours
      .split(/[,，]/)
      .map((s) => s.trim())
      .filter(Boolean)
      .map((p) => {
        const parts = p.split(/[-–]/).map((x) => parseInt(x, 10));
        return parts.length === 2 && parts.every((n) => Number.isFinite(n))
          ? [parts[0], parts[1]]
          : null;
      })
      .filter((x): x is number[] => !!x);

    const entry: PriceTableEntry = {
      currency,
      model: displayName || undefined,
      peak_hours: ph,
      peak: { input: peak.input, input_cached: peak.cached, output: peak.output },
      offpeak: { input: offpeak.input, input_cached: offpeak.cached, output: offpeak.output },
    };
    onSubmit(name.trim(), entry);
  };

  const isSame =
    Math.abs(peak.input - offpeak.input) < 1e-9 &&
    Math.abs(peak.cached - offpeak.cached) < 1e-9 &&
    Math.abs(peak.output - offpeak.output) < 1e-9;

  const perReqSample = (1000 * 0.8 * peak.cached + 1000 * 0.2 * peak.input + 500 * peak.output) / 1e6;

  return (
    <div className="grid gap-5">
      <div className="grid grid-cols-2 gap-4">
        <div>
          <Label>价格表名称（ID）</Label>
          <Input
            value={name}
            onChange={(e) => setName(e.target.value)}
            placeholder="deepseek-v4-flash"
            disabled={!isNew}
          />
        </div>
        <div className="grid grid-cols-2 gap-3">
          <div>
            <Label>币种</Label>
            <Input value={currency} onChange={(e) => setCurrency(e.target.value)} />
          </div>
          <div>
            <Label>备注 / 模型名</Label>
            <Input
              value={displayName}
              onChange={(e) => setDisplayName(e.target.value)}
              placeholder="deepseek-v4-flash"
            />
          </div>
        </div>
      </div>

      <div>
        <Label>高峰时段（北京时，逗号分隔 起点-终点）</Label>
        <Input value={peakHours} onChange={(e) => setPeakHours(e.target.value)} />
      </div>

      <div className="grid grid-cols-2 gap-4">
        <div className="p-4 rounded-xl border border-brand-amber/40 bg-brand-amber/5 space-y-3">
          <div className="flex items-center gap-2">
            <Sparkles className="w-4 h-4 text-brand-amber" />
            <span className="text-sm font-semibold text-brand-amber">
              高峰 {currency}/1M tokens
            </span>
          </div>
          <div className="grid grid-cols-3 gap-3">
            <div>
              <Label>输入（未命中）</Label>
              <Input
                type="number"
                value={peak.input}
                onChange={(e) => setPeak({ ...peak, input: Number(e.target.value) })}
              />
            </div>
            <div>
              <Label>输入（缓存命中）</Label>
              <Input
                type="number"
                value={peak.cached}
                onChange={(e) => setPeak({ ...peak, cached: Number(e.target.value) })}
              />
            </div>
            <div>
              <Label>输出</Label>
              <Input
                type="number"
                value={peak.output}
                onChange={(e) => setPeak({ ...peak, output: Number(e.target.value) })}
              />
            </div>
          </div>
        </div>
        <div className="p-4 rounded-xl border border-brand-cyan/40 bg-brand-cyan/5 space-y-3">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Sparkles className="w-4 h-4 text-brand-cyan" />
              <span className="text-sm font-semibold text-brand-cyan">
                闲时 {currency}/1M tokens
              </span>
            </div>
            <Button
              variant="ghost"
              size="sm"
              onClick={() => setOffpeak({ ...peak })}
              type="button"
            >
              同步高峰价
            </Button>
          </div>
          <div className="grid grid-cols-3 gap-3">
            <div>
              <Label>输入（未命中）</Label>
              <Input
                type="number"
                value={offpeak.input}
                onChange={(e) => setOffpeak({ ...offpeak, input: Number(e.target.value) })}
              />
            </div>
            <div>
              <Label>输入（缓存命中）</Label>
              <Input
                type="number"
                value={offpeak.cached}
                onChange={(e) => setOffpeak({ ...offpeak, cached: Number(e.target.value) })}
              />
            </div>
            <div>
              <Label>输出</Label>
              <Input
                type="number"
                value={offpeak.output}
                onChange={(e) => setOffpeak({ ...offpeak, output: Number(e.target.value) })}
              />
            </div>
          </div>
        </div>
      </div>

      <div className="p-3 rounded-lg border border-brand-borderSubtle bg-slate-900/40 flex items-center justify-between text-xs text-slate-400">
        <div>
          <span className="text-slate-300">示例估算</span>：1K input (80%命中) + 500 output ≈{" "}
          <span className="text-brand-cyan font-semibold">
            {formatMoney(perReqSample, currency)}
          </span>
        </div>
        <div>
          {isSame ? (
            <span className="inline-flex items-center gap-1 px-2 py-1 rounded-md border border-brand-borderSubtle bg-brand-panel2 text-brand-cyan">
              平峰同价
            </span>
          ) : (
            <span className="inline-flex items-center gap-1 px-2 py-1 rounded-md border border-brand-borderSubtle bg-brand-panel2 text-brand-violet">
              启用峰谷计价
            </span>
          )}
        </div>
        <DialogFooter className="!m-0">
          <Button variant="ghost" onClick={onCancel} type="button">
            取消
          </Button>
          <Button variant="primary" onClick={submit} type="button">
            {isNew ? "创建并保存" : "保存修改"}
          </Button>
        </DialogFooter>
      </div>
    </div>
  );
}

export default function PricesPage() {
  const {
    pricesResp,
    fetchPrices,
    upsertPrice,
    deletePrice,
    setBinding,
    deleteBinding,
    loading,
  } = useAdmin();
  const [query, setQuery] = useState("");
  const [bindTarget, setBindTarget] = useState<string | null>(null);
  const [bindRoute, setBindRoute] = useState("");

  useEffect(() => {
    fetchPrices();
  }, [fetchPrices]);

  const list = useMemo(() => {
    const prices = pricesResp?.prices || {};
    const boundRoutes = pricesResp?.bound_routes || {};
    const routeToClaude = pricesResp?.route_to_claude || {};
    return Object.entries(prices)
      .map(([name, entry]) => ({
        name,
        entry,
        routes: boundRoutes[name] || [],
        routeToClaude,
      }))
      .filter(
        (r) =>
          !query.trim() ||
          r.name.toLowerCase().includes(query.toLowerCase()) ||
          (r.entry.model || "").toLowerCase().includes(query.toLowerCase()),
      );
  }, [pricesResp, query]);

  const allRoutes = pricesResp?.all_routes ?? [];
  const priceBindings = pricesResp?.price_bindings ?? {};

  // Routes not yet bound to any price table
  const unboundRoutes = allRoutes.filter((r) => !(r in priceBindings));

  const doBind = async (priceName: string, routeKey: string) => {
    if (!routeKey) return;
    const ok = await setBinding(routeKey, priceName);
    if (ok) {
      setBindTarget(null);
      setBindRoute("");
    }
  };

  return (
    <div className="space-y-6">
      <div className="flex items-end justify-between gap-4 flex-wrap">
        <div>
          <h1 className="text-2xl font-bold text-white tracking-wide glow-text">
            价格表
          </h1>
          <p className="text-sm text-slate-400 mt-1">
            命名价格表，多个路由可共享同一张价格表。
          </p>
        </div>
        <div className="flex items-center gap-3">
          <div className="relative w-72">
            <Search className="w-4 h-4 absolute left-3 top-2.5 text-slate-500" />
            <Input
              className="pl-9"
              placeholder="搜索名称 / 备注"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
            />
          </div>
          <Button variant="ghost" size="icon" onClick={fetchPrices} disabled={loading}>
            <RefreshCw className={cn("w-4 h-4", loading && "animate-spin")} />
          </Button>
          <Dialog>
            <DialogTrigger asChild>
              <Button variant="primary">
                <Plus className="w-4 h-4" />
                新建价格表
              </Button>
            </DialogTrigger>
            <DialogContent>
              <DialogHeader>
                <DialogTitle className="flex items-center gap-2">
                  <Banknote className="w-5 h-5 text-brand-cyan" />
                  新建价格表
                </DialogTitle>
              </DialogHeader>
              <PriceForm
                initialName=""
                initialEntry={null}
                existingNames={Object.keys(pricesResp?.prices || {})}
                onCancel={() => {}}
                onSubmit={async (n, e) => {
                  const ok = await upsertPrice(n, e);
                  if (ok) {
                    // Close dialog by re-rendering; DialogTrigger handles it
                  }
                }}
              />
            </DialogContent>
          </Dialog>
        </div>
      </div>

      <div className="grid grid-cols-1 xl:grid-cols-2 2xl:grid-cols-3 gap-5">
        {list.length === 0 ? (
          <Card className="xl:col-span-2 2xl:col-span-3">
            <CardContent className="py-14 text-center text-slate-400">
              <Layers className="mx-auto w-10 h-10 text-brand-cyan/60 mb-3" />
              <div className="text-white font-semibold mb-1">暂无价格表</div>
              <div>点击右上角「新建价格表」创建第一条计费价目。</div>
            </CardContent>
          </Card>
        ) : (
          list.map(({ name, entry, routes, routeToClaude }) => {
            const n = normalizeEntry(entry);
            return (
              <Card key={name}>
                <CardHeader>
                  <div className="flex items-start justify-between gap-3">
                    <div>
                      <CardTitle className="flex items-center gap-2">
                        <span className="font-mono text-base text-brand-cyan">{name}</span>
                        {routes.length > 0 && (
                          <span className="text-xs px-2 py-0.5 rounded-md border border-brand-green/30 bg-brand-green/10 text-brand-green whitespace-nowrap">
                            绑定 {routes.length} 个路由
                          </span>
                        )}
                      </CardTitle>
                      <CardDescription className="mt-1">
                        {n.displayName || "未命名"} · {n.currency} · 高峰 {n.peakHours || "未设置"}
                      </CardDescription>
                    </div>
                    <div className="flex gap-1">
                      <Dialog>
                        <DialogTrigger asChild>
                          <Button variant="ghost" size="icon">
                            <RefreshCw className="w-4 h-4" />
                          </Button>
                        </DialogTrigger>
                        <DialogContent>
                          <DialogHeader>
                            <DialogTitle className="flex items-center gap-2">
                              <Banknote className="w-5 h-5 text-brand-cyan" />
                              编辑 {name}
                            </DialogTitle>
                          </DialogHeader>
                          <PriceForm
                            initialName={name}
                            initialEntry={entry}
                            existingNames={Object.keys(pricesResp?.prices || {})}
                            onCancel={() => {}}
                            onSubmit={async (n2, e2) => {
                              await upsertPrice(n2, e2);
                            }}
                          />
                        </DialogContent>
                      </Dialog>
                      <Button
                        variant="ghost"
                        size="icon"
                        onClick={async () => {
                          if (confirm(`确定删除价格表 ${name}？绑定的路由将自动解绑。`))
                            await deletePrice(name);
                        }}
                      >
                        <Trash2 className="w-4 h-4 text-rose-400" />
                      </Button>
                    </div>
                  </div>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="overflow-hidden rounded-md border border-brand-borderSubtle text-sm">
                    <table className="w-full">
                      <thead>
                        <tr className="text-slate-400 bg-slate-900/50">
                          <th className="px-3 py-2.5 text-left font-medium">单价 / 百万 token</th>
                          <th className="px-3 py-2.5 text-right font-medium text-brand-amber">高峰</th>
                          <th className="px-3 py-2.5 text-right font-medium text-brand-cyan">闲时</th>
                        </tr>
                      </thead>
                      <tbody className="tabular-nums">
                        <tr className="border-t border-brand-borderSubtle/60">
                          <td className="px-3 py-2 text-slate-300">输入</td>
                          <td className="px-3 py-2 text-right text-brand-amber">{n.peak.input}</td>
                          <td className="px-3 py-2 text-right text-brand-cyan">{n.offpeak.input}</td>
                        </tr>
                        <tr className="border-t border-brand-borderSubtle/60">
                          <td className="px-3 py-2 text-slate-300">缓存命中</td>
                          <td className="px-3 py-2 text-right text-brand-amber">{n.peak.cached}</td>
                          <td className="px-3 py-2 text-right text-brand-cyan">{n.offpeak.cached}</td>
                        </tr>
                        <tr className="border-t border-brand-borderSubtle/60">
                          <td className="px-3 py-2 text-slate-300">输出</td>
                          <td className="px-3 py-2 text-right text-brand-amber">{n.peak.output}</td>
                          <td className="px-3 py-2 text-right text-brand-cyan">{n.offpeak.output}</td>
                        </tr>
                      </tbody>
                    </table>
                  </div>

                  {/* Bound routes */}
                  <div className="pt-3 border-t border-brand-borderSubtle/60">
                    <div className="flex items-center justify-between mb-2">
                      <div className="text-sm text-slate-400">已绑定路由</div>
                      <Button
                        variant="ghost"
                        size="sm"
                        onClick={() => setBindTarget(name)}
                      >
                        <Plus className="w-3 h-3" />
                        添加路由
                      </Button>
                    </div>
                    {routes.length === 0 ? (
                      <div className="text-xs text-slate-500">暂无绑定的路由</div>
                    ) : (
                      <div className="space-y-1.5">
                        {routes.map((rk) => {
                          const claudeModels = routeToClaude[rk] || [];
                          return (
                            <div
                              key={rk}
                              className="flex items-center justify-between p-2 rounded-md bg-slate-900/40 border border-brand-borderSubtle"
                            >
                              <div>
                                <div className="text-sm font-mono text-slate-200">{rk}</div>
                                {claudeModels.length > 0 && (
                                  <div className="text-[10px] text-slate-500 mt-0.5">
                                    ← {claudeModels.join(", ")}
                                  </div>
                                )}
                              </div>
                              <button
                                onClick={() => deleteBinding(rk)}
                                className="text-slate-500 hover:text-rose-400"
                                title="解绑"
                              >
                                <Unlink className="w-3.5 h-3.5" />
                              </button>
                            </div>
                          );
                        })}
                      </div>
                    )}
                  </div>

                  {/* Add binding dialog */}
                  {bindTarget === name && (
                    <Dialog open onOpenChange={(o) => !o && setBindTarget(null)}>
                      <DialogContent className="max-w-md">
                        <DialogHeader>
                          <DialogTitle className="flex items-center gap-2">
                            <Link2 className="w-5 h-5 text-brand-cyan" />
                            添加路由绑定 → {name}
                          </DialogTitle>
                        </DialogHeader>
                        <div className="space-y-3">
                          <div>
                            <Label>选择路由（provider/model_id）</Label>
                            <select
                              className="flex h-9 w-full rounded-md border border-brand-borderSubtle bg-slate-950/40 px-3 py-1 text-sm text-slate-100"
                              value={bindRoute}
                              onChange={(e) => setBindRoute(e.target.value)}
                            >
                              <option value="">— 选择路由 —</option>
                              {unboundRoutes.map((r) => (
                                <option key={r} value={r}>
                                  {r}
                                </option>
                              ))}
                            </select>
                          </div>
                          {unboundRoutes.length === 0 && (
                            <div className="text-xs text-slate-500">
                              所有路由都已绑定价格表。
                            </div>
                          )}
                        </div>
                        <DialogFooter>
                          <Button
                            variant="ghost"
                            onClick={() => setBindTarget(null)}
                            type="button"
                          >
                            取消
                          </Button>
                          <Button
                            variant="primary"
                            onClick={() => doBind(name, bindRoute)}
                            disabled={!bindRoute}
                            type="button"
                          >
                            <Link2 className="w-4 h-4" />
                            绑定
                          </Button>
                        </DialogFooter>
                      </DialogContent>
                    </Dialog>
                  )}
                </CardContent>
              </Card>
            );
          })
        )}
      </div>
    </div>
  );
}
