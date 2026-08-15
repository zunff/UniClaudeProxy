import { useEffect, useMemo, useState } from "react";
import { toast } from "sonner";
import {
  Banknote,
  Calculator,
  Layers,
  Link2,
  Pencil,
  Plus,
  RefreshCw,
  Search,
  Sun,
  Moon,
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
import { SelectField } from "@/components/ui/select";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import { cn, formatMoney, toCny, DEFAULT_FX_TO_CNY } from "@/lib/utils";

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

function UnitCell({
  value,
  currency,
  fxToCny,
  className,
}: {
  value: number;
  currency: string;
  fxToCny: Record<string, number>;
  className?: string;
}) {
  const isForeign = currency.toUpperCase() !== "CNY";
  return (
    <td className={cn("px-3 py-2 text-right tabular-nums font-mono font-bold", className)}>
      <div>{formatMoney(toCny(value, currency, fxToCny), "CNY")}</div>
      {isForeign && (
        <div className="text-[10px] font-normal text-slate-500">
          {formatMoney(value, currency)}
        </div>
      )}
    </td>
  );
}

function PriceForm({
  initialName,
  initialEntry,
  existingNames,
  fxToCny,
  onSubmit,
  onCancel,
}: {
  initialName: string;
  initialEntry: PriceTableEntry | null;
  existingNames: string[];
  fxToCny: Record<string, number>;
  onSubmit: (name: string, entry: PriceTableEntry) => void;
  onCancel: () => void;
}) {
  const [name, setName] = useState(initialName);
  const norm = normalizeEntry(initialEntry);
  const [currency, setCurrency] = useState(norm.currency);
  const [displayName, setDisplayName] = useState(norm.displayName);
  const [peakHours, setPeakHours] = useState(
    initialName ? norm.peakHours : norm.peakHours || "9-12, 14-18"
  );
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
      toast.error("价格表名称不能为空");
      return;
    }
    if (isNew && existingNames.includes(name.trim())) {
      toast.error("价格表名称已存在");
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

  const perReqSample =
    (1000 * 0.8 * peak.cached + 1000 * 0.2 * peak.input + 500 * peak.output) / 1e6;
  const perReqCny = toCny(perReqSample, currency, fxToCny);
  const isForeign = currency.toUpperCase() !== "CNY";

  return (
    <div className="space-y-4 font-mono">
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
        <div>
          <Label>价格表名称 (ID)</Label>
          <Input
            value={name}
            onChange={(e) => setName(e.target.value)}
            placeholder="deepseek-v4-flash"
            disabled={!isNew}
          />
        </div>
        <div className="grid grid-cols-2 gap-2">
          <div>
            <Label>标价币种</Label>
            <SelectField
              value={currency}
              onValueChange={setCurrency}
              placeholder="选择币种"
              options={[
                { value: "CNY", label: "CNY 人民币" },
                { value: "USD", label: "USD 美元" },
              ]}
            />
          </div>
          <div>
            <Label>说明备注</Label>
            <Input
              value={displayName}
              onChange={(e) => setDisplayName(e.target.value)}
              placeholder="DeepSeek V4"
            />
          </div>
        </div>
      </div>

      <div>
        <Label>高峰时段 (如 9-12, 14-18)</Label>
        <Input
          value={peakHours}
          onChange={(e) => setPeakHours(e.target.value)}
          placeholder="9-12, 14-18"
        />
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
        {/* Peak Rates */}
        <div className="p-3.5 rounded-lg border border-amber-500/30 bg-amber-500/5 space-y-2.5">
          <div className="flex items-center gap-1.5 text-xs font-semibold text-amber-400">
            <Sun className="w-3.5 h-3.5" />
            <span>高峰单价 ({currency} / 1M Tokens)</span>
          </div>
          <div className="grid grid-cols-3 gap-2">
            <div>
              <Label className="text-[11px]">输入 (未命中)</Label>
              <Input
                type="number"
                step="any"
                value={peak.input}
                onChange={(e) => setPeak({ ...peak, input: Number(e.target.value) })}
              />
            </div>
            <div>
              <Label className="text-[11px] text-emerald-400">缓存命中</Label>
              <Input
                type="number"
                step="any"
                value={peak.cached}
                onChange={(e) => setPeak({ ...peak, cached: Number(e.target.value) })}
              />
            </div>
            <div>
              <Label className="text-[11px] text-purple-400">输出单价</Label>
              <Input
                type="number"
                step="any"
                value={peak.output}
                onChange={(e) => setPeak({ ...peak, output: Number(e.target.value) })}
              />
            </div>
          </div>
        </div>

        {/* Off-peak Rates */}
        <div className="p-3.5 rounded-lg border border-cyan-500/30 bg-cyan-500/5 space-y-2.5">
          <div className="flex items-center gap-1.5 text-xs font-semibold text-cyan-400">
            <Moon className="w-3.5 h-3.5" />
            <span>闲时单价 ({currency} / 1M Tokens)</span>
          </div>
          <div className="grid grid-cols-3 gap-2">
            <div>
              <Label className="text-[11px]">输入 (未命中)</Label>
              <Input
                type="number"
                step="any"
                value={offpeak.input}
                onChange={(e) => setOffpeak({ ...offpeak, input: Number(e.target.value) })}
              />
            </div>
            <div>
              <Label className="text-[11px] text-emerald-400">缓存命中</Label>
              <Input
                type="number"
                step="any"
                value={offpeak.cached}
                onChange={(e) => setOffpeak({ ...offpeak, cached: Number(e.target.value) })}
              />
            </div>
            <div>
              <Label className="text-[11px] text-purple-400">输出单价</Label>
              <Input
                type="number"
                step="any"
                value={offpeak.output}
                onChange={(e) => setOffpeak({ ...offpeak, output: Number(e.target.value) })}
              />
            </div>
          </div>
        </div>
      </div>

      {/* Estimation */}
      <div className="p-3 rounded-lg border border-brand-borderSubtle bg-slate-950/60 flex items-center justify-between text-xs">
        <div className="text-slate-300">
          <span>
            估算 (1K input 80%命中 + 500 output):{" "}
            <span className="text-emerald-400 font-bold">
              {formatMoney(perReqCny, "CNY")}
            </span>
            {isForeign && (
              <span className="text-slate-500 ml-1">
                (官网 {formatMoney(perReqSample, currency)})
              </span>
            )}
          </span>
        </div>
        <div>
          {isSame ? (
            <span className="px-2 py-0.5 rounded border border-slate-700 bg-slate-800 text-slate-300 text-[10px]">
              平峰同价
            </span>
          ) : (
            <span className="px-2 py-0.5 rounded border border-purple-500/30 bg-purple-500/10 text-purple-300 text-[10px]">
              峰谷分时
            </span>
          )}
        </div>
      </div>

      <DialogFooter>
        <Button variant="ghost" onClick={onCancel} type="button">
          取消
        </Button>
        <Button variant="primary" onClick={submit} type="button">
          {isNew ? "创建并保存" : "保存修改"}
        </Button>
      </DialogFooter>
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
    setFxToCny,
    loading,
  } = useAdmin();
  const [query, setQuery] = useState("");
  const [bindTarget, setBindTarget] = useState<string | null>(null);
  const [bindRoute, setBindRoute] = useState("");
  const fxToCny = pricesResp?.fx_to_cny ?? DEFAULT_FX_TO_CNY;
  const [fxDraft, setFxDraft] = useState(String(fxToCny.USD ?? DEFAULT_FX_TO_CNY.USD));

  useEffect(() => {
    fetchPrices();
  }, [fetchPrices]);

  useEffect(() => {
    setFxDraft(String(fxToCny.USD ?? DEFAULT_FX_TO_CNY.USD));
  }, [fxToCny.USD]);

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
          (r.entry.model || "").toLowerCase().includes(query.toLowerCase())
      );
  }, [pricesResp, query]);

  const allRoutes = pricesResp?.all_routes ?? [];
  const priceBindings = pricesResp?.price_bindings ?? {};

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
    <div className="space-y-6 w-full">
      {/* Header */}
      <div className="flex items-end justify-between gap-4 flex-wrap">
        <div>
          <div className="flex items-center gap-2.5">
            <h1 className="text-2xl font-bold text-white tracking-tight">
              价格表
            </h1>
            <span className="px-2 py-0.5 rounded text-[11px] font-mono border border-emerald-500/30 bg-emerald-500/10 text-emerald-400 font-semibold">
              PRICES
            </span>
          </div>
          <p className="text-xs text-slate-400 mt-1">
            支持原币与分时计价，系统根据汇率折算为人民币核算。
          </p>
        </div>

        <div className="flex items-center gap-3 flex-wrap">
          {/* USD -> CNY FX Rate Widget */}
          <div className="relative flex items-center">
            <span className="pointer-events-none absolute left-2.5 text-[11px] font-mono font-bold text-cyan-400">
              USD→CNY
            </span>
            <Input
              className="w-32 pl-[4.5rem] tabular-nums font-mono text-xs font-bold text-white h-9"
              value={fxDraft}
              onChange={(e) => setFxDraft(e.target.value)}
              onBlur={async () => {
                const n = Number(fxDraft);
                if (!Number.isFinite(n) || n <= 0) {
                  setFxDraft(String(fxToCny.USD ?? DEFAULT_FX_TO_CNY.USD));
                  return;
                }
                if (n !== (fxToCny.USD ?? DEFAULT_FX_TO_CNY.USD)) {
                  await setFxToCny({ ...fxToCny, USD: n });
                }
              }}
            />
          </div>

          <div className="relative w-64">
            <Search className="w-4 h-4 absolute left-3 top-2.5 text-slate-500" />
            <Input
              className="pl-9 h-9"
              placeholder="搜索价格表 / 备注"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
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

          <Dialog>
            <DialogTrigger asChild>
              <Button variant="primary" size="sm">
                <Plus className="w-4 h-4" />
                新建价格表
              </Button>
            </DialogTrigger>
            <DialogContent>
              <DialogHeader>
                <DialogTitle className="flex items-center gap-2">
                  <Banknote className="w-5 h-5 text-emerald-400" />
                  新建价格表
                </DialogTitle>
                <DialogDescription>
                  录入官方定价（支持原币与峰谷分时模式）
                </DialogDescription>
              </DialogHeader>
              <PriceForm
                initialName=""
                initialEntry={null}
                existingNames={Object.keys(pricesResp?.prices || {})}
                fxToCny={fxToCny}
                onCancel={() => {}}
                onSubmit={async (n, e) => {
                  await upsertPrice(n, e);
                }}
              />
            </DialogContent>
          </Dialog>
        </div>
      </div>

      {/* Pricing Cards Grid - 4 columns on wide screens */}
      <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 2xl:grid-cols-4 gap-5">
        {list.length === 0 ? (
          <Card className="col-span-full">
            <CardContent className="py-12 text-center text-slate-500 font-mono text-xs">
              <Layers className="mx-auto w-8 h-8 text-slate-600 mb-2" />
              <div>暂无价格表</div>
            </CardContent>
          </Card>
        ) : (
          list.map(({ name, entry, routes, routeToClaude }) => {
            const n = normalizeEntry(entry);
            return (
              <Card key={name} className="hover:border-slate-700">
                <CardHeader>
                  <div className="flex items-start justify-between gap-2">
                    <div className="min-w-0 flex-1">
                      <div className="flex items-center gap-2 flex-wrap">
                        <CardTitle className="font-mono text-base font-bold text-emerald-300 truncate">
                          {name}
                        </CardTitle>
                        {routes.length > 0 && (
                          <span className="text-[10px] font-mono px-1.5 py-0.2 rounded border border-emerald-500/40 bg-emerald-500/10 text-emerald-300 whitespace-nowrap font-semibold">
                            绑定 {routes.length} 条路由
                          </span>
                        )}
                      </div>
                      <CardDescription className="mt-1 font-mono text-xs text-slate-400">
                        {n.displayName || "未命名备注"} · 币种 {n.currency}
                        {n.peakHours ? ` · 高峰: ${n.peakHours}` : ""}
                      </CardDescription>
                    </div>

                    <div className="flex items-center gap-1">
                      <Dialog>
                        <DialogTrigger asChild>
                          <Button
                            variant="ghost"
                            size="icon"
                            className="h-7 w-7 text-slate-400 hover:text-white"
                          >
                            <Pencil className="w-3.5 h-3.5" />
                          </Button>
                        </DialogTrigger>
                        <DialogContent>
                          <DialogHeader>
                            <DialogTitle className="flex items-center gap-2">
                              <Banknote className="w-5 h-5 text-emerald-400" />
                              编辑价格表 · {name}
                            </DialogTitle>
                            <DialogDescription>
                              修改模型单价与高峰时段
                            </DialogDescription>
                          </DialogHeader>
                          <PriceForm
                            initialName={name}
                            initialEntry={entry}
                            existingNames={Object.keys(pricesResp?.prices || {})}
                            fxToCny={fxToCny}
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
                        className="h-7 w-7 text-slate-400 hover:text-rose-400"
                        onClick={async () => {
                          if (
                            confirm(
                              `确定删除价格表 ${name}？已关联的路由将自动解除绑定。`
                            )
                          )
                            await deletePrice(name);
                        }}
                      >
                        <Trash2 className="w-3.5 h-3.5" />
                      </Button>
                    </div>
                  </div>
                </CardHeader>

                <CardContent className="space-y-3">
                  {/* Pricing Matrix Table */}
                  <div className="overflow-hidden rounded-lg border border-brand-borderSubtle bg-slate-950/70 text-xs font-mono">
                    <table className="w-full">
                      <thead>
                        <tr className="text-slate-400 bg-slate-900/60 border-b border-brand-borderSubtle">
                          <th className="px-3 py-2 text-left font-medium">单价 / 1M (CNY)</th>
                          <th className="px-3 py-2 text-right font-medium text-amber-400">
                            <span className="inline-flex items-center gap-1">
                              <Sun className="w-3 h-3" /> 高峰
                            </span>
                          </th>
                          <th className="px-3 py-2 text-right font-medium text-cyan-400">
                            <span className="inline-flex items-center gap-1">
                              <Moon className="w-3 h-3" /> 闲时
                            </span>
                          </th>
                        </tr>
                      </thead>
                      <tbody>
                        <tr className="border-b border-brand-borderSubtle/50 clean-table-row">
                          <td className="px-3 py-2 text-slate-300">输入 (未命中)</td>
                          <UnitCell
                            value={n.peak.input}
                            currency={n.currency}
                            fxToCny={fxToCny}
                            className="text-amber-400"
                          />
                          <UnitCell
                            value={n.offpeak.input}
                            currency={n.currency}
                            fxToCny={fxToCny}
                            className="text-cyan-400"
                          />
                        </tr>
                        <tr className="border-b border-brand-borderSubtle/50 clean-table-row">
                          <td className="px-3 py-2 text-emerald-400">输入 (缓存命中)</td>
                          <UnitCell
                            value={n.peak.cached}
                            currency={n.currency}
                            fxToCny={fxToCny}
                            className="text-amber-400"
                          />
                          <UnitCell
                            value={n.offpeak.cached}
                            currency={n.currency}
                            fxToCny={fxToCny}
                            className="text-cyan-400"
                          />
                        </tr>
                        <tr className="clean-table-row">
                          <td className="px-3 py-2 text-purple-400">输出单价</td>
                          <UnitCell
                            value={n.peak.output}
                            currency={n.currency}
                            fxToCny={fxToCny}
                            className="text-amber-400"
                          />
                          <UnitCell
                            value={n.offpeak.output}
                            currency={n.currency}
                            fxToCny={fxToCny}
                            className="text-cyan-400"
                          />
                        </tr>
                      </tbody>
                    </table>
                  </div>

                  {/* Bound Routes */}
                  <div className="pt-2 border-t border-brand-borderSubtle">
                    <div className="flex items-center justify-between mb-1.5 text-xs font-mono">
                      <span className="text-slate-400">已绑定路由:</span>
                      <Button
                        variant="ghost"
                        size="sm"
                        className="h-6 text-xs font-mono px-2 text-emerald-400 hover:text-emerald-300"
                        onClick={() => setBindTarget(name)}
                      >
                        <Plus className="w-3 h-3" />
                        绑定路由
                      </Button>
                    </div>

                    {routes.length === 0 ? (
                      <div className="text-xs font-mono text-slate-500 py-1.5 text-center rounded border border-dashed border-brand-borderSubtle">
                        暂未绑定路由
                      </div>
                    ) : (
                      <div className="space-y-1 font-mono">
                        {routes.map((rk) => {
                          const claudeModels = routeToClaude[rk] || [];
                          return (
                            <div
                              key={rk}
                              className="flex items-center justify-between p-1.5 rounded-md bg-slate-950/60 border border-brand-borderSubtle"
                            >
                              <div className="min-w-0 flex-1">
                                <div className="text-xs font-semibold text-slate-200 truncate">
                                  {rk}
                                </div>
                                {claudeModels.length > 0 && (
                                  <div className="text-[10px] text-slate-500 truncate">
                                    ← 映射: {claudeModels.join(", ")}
                                  </div>
                                )}
                              </div>
                              <button
                                onClick={() => deleteBinding(rk)}
                                className="text-slate-500 hover:text-rose-400 p-1 ml-1"
                                title="解除绑定"
                              >
                                <Unlink className="w-3.5 h-3.5" />
                              </button>
                            </div>
                          );
                        })}
                      </div>
                    )}
                  </div>

                  {/* Modal */}
                  {bindTarget === name && (
                    <Dialog open onOpenChange={(o) => !o && setBindTarget(null)}>
                      <DialogContent className="max-w-md">
                        <DialogHeader>
                          <DialogTitle className="flex items-center gap-2">
                            <Link2 className="w-5 h-5 text-emerald-400" />
                            绑定后端路由 → {name}
                          </DialogTitle>
                          <DialogDescription>
                            选择要绑定到该价格表的路由条目
                          </DialogDescription>
                        </DialogHeader>
                        <div className="space-y-4 font-mono">
                          <div>
                            <Label>选择路由</Label>
                            <SelectField
                              className="font-mono"
                              value={bindRoute}
                              onValueChange={setBindRoute}
                              placeholder="— 选择目标路由 —"
                              options={unboundRoutes}
                            />
                          </div>
                          {unboundRoutes.length === 0 && (
                            <div className="text-xs text-slate-500 p-2.5 rounded border border-brand-borderSubtle bg-slate-950/60 font-mono">
                              所有路由均已绑定价格表。
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
                            确认绑定
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
