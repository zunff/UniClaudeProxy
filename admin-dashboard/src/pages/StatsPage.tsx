import { useEffect, useMemo, useRef, useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Legend,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import {
  Activity,
  AlertTriangle,
  CheckCircle2,
  Clock,
  Coins,
  Cpu,
  Database,
  DollarSign,
  Flame,
  RefreshCw,
  Server,
  TrendingUp,
  XCircle,
  Zap,
} from "lucide-react";
import { useAdmin } from "@/store/admin";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Input } from "@/components/ui/input";
import { cn, formatMoney, formatNumber, formatShort, todayStr } from "@/lib/utils";

type RangeKey = "today" | "yesterday" | "7d" | "30d" | "custom";
const RANGES: { key: RangeKey; label: string }[] = [
  { key: "today", label: "今日" },
  { key: "yesterday", label: "昨日" },
  { key: "7d", label: "近 7 天" },
  { key: "30d", label: "近 30 天" },
  { key: "custom", label: "自定义" },
];

function StatTile({
  icon: Icon,
  title,
  value,
  sub,
  color,
  topBorder,
  loading,
}: {
  icon: any;
  title: string;
  value: string;
  sub?: string;
  color: string;
  topBorder: string;
  loading?: boolean;
}) {
  return (
    <div
      className={cn(
        "rounded-xl border border-brand-borderSubtle bg-brand-panel p-5 transition-all duration-150 hover:border-slate-700 min-w-0 border-t-2",
        topBorder
      )}
    >
      <div className="flex items-center justify-between gap-2">
        <span className="text-xs font-mono text-slate-400 uppercase tracking-wider">
          {title}
        </span>
        <div className="w-8 h-8 rounded-lg bg-slate-950 border border-slate-800 flex items-center justify-center shrink-0">
          <Icon className={cn("w-4 h-4", color)} />
        </div>
      </div>

      <div className="mt-3">
        {loading ? (
          <div className="h-8 w-28 skeleton my-1" />
        ) : (
          <div className={cn("text-2xl lg:text-3xl font-bold font-mono tracking-tight tabular-nums truncate", color)}>
            {value}
          </div>
        )}
      </div>

      {sub && (
        <div className="mt-2 text-xs text-slate-400 font-mono truncate">
          {loading ? <div className="h-3.5 w-36 skeleton mt-1" /> : sub}
        </div>
      )}
    </div>
  );
}

const AUTO_REFRESH_MS = 30_000;

export default function StatsPage() {
  const { stats, statsRange, setStatsRange, fetchStats, loading, pricesResp } =
    useAdmin();
  const [start, setStart] = useState(todayStr(-6));
  const [end, setEnd] = useState(todayStr(0));
  const [autoRefresh, setAutoRefresh] = useState(true);
  const [lastUpdated, setLastUpdated] = useState<Date | null>(null);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const loadingRef = useRef(loading);

  useEffect(() => {
    loadingRef.current = loading;
  }, [loading]);

  useEffect(() => {
    fetchStats().then(() => setLastUpdated(new Date()));
  }, []);

  useEffect(() => {
    if (!autoRefresh) return;

    const tick = () => {
      if (document.visibilityState !== "visible") return;
      if (loadingRef.current) return;
      fetchStats().then(() => setLastUpdated(new Date()));
    };

    timerRef.current = setInterval(tick, AUTO_REFRESH_MS);

    const onVis = () => {
      if (document.visibilityState === "visible") {
        tick();
      }
    };
    document.addEventListener("visibilitychange", onVis);

    return () => {
      if (timerRef.current) clearInterval(timerRef.current);
      document.removeEventListener("visibilitychange", onVis);
    };
  }, [autoRefresh, fetchStats]);

  const handleRefresh = () => {
    fetchStats().then(() => setLastUpdated(new Date()));
  };

  const formatLastUpdated = () => {
    if (!lastUpdated) return "";
    const diff = Math.floor((Date.now() - lastUpdated.getTime()) / 1000);
    if (diff < 5) return "刚刚";
    if (diff < 60) return `${diff} 秒前`;
    if (diff < 3600) return `${Math.floor(diff / 60)} 分钟前`;
    return lastUpdated.toLocaleTimeString("zh-CN", {
      hour: "2-digit",
      minute: "2-digit",
    });
  };

  const totals = stats?.total.totals;
  const cache = stats?.total.cache;
  const dateKeys = stats?.date_keys ?? [];
  const recent = stats?.recent ?? [];
  const currency = "CNY";

  const missingPriceRoutes = useMemo(() => {
    const byModel = stats?.total.by_model ?? {};
    const bindings = pricesResp?.price_bindings ?? {};
    const priceKeys = new Set([
      ...Object.keys(pricesResp?.prices ?? {}),
      ...Object.keys(bindings),
    ]);
    return Object.keys(byModel).filter((k) => !priceKeys.has(k));
  }, [stats, pricesResp]);

  const chartData = useMemo(() => {
    return dateKeys.map((d) => {
      const b = stats?.per_day?.[d]?.totals;
      return {
        date: d.slice(5),
        请求数: b?.requests ?? 0,
        输入Tokens: b?.input_tokens ?? 0,
        输出Tokens: b?.output_tokens ?? 0,
        命中Tokens: b?.cache_read_tokens ?? 0,
        成本: Number((b?.cost ?? 0).toFixed(4)),
      };
    });
  }, [stats, dateKeys]);

  const modelData = useMemo(() => {
    const bm = stats?.total.by_model ?? {};
    return Object.entries(bm).map(([key, v]) => ({
      name: key,
      requests: v.requests,
      input: v.input_tokens,
      output: v.output_tokens,
      cache: v.cache_read_tokens,
      cost: Number((v.cost ?? 0).toFixed(4)),
    }));
  }, [stats]);

  return (
    <div className="space-y-6 w-full">
      {/* Header */}
      <div className="flex items-center justify-between gap-4 flex-wrap">
        <div>
          <div className="flex items-center gap-2">
            <h1 className="text-2xl font-bold text-white tracking-tight">
              使用统计
            </h1>
            <span className="px-2 py-0.5 rounded text-[11px] font-mono border border-cyan-500/30 bg-cyan-500/10 text-cyan-400 font-semibold">
              SQLite
            </span>
          </div>
          <p className="text-xs text-slate-400 mt-1 flex items-center gap-2">
            <span>用量、缓存命中率与调用成本分析</span>
            {lastUpdated && (
              <span className="text-slate-500 font-mono">
                · 更新于: {formatLastUpdated()}
              </span>
            )}
          </p>
        </div>

        <div className="flex items-center gap-3">
          <button
            onClick={() => setAutoRefresh((v) => !v)}
            className={cn(
              "inline-flex items-center gap-2 px-3 py-1.5 rounded-lg border text-xs font-mono transition-colors",
              autoRefresh
                ? "border-cyan-500/40 bg-cyan-500/10 text-cyan-400 font-medium"
                : "border-brand-borderSubtle bg-brand-panel2 text-slate-400 hover:text-slate-200"
            )}
            title={autoRefresh ? "自动定时刷新已开启 (30s)" : "自动定时刷新已关闭"}
          >
            <span
              className={cn(
                "w-1.5 h-1.5 rounded-full",
                autoRefresh ? "bg-cyan-400 animate-pulse" : "bg-slate-600"
              )}
            />
            <span>自动刷新 {autoRefresh ? "ON" : "OFF"}</span>
          </button>

          <Button
            variant="secondary"
            size="sm"
            onClick={handleRefresh}
            disabled={loading}
            className="font-mono text-xs"
          >
            <RefreshCw className={cn("w-3.5 h-3.5 text-cyan-400", loading && "animate-spin")} />
            <span>刷新</span>
          </Button>
        </div>
      </div>

      {/* Range Tabs */}
      <div className="flex items-center justify-between gap-4 flex-wrap p-1.5 rounded-xl border border-brand-borderSubtle bg-brand-panel">
        <Tabs
          value={statsRange}
          onValueChange={(v) =>
            setStatsRange(
              v as RangeKey,
              statsRange === "custom" ? { start, end } : undefined
            )
          }
        >
          <TabsList className="bg-slate-950/60">
            {RANGES.map((r) => (
              <TabsTrigger key={r.key} value={r.key}>
                {r.label}
              </TabsTrigger>
            ))}
          </TabsList>
        </Tabs>

        {statsRange === "custom" && (
          <div className="flex items-center gap-2 px-2 py-1 font-mono text-xs">
            <span className="text-slate-400">时间:</span>
            <Input
              type="date"
              value={start}
              onChange={(e) => setStart(e.target.value)}
              className="w-36 h-8 text-xs"
            />
            <span className="text-slate-500">至</span>
            <Input
              type="date"
              value={end}
              onChange={(e) => setEnd(e.target.value)}
              className="w-36 h-8 text-xs"
            />
            <Button
              variant="primary"
              size="sm"
              className="h-8 text-xs"
              onClick={() => setStatsRange("custom", { start, end })}
            >
              应用
            </Button>
          </div>
        )}
      </div>

      {/* Warning Banner */}
      {missingPriceRoutes.length > 0 && (
        <div className="rounded-xl border border-amber-500/30 bg-amber-500/10 p-4 flex items-start gap-3">
          <AlertTriangle className="w-4 h-4 text-amber-400 shrink-0 mt-0.5" />
          <div className="text-xs text-slate-300">
            <span className="font-semibold text-amber-300">
              提示：部分模型未绑定价格表
            </span>
            <div className="mt-0.5 text-slate-400 font-mono">
              以下路由因缺失价格表，成本无法计入：
              <span className="text-rose-400 font-bold">
                {" "}
                {missingPriceRoutes.join(", ")}
              </span>
              。可在「模型配置」页面关联价格表。
            </div>
          </div>
        </div>
      )}

      {/* Stat Tiles - 6 Columns / Responsive */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-6 gap-4">
        <StatTile
          icon={Server}
          title="请求总数"
          value={formatNumber(totals?.requests)}
          sub={`成功 ${formatNumber(totals?.success)} 次 (${
            totals?.requests
              ? Math.round(((totals.success ?? 0) / totals.requests) * 100)
              : 0
          }%)`}
          color="text-cyan-400"
          topBorder="border-t-cyan-500"
          loading={loading && !totals}
        />
        <StatTile
          icon={Database}
          title="输入 Tokens"
          value={formatShort(totals?.input_tokens)}
          sub={`命中 ${formatShort(totals?.cache_read_tokens)}`}
          color="text-blue-400"
          topBorder="border-t-blue-500"
          loading={loading && !totals}
        />
        <StatTile
          icon={TrendingUp}
          title="输出 Tokens"
          value={formatShort(totals?.output_tokens)}
          sub={`平均 ${
            totals?.requests
              ? formatNumber(
                  Math.round((totals.output_tokens ?? 0) / totals.requests)
                )
              : 0
          } / 次`}
          color="text-purple-400"
          topBorder="border-t-purple-500"
          loading={loading && !totals}
        />
        <StatTile
          icon={Flame}
          title="缓存命中率"
          value={`${Math.round((cache?.cached_token_ratio ?? 0) * 100)}%`}
          sub={`请求命中 ${Math.round((cache?.hit_rate ?? 0) * 100)}%`}
          color="text-emerald-400"
          topBorder="border-t-emerald-500"
          loading={loading && !totals}
        />
        <StatTile
          icon={DollarSign}
          title="总成本合计"
          value={formatMoney(totals?.cost, currency)}
          sub={`${dateKeys.length} 个自然日`}
          color="text-amber-400"
          topBorder="border-t-amber-500"
          loading={loading && !totals}
        />
        <StatTile
          icon={Coins}
          title="请求均价"
          value={
            totals?.requests
              ? formatMoney((totals.cost ?? 0) / totals.requests, currency)
              : "—"
          }
          sub="单次请求均摊"
          color="text-rose-400"
          topBorder="border-t-rose-500"
          loading={loading && !totals}
        />
      </div>

      {/* Top Views: Trend / Models / Daily Tabs */}
      <Tabs defaultValue="trend" className="w-full">
        <TabsList className="bg-slate-950/60 p-1">
          <TabsTrigger value="trend">趋势图表</TabsTrigger>
          <TabsTrigger value="models">按模型拆分</TabsTrigger>
          <TabsTrigger value="raw">每日明细</TabsTrigger>
        </TabsList>

        {/* 1. Trend */}
        <TabsContent value="trend">
          <div className="grid grid-cols-1 xl:grid-cols-3 gap-5">
            <Card className="xl:col-span-2">
              <CardHeader>
                <div className="flex items-center justify-between">
                  <CardTitle className="text-base">每日用量与请求趋势</CardTitle>
                  <span className="text-xs font-mono text-slate-400">
                    {dateKeys.length} 天数据
                  </span>
                </div>
                <CardDescription>
                  多维堆叠展示每日输入、缓存与输出 Token 规模
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="h-72">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={chartData} barCategoryGap="20%">
                      <CartesianGrid
                        strokeDasharray="3 3"
                        stroke="#1e293b"
                        vertical={false}
                      />
                      <XAxis
                        dataKey="date"
                        stroke="#64748b"
                        fontSize={12}
                        tickLine={false}
                        axisLine={{ stroke: "#1e293b" }}
                        fontFamily="monospace"
                      />
                      <YAxis
                        stroke="#64748b"
                        fontSize={12}
                        tickLine={false}
                        axisLine={{ stroke: "#1e293b" }}
                        tickFormatter={(v) => formatShort(v)}
                        fontFamily="monospace"
                      />
                      <Tooltip
                        cursor={{ fill: "rgba(255, 255, 255, 0.03)" }}
                        contentStyle={{
                          background: "#0f172a",
                          border: "1px solid #334155",
                          borderRadius: 8,
                          color: "#f8fafc",
                          fontFamily: "monospace",
                          fontSize: "12px",
                        }}
                      />
                      <Legend wrapperStyle={{ paddingTop: "8px", fontSize: "12px" }} />
                      <Bar
                        dataKey="请求数"
                        stackId="a"
                        fill="#06b6d4"
                        radius={[3, 3, 0, 0]}
                      />
                      <Bar
                        dataKey="命中Tokens"
                        stackId="b"
                        fill="#10b981"
                      />
                      <Bar
                        dataKey="输入Tokens"
                        stackId="b"
                        fill="#3b82f6"
                      />
                      <Bar
                        dataKey="输出Tokens"
                        stackId="b"
                        fill="#8b5cf6"
                        radius={[3, 3, 0, 0]}
                      />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle className="text-base">模型成本占比</CardTitle>
                <CardDescription>按各模型产生的费用消耗比例分析</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="h-72 flex items-center justify-center">
                  {modelData.filter((m) => m.cost > 0).length === 0 ? (
                    <div className="text-slate-500 text-xs font-mono">
                      当前筛选范围暂无成本产生
                    </div>
                  ) : (
                    <ResponsiveContainer width="100%" height="100%">
                      <PieChart>
                        <Tooltip
                          contentStyle={{
                            background: "#0f172a",
                            border: "1px solid #334155",
                            borderRadius: 8,
                            color: "#f8fafc",
                            fontFamily: "monospace",
                            fontSize: "12px",
                          }}
                          formatter={(v: any, n: any) => [
                            formatMoney(Number(v), currency),
                            n,
                          ]}
                        />
                        <Pie
                          data={modelData.filter((m) => m.cost > 0)}
                          dataKey="cost"
                          nameKey="name"
                          innerRadius={55}
                          outerRadius={85}
                          paddingAngle={2}
                        >
                          {modelData.map((_, i) => (
                            <Cell
                              key={i}
                              fill={
                                [
                                  "#06b6d4",
                                  "#8b5cf6",
                                  "#10b981",
                                  "#f59e0b",
                                  "#f43f5e",
                                  "#3b82f6",
                                ][i % 6]
                              }
                            />
                          ))}
                        </Pie>
                        <Legend wrapperStyle={{ fontSize: "11px" }} />
                      </PieChart>
                    </ResponsiveContainer>
                  )}
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        {/* 2. Models Breakdown */}
        <TabsContent value="models">
          <Card>
            <CardHeader>
              <CardTitle className="text-base">模型用量汇总</CardTitle>
              <CardDescription>
                各模型路由对应的请求数、Token 规模、成本及缓存命中率
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="overflow-x-auto scrollbar-thin">
                <table className="w-full text-xs whitespace-nowrap font-mono">
                  <thead>
                    <tr className="text-left text-slate-400 border-b border-brand-borderSubtle bg-slate-950/40">
                      <th className="px-4 py-2.5 font-semibold">模型路由</th>
                      <th className="px-4 py-2.5 font-semibold text-right">请求数</th>
                      <th className="px-4 py-2.5 font-semibold text-right">输入 Tokens</th>
                      <th className="px-4 py-2.5 font-semibold text-right text-emerald-400">缓存命中</th>
                      <th className="px-4 py-2.5 font-semibold text-right">未命中</th>
                      <th className="px-4 py-2.5 font-semibold text-right text-purple-400">输出 Tokens</th>
                      <th className="px-4 py-2.5 font-semibold text-right text-amber-400">成本合计</th>
                      <th className="px-4 py-2.5 font-semibold text-right">命中率</th>
                    </tr>
                  </thead>
                  <tbody>
                    {modelData.length === 0 && (
                      <tr>
                        <td colSpan={8} className="px-4 py-8 text-center text-slate-500">
                          暂无数据
                        </td>
                      </tr>
                    )}
                    {modelData.map((m) => {
                      const hitR =
                        m.requests > 0
                          ? Math.min(
                              1,
                              (stats?.total.by_model?.[m.name]?.hit_requests ?? 0) /
                                m.requests
                            )
                          : 0;
                      return (
                        <tr
                          key={m.name}
                          className="clean-table-row border-b border-brand-borderSubtle/60"
                        >
                          <td className="px-4 py-3 font-bold text-cyan-300">
                            {m.name}
                          </td>
                          <td className="px-4 py-3 text-right tabular-nums text-slate-200">
                            {formatNumber(m.requests)}
                          </td>
                          <td className="px-4 py-3 text-right tabular-nums text-slate-300">
                            {formatShort(m.input)}
                          </td>
                          <td className="px-4 py-3 text-right tabular-nums text-emerald-400 font-semibold">
                            {formatShort(m.cache)}
                          </td>
                          <td className="px-4 py-3 text-right tabular-nums text-slate-400">
                            {formatShort(Math.max(m.input - m.cache, 0))}
                          </td>
                          <td className="px-4 py-3 text-right tabular-nums text-purple-400">
                            {formatShort(m.output)}
                          </td>
                          <td className="px-4 py-3 text-right tabular-nums text-amber-400 font-bold">
                            {formatMoney(m.cost, currency)}
                          </td>
                          <td className="px-4 py-3 text-right">
                            <div className="inline-flex items-center gap-2 justify-end">
                              <div className="h-1.5 w-14 rounded-full bg-slate-800 overflow-hidden">
                                <div
                                  className="h-full bg-cyan-400"
                                  style={{ width: `${hitR * 100}%` }}
                                />
                              </div>
                              <span className="tabular-nums w-8 text-right text-cyan-300 font-bold">
                                {Math.round(hitR * 100)}%
                              </span>
                            </div>
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        {/* 3. Daily Table */}
        <TabsContent value="raw">
          <Card>
            <CardHeader>
              <CardTitle className="text-base">每日聚合明细</CardTitle>
              <CardDescription>
                北京时间自然日维度的请求汇总
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="overflow-x-auto scrollbar-thin">
                <table className="w-full text-xs whitespace-nowrap font-mono">
                  <thead>
                    <tr className="text-left text-slate-400 border-b border-brand-borderSubtle bg-slate-950/40">
                      <th className="px-4 py-2.5 font-semibold">日期</th>
                      <th className="px-4 py-2.5 font-semibold text-right">请求数</th>
                      <th className="px-4 py-2.5 font-semibold text-right">输入 Tokens</th>
                      <th className="px-4 py-2.5 font-semibold text-right text-emerald-400">命中 Tokens</th>
                      <th className="px-4 py-2.5 font-semibold text-right text-purple-400">输出 Tokens</th>
                      <th className="px-4 py-2.5 font-semibold text-right text-amber-400">成本合计</th>
                      <th className="px-4 py-2.5 font-semibold text-right">数据源</th>
                    </tr>
                  </thead>
                  <tbody>
                    {dateKeys.length === 0 && (
                      <tr>
                        <td colSpan={7} className="px-4 py-8 text-center text-slate-500">
                          暂无数据
                        </td>
                      </tr>
                    )}
                    {dateKeys.map((d) => {
                      const b = stats?.per_day?.[d];
                      const t = b?.totals;
                      return (
                        <tr
                          key={d}
                          className="clean-table-row border-b border-brand-borderSubtle/60"
                        >
                          <td className="px-4 py-3 font-semibold text-slate-200">{d}</td>
                          <td className="px-4 py-3 text-right tabular-nums text-slate-300">
                            {formatNumber(t?.requests)}
                          </td>
                          <td className="px-4 py-3 text-right tabular-nums text-slate-300">
                            {formatShort(t?.input_tokens)}
                          </td>
                          <td className="px-4 py-3 text-right tabular-nums text-emerald-400 font-semibold">
                            {formatShort(t?.cache_read_tokens)}
                          </td>
                          <td className="px-4 py-3 text-right tabular-nums text-purple-400">
                            {formatShort(t?.output_tokens)}
                          </td>
                          <td className="px-4 py-3 text-right tabular-nums text-amber-400 font-bold">
                            {formatMoney(t?.cost, currency)}
                          </td>
                          <td className="px-4 py-3 text-right">
                            <span className="px-1.5 py-0.5 rounded text-[10px] font-mono border border-cyan-500/30 bg-cyan-500/10 text-cyan-300 font-medium">
                              {b?.source === "sqlite" ? "SQLite" : b?.source === "memory" ? "内存" : "JSONL"}
                            </span>
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>

      {/* Flat Recent Requests Section (Always Visible at Bottom) */}
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between flex-wrap gap-2">
            <div>
              <div className="flex items-center gap-2">
                <CardTitle className="text-base flex items-center gap-2">
                  <Clock className="w-4 h-4 text-cyan-400" />
                  <span>近期请求流水 // RECENT REQUESTS</span>
                </CardTitle>
                {recent.length > 0 && (
                  <span className="px-2 py-0.5 rounded-full bg-cyan-500/20 text-cyan-400 text-[11px] font-mono font-semibold border border-cyan-500/30">
                    {recent.length} 条记录
                  </span>
                )}
              </div>
              <CardDescription>
                当前查询时间范围内的实时调用记录（按时间倒序排列）
              </CardDescription>
            </div>
            <Button
              variant="secondary"
              size="sm"
              onClick={handleRefresh}
              disabled={loading}
              className="font-mono text-xs h-8"
            >
              <RefreshCw className={cn("w-3.5 h-3.5 text-cyan-400", loading && "animate-spin")} />
              <span>刷新流水</span>
            </Button>
          </div>
        </CardHeader>
        <CardContent>
          {recent.length === 0 ? (
            <div className="py-12 text-center text-slate-500 font-mono text-xs">
              <Server className="mx-auto w-8 h-8 text-slate-600 mb-2" />
              <div>当前时间范围内暂无请求记录</div>
            </div>
          ) : (
            <div className="overflow-x-auto scrollbar-thin">
              <table className="w-full text-xs whitespace-nowrap font-mono">
                <thead>
                  <tr className="text-left text-slate-400 border-b border-brand-borderSubtle bg-slate-950/40">
                    <th className="px-3.5 py-2.5 font-semibold">请求时间</th>
                    <th className="px-3.5 py-2.5 font-semibold">状态</th>
                    <th className="px-3.5 py-2.5 font-semibold">Provider / Model</th>
                    <th className="px-3.5 py-2.5 font-semibold text-right">输入</th>
                    <th className="px-3.5 py-2.5 font-semibold text-right text-purple-400">输出</th>
                    <th className="px-3.5 py-2.5 font-semibold text-right text-emerald-400">缓存命中</th>
                    <th className="px-3.5 py-2.5 font-semibold text-right text-amber-400">计费成本</th>
                    <th className="px-3.5 py-2.5 font-semibold text-right">耗时</th>
                    <th className="px-3.5 py-2.5 font-semibold">模式</th>
                  </tr>
                </thead>
                <tbody>
                  {recent.map((r) => (
                    <tr
                      key={r.id}
                      className="clean-table-row border-b border-brand-borderSubtle/60"
                    >
                      <td className="px-3.5 py-2.5 text-slate-400">
                        {r.ts?.replace("T", " ")}
                      </td>
                      <td className="px-3.5 py-2.5">
                        {r.success ? (
                          <span className="inline-flex items-center gap-1 text-emerald-400 font-semibold">
                            <CheckCircle2 className="w-3.5 h-3.5" />
                            成功
                          </span>
                        ) : (
                          <span className="inline-flex items-center gap-1 text-rose-400 font-semibold">
                            <XCircle className="w-3.5 h-3.5" />
                            失败
                          </span>
                        )}
                      </td>
                      <td className="px-3.5 py-2.5">
                        <div className="font-bold text-cyan-300">
                          {r.provider}/{r.model}
                        </div>
                        <div className="text-[10px] text-slate-500">
                          ← {r.anthropic_model}
                        </div>
                      </td>
                      <td className="px-3.5 py-2.5 text-right tabular-nums text-slate-300">
                        {formatShort(r.input_tokens)}
                      </td>
                      <td className="px-3.5 py-2.5 text-right tabular-nums text-purple-400 font-medium">
                        {formatShort(r.output_tokens)}
                      </td>
                      <td className="px-3.5 py-2.5 text-right tabular-nums">
                        {r.cache_read_tokens > 0 ? (
                          <span className="text-emerald-400 font-semibold">
                            {formatShort(r.cache_read_tokens)}
                          </span>
                        ) : (
                          <span className="text-slate-600">—</span>
                        )}
                      </td>
                      <td className="px-3.5 py-2.5 text-right tabular-nums text-amber-400 font-bold">
                        {r.cost != null
                          ? formatMoney(r.cost, r.currency || currency)
                          : "—"}
                      </td>
                      <td className="px-3.5 py-2.5 text-right tabular-nums">
                        {r.latency_ms != null ? (
                          <span
                            className={cn(
                              "px-1.5 py-0.5 rounded text-[11px] font-mono",
                              r.latency_ms < 600
                                ? "text-emerald-400 bg-emerald-500/10"
                                : r.latency_ms < 2000
                                ? "text-amber-400 bg-amber-500/10"
                                : "text-rose-400 bg-rose-500/10"
                            )}
                          >
                            {r.latency_ms.toFixed(0)}ms
                          </span>
                        ) : (
                          <span className="text-slate-600">—</span>
                        )}
                      </td>
                      <td className="px-3.5 py-2.5">
                        <span
                          className={cn(
                            "px-2 py-0.5 rounded text-[10px] font-mono font-medium border",
                            r.is_stream
                              ? "bg-amber-500/10 text-amber-300 border-amber-500/30"
                              : "bg-purple-500/10 text-purple-300 border-purple-500/30"
                          )}
                        >
                          {r.is_stream ? "STREAM" : "SYNC"}
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
