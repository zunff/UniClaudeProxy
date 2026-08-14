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
  AlertTriangle,
  CheckCircle2,
  Clock,
  Coins,
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
  iconClass,
  title,
  value,
  sub,
  accent,
}: {
  icon: any;
  iconClass?: string;
  title: string;
  value: string;
  sub?: string;
  accent: "cyan" | "violet" | "green" | "amber" | "rose";
}) {
  const colors: Record<string, string> = {
    cyan: "from-brand-cyan/20 to-brand-cyan/0 border-brand-cyan/30 text-brand-cyan",
    violet:
      "from-brand-violet/20 to-brand-violet/0 border-brand-violet/30 text-brand-violet",
    green:
      "from-brand-green/20 to-brand-green/0 border-brand-green/30 text-brand-green",
    amber:
      "from-brand-amber/20 to-brand-amber/0 border-brand-amber/30 text-brand-amber",
    rose: "from-rose-500/20 to-rose-500/0 border-rose-500/30 text-rose-300",
  };
  return (
    <div className="tech-card p-5 min-w-0">
      <div
        className={cn(
          "inline-flex items-center justify-center h-10 w-10 rounded-lg border bg-gradient-to-br",
          colors[accent],
        )}
      >
        <Icon className={cn("w-5 h-5", iconClass)} />
      </div>
      <div className="mt-4 min-w-0">
        <div className="text-sm text-slate-400 whitespace-nowrap">{title}</div>
        <div className="mt-1 text-[28px] leading-none font-bold text-white tabular-nums tracking-tight whitespace-nowrap">
          {value}
        </div>
        {sub && (
          <div className="mt-2 text-sm text-slate-500 whitespace-nowrap">{sub}</div>
        )}
      </div>
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
    return lastUpdated.toLocaleTimeString("zh-CN", { hour: "2-digit", minute: "2-digit" });
  };

  const totals = stats?.total.totals;
  const cache = stats?.total.cache;
  const dateKeys = stats?.date_keys ?? [];
  const recent = stats?.recent ?? [];
  const currency = "CNY";

  const missingPriceRoutes = useMemo(() => {
    const byModel = stats?.total.by_model ?? {};
    const bindings = pricesResp?.price_bindings ?? {};
    // Also include direct price keys for backward compat
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
    <div className="space-y-6">
      <div className="flex items-center justify-between gap-4">
        <div className="min-w-0">
          <h1 className="text-2xl font-bold text-white tracking-wide glow-text">
            使用统计
          </h1>
          <p className="text-sm text-slate-400 mt-1 whitespace-nowrap">
            用量、缓存命中与成本 · 来源 <code className="text-brand-cyan">SQLite</code>
            {lastUpdated && (
              <span className="ml-2 text-slate-500">· 更新于 {formatLastUpdated()}</span>
            )}
          </p>
        </div>
        <div className="flex items-center gap-2 shrink-0">
          <button
            onClick={() => setAutoRefresh((v) => !v)}
            className={cn(
              "inline-flex items-center gap-1.5 px-2.5 py-1.5 rounded-md border text-xs transition-colors",
              autoRefresh
                ? "border-brand-cyan/30 bg-brand-cyan/10 text-brand-cyan"
                : "border-brand-borderSubtle bg-slate-900/40 text-slate-500 hover:text-slate-300",
            )}
            title={autoRefresh ? "自动刷新已开启" : "自动刷新已关闭"}
          >
            <span
              className={cn(
                "w-1.5 h-1.5 rounded-full",
                autoRefresh ? "bg-brand-cyan animate-pulse" : "bg-slate-600",
              )}
            />
            自动刷新
          </button>
          <Button
            variant="ghost"
            size="icon"
            onClick={handleRefresh}
            disabled={loading}
            title="刷新数据"
          >
            <RefreshCw className={cn("w-4 h-4", loading && "animate-spin")} />
          </Button>
        </div>
      </div>

      <div className="flex items-center gap-3 overflow-x-auto scrollbar-thin pb-1">
        <Tabs
          value={statsRange}
          onValueChange={(v) => setStatsRange(v as RangeKey, statsRange === "custom" ? { start, end } : undefined)}
        >
          <TabsList>
            {RANGES.map((r) => (
              <TabsTrigger key={r.key} value={r.key}>
                {r.label}
              </TabsTrigger>
            ))}
          </TabsList>
        </Tabs>
        {statsRange === "custom" && (
          <div className="flex items-center gap-2 shrink-0">
            <Input
              type="date"
              value={start}
              onChange={(e) => setStart(e.target.value)}
              className="w-[10.5rem]"
            />
            <span className="text-slate-500">→</span>
            <Input
              type="date"
              value={end}
              onChange={(e) => setEnd(e.target.value)}
              className="w-[10.5rem]"
            />
            <Button
              variant="primary"
              onClick={() =>
                setStatsRange("custom", { start, end })
              }
            >
              应用
            </Button>
          </div>
        )}
      </div>

      {missingPriceRoutes.length > 0 && (
        <div className="flex items-start gap-3 p-4 rounded-lg border border-brand-amber/30 bg-brand-amber/5">
          <AlertTriangle className="w-5 h-5 text-brand-amber shrink-0 mt-0.5" />
          <div className="text-sm text-slate-300">
            <span className="font-semibold text-brand-amber">
              缺少价格表，统计可能不准
            </span>
            <div className="mt-1 text-sm text-slate-400">
              以下模型路由没有对应的价格表，其成本不会被计入统计：
              <span className="text-rose-300 font-mono">
                {" "}
                {missingPriceRoutes.join(", ")}
              </span>
              。请前往「模型配置」页面为它们绑定价格表。
            </div>
          </div>
        </div>
      )}

      <div className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-3 gap-4">
        <StatTile
          icon={Server}
          accent="cyan"
          title="请求数"
          value={formatNumber(totals?.requests)}
          sub={`成功 ${formatNumber(totals?.success)}`}
        />
        <StatTile
          icon={Database}
          accent="violet"
          title="输入 Tokens"
          value={formatShort(totals?.input_tokens)}
          sub={`缓存命中 ${formatShort(totals?.cache_read_tokens)}`}
        />
        <StatTile
          icon={TrendingUp}
          accent="green"
          title="输出 Tokens"
          value={formatShort(totals?.output_tokens)}
        />
        <StatTile
          icon={Flame}
          accent="amber"
          title="缓存命中率"
          value={`${Math.round((cache?.cached_token_ratio ?? 0) * 100)}%`}
          sub={`请求命中 ${Math.round((cache?.hit_rate ?? 0) * 100)}%`}
        />
        <StatTile
          icon={DollarSign}
          accent="rose"
          title="成本合计"
          value={formatMoney(totals?.cost, currency)}
          sub={`日期：${dateKeys.length} 天`}
        />
        <StatTile
          icon={Coins}
          accent="cyan"
          title="每请求均价"
          value={
            totals?.requests
              ? formatMoney((totals.cost ?? 0) / totals.requests, currency)
              : "—"
          }
        />
      </div>

      <Tabs defaultValue="trend" className="w-full">
        <TabsList>
          <TabsTrigger value="trend">趋势（每日）</TabsTrigger>
          <TabsTrigger value="models">按模型拆分</TabsTrigger>
          <TabsTrigger value="raw">原始明细</TabsTrigger>
          <TabsTrigger value="recent">
            近期请求
            {recent.length > 0 && (
              <span className="ml-1.5 px-1.5 py-0.5 rounded-full bg-brand-cyan/20 text-brand-cyan text-[10px] font-medium">
                {recent.length}
              </span>
            )}
          </TabsTrigger>
        </TabsList>

        <TabsContent value="trend">
          <div className="grid grid-cols-1 xl:grid-cols-3 gap-5">
            <Card className="xl:col-span-2">
              <CardHeader>
                <CardTitle>每日用量与成本</CardTitle>
                <CardDescription>
                  范围：{stats?.range} · {dateKeys.length} 天
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="h-72">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={chartData} barCategoryGap="20%">
                      <CartesianGrid
                        strokeDasharray="3 3"
                        stroke="rgba(99,179,237,0.1)"
                      />
                      <XAxis
                        dataKey="date"
                        stroke="#64748b"
                        fontSize={13}
                        tickLine={false}
                        axisLine={false}
                      />
                      <YAxis
                        stroke="#64748b"
                        fontSize={13}
                        tickLine={false}
                        axisLine={false}
                      />
                      <Tooltip
                        cursor={{ fill: "rgba(99,179,237,0.06)" }}
                        contentStyle={{
                          background: "#0b1220",
                          border: "1px solid #1f2a44",
                          borderRadius: 10,
                          color: "#e2e8f0",
                        }}
                      />
                      <Legend />
                      <Bar dataKey="请求数" stackId="a" fill="#22d3ee" radius={[4, 4, 0, 0]} />
                      <Bar dataKey="命中Tokens" stackId="b" fill="#34d399" radius={[4, 4, 0, 0]} />
                      <Bar dataKey="输入Tokens" stackId="b" fill="#60a5fa" radius={[4, 4, 0, 0]} />
                      <Bar dataKey="输出Tokens" stackId="b" fill="#8b5cf6" radius={[4, 4, 0, 0]} />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>成本 Top 模型</CardTitle>
                <CardDescription>按成本占比</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="h-72">
                  {modelData.filter((m) => m.cost > 0).length === 0 ? (
                    <div className="h-full flex flex-col items-center justify-center text-slate-400 text-sm">
                      <Coins className="w-8 h-8 text-brand-cyan/60 mb-2" />
                      当前范围暂无成本数据
                    </div>
                  ) : (
                    <ResponsiveContainer width="100%" height="100%">
                      <PieChart>
                        <Tooltip
                          contentStyle={{
                            background: "#0b1220",
                            border: "1px solid #1f2a44",
                            borderRadius: 10,
                            color: "#e2e8f0",
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
                          outerRadius={90}
                          paddingAngle={2}
                        >
                          {modelData.map((_, i) => (
                            <Cell
                              key={i}
                              fill={
                                [
                                  "#22d3ee",
                                  "#8b5cf6",
                                  "#34d399",
                                  "#fbbf24",
                                  "#fb7185",
                                  "#60a5fa",
                                ][i % 6]
                              }
                            />
                          ))}
                        </Pie>
                        <Legend />
                      </PieChart>
                    </ResponsiveContainer>
                  )}
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        <TabsContent value="models">
          <Card>
            <CardHeader>
              <CardTitle>按模型汇总</CardTitle>
              <CardDescription>
                每一条 provider/model_id 的请求、token、成本、缓存统计。
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="overflow-x-auto scrollbar-thin">
                <table className="w-full text-[15px] whitespace-nowrap">
                  <thead>
                    <tr className="text-left text-slate-400 border-b border-brand-borderSubtle">
                      <th className="px-4 py-3 font-medium">模型路由</th>
                      <th className="px-4 py-3 font-medium text-right">请求</th>
                      <th className="px-4 py-3 font-medium text-right">输入</th>
                      <th className="px-4 py-3 font-medium text-right">缓存命中</th>
                      <th className="px-4 py-3 font-medium text-right">未命中</th>
                      <th className="px-4 py-3 font-medium text-right">输出</th>
                      <th className="px-4 py-3 font-medium text-right">成本</th>
                      <th className="px-4 py-3 font-medium text-right">命中率</th>
                    </tr>
                  </thead>
                  <tbody>
                    {modelData.length === 0 && (
                      <tr>
                        <td colSpan={8} className="px-4 py-10 text-center text-slate-400">
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
                                m.requests,
                            )
                          : 0;
                      return (
                        <tr
                          key={m.name}
                          className="border-b border-brand-borderSubtle/40 hover:bg-white/5"
                        >
                          <td className="px-4 py-3 font-mono text-brand-cyan">{m.name}</td>
                          <td className="px-4 py-3 text-right tabular-nums">{formatNumber(m.requests)}</td>
                          <td className="px-4 py-3 text-right tabular-nums">{formatShort(m.input)}</td>
                          <td className="px-4 py-3 text-right tabular-nums text-brand-green">
                            {formatShort(m.cache)}
                          </td>
                          <td className="px-4 py-3 text-right tabular-nums">
                            {formatShort(Math.max(m.input - m.cache, 0))}
                          </td>
                          <td className="px-4 py-3 text-right tabular-nums">{formatShort(m.output)}</td>
                          <td className="px-4 py-3 text-right tabular-nums text-brand-amber font-semibold">
                            {formatMoney(m.cost, currency)}
                          </td>
                          <td className="px-4 py-3 text-right">
                            <span className="inline-flex items-center gap-2">
                              <span className="h-1.5 w-16 rounded-full bg-slate-800 overflow-hidden">
                                <span
                                  className="block h-full bg-brand-cyan"
                                  style={{ width: `${hitR * 100}%` }}
                                />
                              </span>
                              <span className="tabular-nums w-10 text-right">{Math.round(hitR * 100)}%</span>
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

        <TabsContent value="raw">
          <Card>
            <CardHeader>
              <CardTitle>每日明细</CardTitle>
              <CardDescription>
                每行代表一个自然日（北京时间）的聚合结果。
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="overflow-x-auto scrollbar-thin">
                <table className="w-full text-[15px] whitespace-nowrap">
                  <thead>
                    <tr className="text-left text-slate-400 border-b border-brand-borderSubtle">
                      <th className="px-4 py-3 font-medium">日期</th>
                      <th className="px-4 py-3 font-medium text-right">请求</th>
                      <th className="px-4 py-3 font-medium text-right">输入</th>
                      <th className="px-4 py-3 font-medium text-right">命中</th>
                      <th className="px-4 py-3 font-medium text-right">输出</th>
                      <th className="px-4 py-3 font-medium text-right">成本</th>
                      <th className="px-4 py-3 font-medium text-right">数据源</th>
                    </tr>
                  </thead>
                  <tbody>
                    {dateKeys.length === 0 && (
                      <tr>
                        <td colSpan={7} className="px-4 py-10 text-center text-slate-400">
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
                          className="border-b border-brand-borderSubtle/40 hover:bg-white/5"
                        >
                          <td className="px-4 py-3 font-mono">{d}</td>
                          <td className="px-4 py-3 text-right tabular-nums">
                            {formatNumber(t?.requests)}
                          </td>
                          <td className="px-4 py-3 text-right tabular-nums">
                            {formatShort(t?.input_tokens)}
                          </td>
                          <td className="px-4 py-3 text-right tabular-nums text-brand-green">
                            {formatShort(t?.cache_read_tokens)}
                          </td>
                          <td className="px-4 py-3 text-right tabular-nums">
                            {formatShort(t?.output_tokens)}
                          </td>
                          <td className="px-4 py-3 text-right tabular-nums text-brand-amber font-semibold">
                            {formatMoney(t?.cost, currency)}
                          </td>
                          <td className="px-4 py-3 text-right">
                            <span
                              className={cn(
                                "px-2.5 py-1 rounded-md border text-sm",
                                b?.source === "sqlite"
                                  ? "border-brand-cyan/30 bg-brand-cyan/10 text-brand-cyan"
                                  : "border-brand-violet/30 bg-brand-violet/10 text-brand-violet",
                              )}
                            >
                              {b?.source === "sqlite" ? "SQLite" : b?.source === "memory" ? "内存聚合" : "JSONL 文件"}
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

        <TabsContent value="recent">
          <Card>
            <CardHeader>
              <div className="flex items-center justify-between">
                <div>
                  <CardTitle className="flex items-center gap-2">
                    <Clock className="w-4 h-4 text-brand-cyan" />
                    近期请求列表
                  </CardTitle>
                  <CardDescription>
                    最近 {recent.length} 条请求记录（当前查询范围）
                  </CardDescription>
                </div>
                <Button variant="ghost" size="icon" onClick={handleRefresh} disabled={loading}>
                  <RefreshCw className={cn("w-4 h-4", loading && "animate-spin")} />
                </Button>
              </div>
            </CardHeader>
            <CardContent>
              {recent.length === 0 ? (
                <div className="py-14 text-center text-slate-400">
                  <Server className="mx-auto w-10 h-10 text-brand-cyan/60 mb-3" />
                  <div>当前范围暂无请求记录</div>
                </div>
              ) : (
                <div className="overflow-x-auto scrollbar-thin">
                  <table className="w-full text-sm whitespace-nowrap">
                    <thead>
                      <tr className="text-left text-slate-400 border-b border-brand-borderSubtle">
                        <th className="px-3 py-2.5 font-medium">时间</th>
                        <th className="px-3 py-2.5 font-medium">状态</th>
                        <th className="px-3 py-2.5 font-medium">Provider / Model</th>
                        <th className="px-3 py-2.5 font-medium text-right">输入</th>
                        <th className="px-3 py-2.5 font-medium text-right">输出</th>
                        <th className="px-3 py-2.5 font-medium text-right">缓存</th>
                        <th className="px-3 py-2.5 font-medium text-right">成本</th>
                        <th className="px-3 py-2.5 font-medium text-right">延迟</th>
                        <th className="px-3 py-2.5 font-medium">类型</th>
                      </tr>
                    </thead>
                    <tbody>
                      {recent.map((r) => (
                        <tr
                          key={r.id}
                          className="border-b border-brand-borderSubtle/40 hover:bg-white/5"
                        >
                          <td className="px-3 py-2.5 text-slate-300 font-mono text-xs">
                            {r.ts?.replace("T", " ")}
                          </td>
                          <td className="px-3 py-2.5">
                            {r.success ? (
                              <span className="inline-flex items-center gap-1 text-brand-green">
                                <CheckCircle2 className="w-3.5 h-3.5" />
                                成功
                              </span>
                            ) : (
                              <span className="inline-flex items-center gap-1 text-rose-400">
                                <XCircle className="w-3.5 h-3.5" />
                                失败
                              </span>
                            )}
                          </td>
                          <td className="px-3 py-2.5">
                            <div className="font-mono text-brand-cyan text-xs">
                              {r.provider}/{r.model}
                            </div>
                            {r.is_stream && (
                              <Zap className="w-3 h-3 text-brand-amber inline mt-0.5" />
                            )}
                          </td>
                          <td className="px-3 py-2.5 text-right tabular-nums text-slate-300">
                            {formatShort(r.input_tokens)}
                          </td>
                          <td className="px-3 py-2.5 text-right tabular-nums text-slate-300">
                            {formatShort(r.output_tokens)}
                          </td>
                          <td className="px-3 py-2.5 text-right tabular-nums">
                            {r.cache_read_tokens > 0 ? (
                              <span className="text-brand-green">{formatShort(r.cache_read_tokens)}</span>
                            ) : (
                              <span className="text-slate-600">—</span>
                            )}
                          </td>
                          <td className="px-3 py-2.5 text-right tabular-nums text-brand-amber font-semibold">
                            {r.cost != null ? formatMoney(r.cost, r.currency || currency) : "—"}
                          </td>
                          <td className="px-3 py-2.5 text-right tabular-nums text-slate-400">
                            {r.latency_ms != null ? `${r.latency_ms.toFixed(0)}ms` : "—"}
                          </td>
                          <td className="px-3 py-2.5">
                            <span
                              className={cn(
                                "px-1.5 py-0.5 rounded text-[10px] font-medium",
                                r.is_stream
                                  ? "bg-brand-amber/10 text-brand-amber border border-brand-amber/20"
                                  : "bg-brand-violet/10 text-brand-violet border border-brand-violet/20",
                              )}
                            >
                              {r.is_stream ? "流式" : "非流式"}
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
        </TabsContent>
      </Tabs>
    </div>
  );
}
