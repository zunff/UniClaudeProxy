import { useEffect, useMemo, useState } from "react";
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
  CalendarDays,
  Coins,
  Database,
  DollarSign,
  Flame,
  RefreshCw,
  Server,
  TrendingUp,
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
    <div className="tech-card p-5">
      <div
        className={cn(
          "inline-flex items-center justify-center h-10 w-10 rounded-lg border bg-gradient-to-br",
          colors[accent],
        )}
      >
        <Icon className={cn("w-5 h-5", iconClass)} />
      </div>
      <div className="mt-4">
        <div className="text-xs text-slate-400 uppercase tracking-wider">{title}</div>
        <div className="mt-1 text-2xl font-bold text-white">{value}</div>
        {sub && <div className="mt-1 text-[11px] text-slate-500">{sub}</div>}
      </div>
    </div>
  );
}

export default function StatsPage() {
  const { stats, statsRange, setStatsRange, fetchStats, loading, pricesResp } =
    useAdmin();
  const [start, setStart] = useState(todayStr(-6));
  const [end, setEnd] = useState(todayStr(0));

  useEffect(() => {
    fetchStats();
  }, []);

  const totals = stats?.total.totals;
  const cache = stats?.total.cache;
  const dateKeys = stats?.date_keys ?? [];
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
      <div className="flex items-end justify-between gap-4 flex-wrap">
        <div>
          <h1 className="text-2xl font-bold text-white tracking-wide glow-text">
            使用统计
          </h1>
          <p className="text-sm text-slate-400 mt-1">
            多维度查看用量、缓存命中与成本。数据来源：内存聚合 +
            <code className="text-brand-cyan"> logs/billing.jsonl</code>。
          </p>
        </div>
        <div className="flex items-center gap-3 flex-wrap">
          <Tabs
            value={statsRange}
            onValueChange={(v) => setStatsRange(v as RangeKey, statsRange === "custom" ? { start, end } : undefined)}
          >
            <TabsList>
              {RANGES.map((r) => (
                <TabsTrigger key={r.key} value={r.key}>
                  <CalendarDays className="w-3.5 h-3.5 mr-1.5" />
                  {r.label}
                </TabsTrigger>
              ))}
            </TabsList>
          </Tabs>
          {statsRange === "custom" && (
            <div className="flex items-center gap-2">
              <Input
                type="date"
                value={start}
                onChange={(e) => setStart(e.target.value)}
              />
              <span className="text-slate-500">→</span>
              <Input type="date" value={end} onChange={(e) => setEnd(e.target.value)} />
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
          <Button variant="ghost" size="icon" onClick={fetchStats} disabled={loading}>
            <RefreshCw className={cn("w-4 h-4", loading && "animate-spin")} />
          </Button>
        </div>
      </div>

      {missingPriceRoutes.length > 0 && (
        <div className="flex items-start gap-3 p-4 rounded-lg border border-brand-amber/30 bg-brand-amber/5">
          <AlertTriangle className="w-5 h-5 text-brand-amber shrink-0 mt-0.5" />
          <div className="text-sm text-slate-300">
            <span className="font-semibold text-brand-amber">
              缺少价格表，统计可能不准
            </span>
            <div className="mt-1 text-xs text-slate-400">
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

      <div className="grid grid-cols-2 md:grid-cols-3 xl:grid-cols-6 gap-4">
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
                        fontSize={11}
                        tickLine={false}
                        axisLine={false}
                      />
                      <YAxis
                        stroke="#64748b"
                        fontSize={11}
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
                <table className="w-full text-sm">
                  <thead>
                    <tr className="text-left text-slate-400 border-b border-brand-borderSubtle">
                      <th className="px-3 py-2 font-medium">模型路由</th>
                      <th className="px-3 py-2 font-medium text-right">请求</th>
                      <th className="px-3 py-2 font-medium text-right">输入</th>
                      <th className="px-3 py-2 font-medium text-right">缓存命中</th>
                      <th className="px-3 py-2 font-medium text-right">缓存未命中</th>
                      <th className="px-3 py-2 font-medium text-right">输出</th>
                      <th className="px-3 py-2 font-medium text-right">成本</th>
                      <th className="px-3 py-2 font-medium text-right">请求命中率</th>
                    </tr>
                  </thead>
                  <tbody>
                    {modelData.length === 0 && (
                      <tr>
                        <td colSpan={8} className="px-3 py-10 text-center text-slate-400">
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
                          <td className="px-3 py-2 font-mono text-brand-cyan">{m.name}</td>
                          <td className="px-3 py-2 text-right">{formatNumber(m.requests)}</td>
                          <td className="px-3 py-2 text-right">{formatShort(m.input)}</td>
                          <td className="px-3 py-2 text-right text-brand-green">
                            {formatShort(m.cache)}
                          </td>
                          <td className="px-3 py-2 text-right">
                            {formatShort(Math.max(m.input - m.cache, 0))}
                          </td>
                          <td className="px-3 py-2 text-right">{formatShort(m.output)}</td>
                          <td className="px-3 py-2 text-right text-brand-amber font-semibold">
                            {formatMoney(m.cost, currency)}
                          </td>
                          <td className="px-3 py-2 text-right">
                            <span className="inline-flex items-center gap-1.5">
                              <span className="h-1.5 w-16 rounded-full bg-slate-800 overflow-hidden">
                                <span
                                  className="block h-full bg-brand-cyan"
                                  style={{ width: `${hitR * 100}%` }}
                                />
                              </span>
                              {Math.round(hitR * 100)}%
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
                <table className="w-full text-sm">
                  <thead>
                    <tr className="text-left text-slate-400 border-b border-brand-borderSubtle">
                      <th className="px-3 py-2 font-medium">日期</th>
                      <th className="px-3 py-2 font-medium text-right">请求</th>
                      <th className="px-3 py-2 font-medium text-right">输入</th>
                      <th className="px-3 py-2 font-medium text-right">命中</th>
                      <th className="px-3 py-2 font-medium text-right">输出</th>
                      <th className="px-3 py-2 font-medium text-right">成本</th>
                      <th className="px-3 py-2 font-medium text-right">数据源</th>
                    </tr>
                  </thead>
                  <tbody>
                    {dateKeys.length === 0 && (
                      <tr>
                        <td colSpan={7} className="px-3 py-10 text-center text-slate-400">
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
                          <td className="px-3 py-2 font-mono">{d}</td>
                          <td className="px-3 py-2 text-right">
                            {formatNumber(t?.requests)}
                          </td>
                          <td className="px-3 py-2 text-right">
                            {formatShort(t?.input_tokens)}
                          </td>
                          <td className="px-3 py-2 text-right text-brand-green">
                            {formatShort(t?.cache_read_tokens)}
                          </td>
                          <td className="px-3 py-2 text-right">
                            {formatShort(t?.output_tokens)}
                          </td>
                          <td className="px-3 py-2 text-right text-brand-amber font-semibold">
                            {formatMoney(t?.cost, currency)}
                          </td>
                          <td className="px-3 py-2 text-right text-[11px]">
                            <span
                              className={cn(
                                "px-2 py-0.5 rounded-md border",
                                b?.source === "memory"
                                  ? "border-brand-cyan/30 bg-brand-cyan/10 text-brand-cyan"
                                  : "border-brand-violet/30 bg-brand-violet/10 text-brand-violet",
                              )}
                            >
                              {b?.source === "memory" ? "内存聚合" : "JSONL 文件"}
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
    </div>
  );
}
