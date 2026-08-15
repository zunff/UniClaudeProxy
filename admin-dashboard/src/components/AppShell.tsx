import { useState, useEffect } from "react";
import { NavKey, useAdmin } from "@/store/admin";
import {
  Activity,
  Bot,
  Database,
  GitBranch,
  Layers,
  LineChart,
  RefreshCw,
  Server,
  Shield,
  Zap,
} from "lucide-react";
import { cn } from "@/lib/utils";

const items: {
  key: NavKey;
  label: string;
  icon: any;
  hint: string;
  activeColor: string;
  activeBg: string;
  activeBorder: string;
}[] = [
  {
    key: "stats",
    label: "使用统计",
    icon: LineChart,
    hint: "用量监控与成本分析",
    activeColor: "text-cyan-400",
    activeBg: "bg-cyan-500/10",
    activeBorder: "border-cyan-500/40",
  },
  {
    key: "models",
    label: "模型配置",
    icon: Database,
    hint: "Provider 与上游参数",
    activeColor: "text-blue-400",
    activeBg: "bg-blue-500/10",
    activeBorder: "border-blue-500/40",
  },
  {
    key: "mappings",
    label: "模型映射",
    icon: GitBranch,
    hint: "Claude → 后端路由",
    activeColor: "text-purple-400",
    activeBg: "bg-purple-500/10",
    activeBorder: "border-purple-500/40",
  },
  {
    key: "prices",
    label: "价格表",
    icon: Layers,
    hint: "计费价目与汇率维护",
    activeColor: "text-emerald-400",
    activeBg: "bg-emerald-500/10",
    activeBorder: "border-emerald-500/40",
  },
];

export function AppShell({ children }: { children: React.ReactNode }) {
  const { nav, setNav, config, fetchAll, loading } = useAdmin();
  const host = config?.server?.host || "127.0.0.1";
  const port = config?.server?.port ?? 9223;
  const isLocalOnly = config?.server?.local_only ?? true;

  const [timeStr, setTimeStr] = useState("");

  useEffect(() => {
    const updateTime = () => {
      const now = new Date();
      setTimeStr(
        now.toLocaleDateString("zh-CN", {
          year: "numeric",
          month: "2-digit",
          day: "2-digit",
        }) +
          " " +
          now.toLocaleTimeString("zh-CN", { hour12: false })
      );
    };
    updateTime();
    const timer = setInterval(updateTime, 1000);
    return () => clearInterval(timer);
  }, []);

  const totalProviders = Object.keys(config?.providers ?? {}).length;
  const totalModels = Object.keys(config?.models ?? {}).length;

  return (
    <div className="min-h-screen bg-brand-bg text-slate-200 flex font-sans selection:bg-cyan-500/30 selection:text-cyan-200 w-full overflow-x-hidden">
      {/* Sidebar */}
      <aside className="w-64 shrink-0 border-r border-brand-borderSubtle bg-brand-panel flex flex-col justify-between select-none">
        <div>
          {/* Logo Header */}
          <div className="h-16 px-5 flex items-center gap-3 border-b border-brand-borderSubtle bg-slate-950/60">
            <div className="w-9 h-9 rounded-lg bg-gradient-to-br from-cyan-500/20 to-blue-600/20 border border-cyan-500/40 flex items-center justify-center text-cyan-400 shrink-0 shadow-sm">
              <Bot className="w-5 h-5" />
            </div>
            <div className="min-w-0 flex-1">
              <div className="text-sm font-bold text-white tracking-wide">
                UniClaudeProxy
              </div>
              <div className="text-[11px] text-cyan-400/80 font-mono font-medium tracking-wider">
                ADMIN CONSOLE
              </div>
            </div>
          </div>

          {/* Navigation */}
          <div className="p-3">
            <div className="px-3 py-1.5 text-[11px] font-mono text-slate-500 uppercase tracking-wider">
              系统功能
            </div>
            <nav className="space-y-1.5 mt-1">
              {items.map((it) => {
                const Icon = it.icon;
                const active = nav === it.key;
                return (
                  <button
                    key={it.key}
                    onClick={() => setNav(it.key)}
                    className={cn(
                      "w-full text-left flex items-center gap-3 px-3 py-2.5 rounded-xl text-sm transition-all duration-150 border",
                      active
                        ? cn(it.activeBg, it.activeBorder, "font-medium shadow-sm")
                        : "text-slate-400 hover:text-slate-100 hover:bg-slate-800/60 border-transparent"
                    )}
                  >
                    <Icon className={cn("w-4 h-4 shrink-0", active ? it.activeColor : "text-slate-400")} />
                    <div className="flex-1 min-w-0">
                      <div className={cn("text-xs font-semibold leading-none", active ? it.activeColor : "text-slate-200")}>
                        {it.label}
                      </div>
                      <div className="text-[11px] text-slate-400 truncate mt-1 leading-tight font-normal">
                        {it.hint}
                      </div>
                    </div>
                    {active && (
                      <span className={cn("w-1.5 h-1.5 rounded-full", it.activeColor === "text-cyan-400" ? "bg-cyan-400" : it.activeColor === "text-blue-400" ? "bg-blue-400" : it.activeColor === "text-purple-400" ? "bg-purple-400" : "bg-emerald-400")} />
                    )}
                  </button>
                );
              })}
            </nav>
          </div>
        </div>

        {/* Telemetry Footer */}
        <div className="p-3.5 m-3 rounded-xl border border-brand-borderSubtle bg-slate-950/70 text-xs font-mono">
          <div className="flex items-center justify-between text-slate-400 mb-2">
            <span className="flex items-center gap-1.5">
              <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
              <span className="text-slate-300 font-semibold">服务运行状态</span>
            </span>
            <span className="text-[10px] text-emerald-400 bg-emerald-500/10 px-1.5 py-0.5 rounded border border-emerald-500/30 font-bold">
              ACTIVE
            </span>
          </div>

          <div className="space-y-1.5 text-[11px]">
            <div className="flex justify-between">
              <span className="text-slate-400">端口:</span>
              <span className="text-cyan-300 font-semibold">{host}:{port}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-slate-400">Providers:</span>
              <span className="text-blue-300 font-semibold">{totalProviders} 组已就绪</span>
            </div>
            <div className="flex justify-between">
              <span className="text-slate-400">模型映射:</span>
              <span className="text-purple-300 font-semibold">{totalModels} 条路由</span>
            </div>
          </div>
        </div>
      </aside>

      {/* Main Content Area - Full Width */}
      <div className="flex-1 flex flex-col min-w-0 bg-brand-bg">
        {/* Top Header Bar */}
        <header className="h-16 border-b border-brand-borderSubtle bg-brand-panel/90 backdrop-blur-md px-8 flex items-center justify-between sticky top-0 z-20">
          <div className="flex items-center gap-3">
            <div className="flex items-center gap-2 px-2.5 py-1 rounded-lg border border-cyan-500/30 bg-cyan-500/10 text-xs font-mono text-cyan-400 font-semibold">
              <Activity className="w-3.5 h-3.5" />
              <span>已连接 · {host}:{port}</span>
            </div>

            {isLocalOnly && (
              <div className="flex items-center gap-1.5 px-2.5 py-1 rounded-lg border border-emerald-500/30 bg-emerald-500/10 text-xs font-mono text-emerald-400 font-semibold">
                <Shield className="w-3.5 h-3.5" />
                <span>仅限本地安全访问</span>
              </div>
            )}
          </div>

          <div className="flex items-center gap-3">
            <div className="hidden md:flex items-center gap-2 px-3 py-1.5 rounded-lg border border-brand-borderSubtle bg-slate-950/70 font-mono text-xs text-slate-300">
              <span className="w-2 h-2 rounded-full bg-cyan-400" />
              <span>{timeStr}</span>
            </div>

            <button
              onClick={() => fetchAll()}
              disabled={loading}
              className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg border border-cyan-500/30 bg-cyan-500/10 hover:bg-cyan-500/20 text-xs font-semibold text-cyan-300 transition-colors disabled:opacity-50"
              title="全局同步刷新"
            >
              <RefreshCw className={cn("w-3.5 h-3.5 text-cyan-400", loading && "animate-spin")} />
              <span>{loading ? "同步中..." : "全局刷新"}</span>
            </button>
          </div>
        </header>

        {/* Full-width Responsive Main Viewport */}
        <main className="flex-1 px-8 py-7 overflow-y-auto scrollbar-thin w-full max-w-full">
          {children}
        </main>
      </div>
    </div>
  );
}
