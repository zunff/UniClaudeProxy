import { NavKey, useAdmin } from "@/store/admin";
import {
  Activity,
  Bot,
  Database,
  GitBranch,
  Layers,
  LineChart,
  RefreshCw,
  Settings,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

const items: { key: NavKey; label: string; icon: any; hint: string }[] = [
  { key: "stats", label: "使用统计", icon: LineChart, hint: "多维度用量 / 成本" },
  { key: "models", label: "模型配置", icon: Database, hint: "Provider / 模型列表" },
  { key: "mappings", label: "模型映射", icon: GitBranch, hint: "Claude 模型 → 后端路由" },
  { key: "prices", label: "价格表", icon: Layers, hint: "计费价目维护" },
];

export function AppShell({ children }: { children: React.ReactNode }) {
  const { nav, setNav, loading, fetchAll } = useAdmin();

  return (
    <div className="relative min-h-screen bg-brand-bg text-slate-100">
      {/* Background decorations */}
      <div className="pointer-events-none absolute inset-0 tech-grid opacity-30" style={{ animation: "none" }} />
      <div className="pointer-events-none absolute -top-32 -left-32 w-96 h-96 rounded-full bg-brand-cyan/15 blur-[120px]" />
      <div className="pointer-events-none absolute -bottom-40 -right-32 w-[480px] h-[480px] rounded-full bg-brand-violet/10 blur-[140px]" />

      <div className="relative z-10 flex min-h-screen">
        {/* Sidebar */}
        <aside className="w-72 shrink-0 border-r border-brand-borderSubtle/60 bg-brand-panel/80">
          <div className="h-16 px-5 flex items-center gap-3 border-b border-brand-borderSubtle/60">
            <div className="relative">
              <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-brand-cyan to-brand-violet flex items-center justify-center shadow-glowViolet">
                <Bot className="w-5 h-5 text-slate-950" />
              </div>
              <span className="absolute -bottom-0.5 -right-0.5 w-3 h-3 rounded-full bg-brand-green border-2 border-brand-panel animate-pulse-slow" />
            </div>
            <div>
              <div className="text-sm font-semibold text-white tracking-wide glow-text">
                UniClaudeProxy
              </div>
              <div className="text-xs tracking-[0.18em] text-brand-cyan/70">
                Admin Console
              </div>
            </div>
          </div>

          <nav className="p-3 space-y-1">
            {items.map((it) => {
              const Icon = it.icon;
              const active = nav === it.key;
              return (
                <button
                  key={it.key}
                  onClick={() => setNav(it.key)}
                  className={cn(
                    "w-full text-left flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm transition-all",
                    active
                      ? "bg-brand-cyan/10 text-white border border-brand-cyan/30 shadow-glow"
                      : "text-slate-300 hover:bg-white/5 hover:text-white border border-transparent",
                  )}
                >
                  <Icon className={cn("w-4 h-4", active && "text-brand-cyan")} />
                  <div className="flex-1">
                    <div className="font-medium">{it.label}</div>
                    <div className="text-xs text-slate-500 leading-snug">{it.hint}</div>
                  </div>
                  <span
                    className={cn(
                      "h-1.5 w-1.5 rounded-full",
                      active ? "bg-brand-cyan shadow-[0_0_10px_rgba(34,211,238,0.9)]" : "bg-transparent",
                    )}
                  />
                </button>
              );
            })}
          </nav>

          <div className="absolute bottom-5 left-3 right-3">
            <div className="px-3 py-2 text-xs text-slate-500 flex items-center gap-2 leading-snug">
              <Settings className="w-3.5 h-3.5 shrink-0" />
              <span>读写 config.json · 热重载配置</span>
            </div>
          </div>
        </aside>

        {/* Main */}
        <div className="flex-1 flex flex-col min-w-0">
          <header className="h-16 border-b border-brand-borderSubtle/60 bg-brand-panel/80 px-8 flex items-center justify-between">
            <div className="flex items-center gap-3">
              <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md border border-brand-borderSubtle bg-brand-panel2 text-xs text-brand-cyan whitespace-nowrap">
                <Activity className="w-3 h-3" />
                CONNECTED · 127.0.0.1:10388
              </span>
              <span className="text-sm text-slate-400">
                今日概览 · {new Date().toLocaleDateString("zh-CN")}
              </span>
            </div>
            <div className="flex items-center gap-2 text-xs text-slate-400">
              <span className="inline-flex items-center gap-1.5 px-2 py-1 rounded-md border border-brand-borderSubtle bg-brand-panel2">
                <span className="w-1.5 h-1.5 rounded-full bg-brand-green animate-pulse-slow" />
                Local Only Mode
              </span>
            </div>
          </header>

          <main className="flex-1 p-8 overflow-y-auto scrollbar-thin" style={{ scrollBehavior: "smooth", WebkitOverflowScrolling: "touch" }}>
            {children}
          </main>
        </div>
      </div>
    </div>
  );
}
