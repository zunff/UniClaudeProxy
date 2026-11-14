import { useCallback, useEffect, useMemo, useState } from "react";
import { toast } from "sonner";
import {
  ChevronRight,
  Check,
  ExternalLink,
  Globe,
  KeyRound,
  Plus,
  RefreshCw,
  Route,
  Trash2,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { useAdmin } from "@/store/admin";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Input, Label } from "@/components/ui/input";
import { cn } from "@/lib/utils";

interface AccountStatus {
  account: string;
  label?: string;
  transport: "browser" | "http";
  profile_dir: string | null;
  headless: boolean;
  browser_running: boolean;
  busy: boolean;
  logged_in: boolean | null;
  url: string | null;
}

interface WebStatus {
  provider: string;
  provider_type: string;
  label: string;
  transport: "browser" | "http";
  home: string;
  active_account: string;
  active_account_label?: string;
  headless: boolean;
  token_hint: string | null;
  cookies_configured?: boolean;
  accounts: AccountStatus[];
}

interface SiteTemplate {
  site: string;
  label: string;
  transport: "browser" | "http";
  home: string;
  configured: boolean;
}

interface SiteEntry {
  provider: string;
  provider_type: string;
  label: string;
  transport: "browser" | "http" | "unknown";
  home: string;
  accounts: number;
  models?: string[];
  /** `provider_type` has no site implementation (removed site or a typo). */
  unregistered?: boolean;
}

async function call<T>(url: string, init?: RequestInit): Promise<T> {
  const res = await fetch(url, init);
  const text = await res.text();
  const data = text ? JSON.parse(text) : null;
  if (!res.ok) throw new Error(data?.error || data?.detail || `${res.status} ${res.statusText}`);
  return data as T;
}

function accountLabel(a: AccountStatus): string {
  return a.label ?? a.account ?? "默认账号";
}

function describe(a: AccountStatus): { label: string; tone: string; hint: string } {
  if (a.transport === "http") {
    if (a.busy)
      return { label: "请求处理中", tone: "text-sky-400", hint: "正在使用这个 token 调用网页接口。" };
    return {
      label: "Token 已配置",
      tone: "text-emerald-400",
      hint: "直接调用站点接口,不需要浏览器;token 失效时删除并重新添加。",
    };
  }
  if (a.logged_in === true)
    return { label: "已登录", tone: "text-emerald-400", hint: "可以通过对应的网页端路由使用。" };
  if (a.logged_in === false)
    return { label: "未登录", tone: "text-amber-400", hint: "点击“打开登录窗口”,在弹出的浏览器里登录。" };
  if (a.busy)
    return {
      label: "请求处理中",
      tone: "text-sky-400",
      hint: "当前有请求在使用该账号的浏览器,登录状态暂时无法读取。若窗口里出现验证码,请直接在窗口中完成。",
    };
  if (!a.browser_running)
    return {
      label: "浏览器未启动",
      tone: "text-muted-foreground",
      hint: "首次请求或点击“打开登录窗口”时会启动浏览器;profile 中保存过的登录态会被沿用。",
    };
  return { label: "状态未知", tone: "text-muted-foreground", hint: "浏览器当前不在站点首页。" };
}

export default function WebProvidersPage() {
  const { config, saveConfig, setNav } = useAdmin();
  const [sites, setSites] = useState<SiteEntry[]>([]);
  const [templates, setTemplates] = useState<SiteTemplate[]>([]);
  const [provider, setProvider] = useState<string>("");
  const [status, setStatus] = useState<WebStatus | null>(null);
  const [pending, setPending] = useState<string | null>(null);
  const [cookies, setCookies] = useState("");
  const [storage, setStorage] = useState("");
  const [probeOpen, setProbeOpen] = useState(false);
  const [probeCurl, setProbeCurl] = useState("");
  const [probeResult, setProbeResult] = useState<string>("");
  const [newAccount, setNewAccount] = useState("");

  const isHttp = status?.transport === "http";

  const currentSite = useMemo(
    () => sites.find((s) => s.provider === provider),
    [sites, provider],
  );
  const isUnregistered = Boolean(currentSite?.unregistered);

  const refreshSites = useCallback(async () => {
    try {
      const data = await call<{ sites: SiteEntry[]; templates?: SiteTemplate[] }>("/api/web/sites");
      setSites(data.sites);
      setTemplates(data.templates ?? []);
      setProvider((current) => current || data.sites[0]?.provider || "");
    } catch (e) {
      toast.error(`读取网页端列表失败: ${(e as Error).message}`);
    }
  }, []);

  const refresh = useCallback(async () => {
    if (!provider) return;
    // A block with no site implementation has no status endpoint: show the
    // warning card instead of hammering a 404 every 3s.
    if (!currentSite || currentSite.unregistered) {
      setStatus(null);
      return;
    }
    try {
      setStatus(await call<WebStatus>(`/api/web/${encodeURIComponent(provider)}/status`));
    } catch (e) {
      toast.error(`读取状态失败: ${(e as Error).message}`);
    }
  }, [provider, currentSite]);

  useEffect(() => {
    refreshSites();
  }, [refreshSites]);

  useEffect(() => {
    refresh();
    const t = setInterval(refresh, 3000);
    return () => clearInterval(t);
  }, [refresh]);

  const addSite = async (site: string) => {
    setPending(`add-site:${site}`);
    try {
      const res = await call<{ ok: boolean; provider: string }>("/api/web/add-site", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ site }),
      });
      toast.success(`已添加 ${res.provider}，接着登录一次即可`);
      await refreshSites();
      setProvider(res.provider);
    } catch (e) {
      toast.error(`添加失败: ${(e as Error).message}`);
    } finally {
      setPending(null);
    }
  };

  const saveCookies = async () => {
    setPending("cookies");
    try {
      const res = await call<{ ok: boolean; cookies: number }>(
        `/api/web/${encodeURIComponent(provider)}/cookies`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ cookies, storage }),
        },
      );
      toast.success(`已保存 ${res.cookies} 个 cookie`);
      setCookies("");
      setStorage("");
      await refresh();
    } catch (e) {
      toast.error(`保存失败: ${(e as Error).message}`);
    } finally {
      setPending(null);
    }
  };

  const runProbe = async () => {
    setPending("probe");
    setProbeResult("请求中…");
    try {
      const res = await call<any>("/api/web/probe", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ curl: probeCurl }),
      });
      if (!res.ok) {
        setProbeResult(`失败: ${res.error ?? "未知错误"}`);
      } else {
        setProbeResult(
          `${res.method} ${res.url}
HTTP ${res.status}  content-type: ${res.content_type}

${res.head}`,
        );
      }
    } catch (e) {
      setProbeResult(`失败: ${(e as Error).message}`);
    } finally {
      setPending(null);
    }
  };

  const openLogin = async (account: string) => {
    setPending(`login:${account}`);
    try {
      await call(`/api/web/${encodeURIComponent(provider)}/login`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ account }),
      });
      toast.success("已打开浏览器窗口,请在其中登录");
      await refresh();
    } catch (e) {
      toast.error(`打开失败: ${(e as Error).message}`);
    } finally {
      setPending(null);
    }
  };

  const captureToken = async () => {
    setPending("capture");
    toast.info("已打开浏览器窗口:登录后在页面里发一条消息,token 会自动保存(最长等 5 分钟)");
    try {
      const res = await call<{ ok: boolean; token: string }>(
        `/api/web/${encodeURIComponent(provider)}/capture-token`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ timeout: 300 }),
        },
      );
      toast.success(`已保存 token ${res.token} 到 config.json`);
      await refresh();
      await refreshSites();
    } catch (e) {
      toast.error(`抓取失败: ${(e as Error).message}`);
    } finally {
      setPending(null);
    }
  };

  const setActive = async (account: string) => {
    setPending(`active:${account}`);
    try {
      await call(`/api/web/${encodeURIComponent(provider)}/active-account`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ account }),
      });
      toast.success("已切换当前账号");
      await refresh();
    } catch (e) {
      toast.error(`切换失败: ${(e as Error).message}`);
    } finally {
      setPending(null);
    }
  };

  const addAccount = async () => {
    const name = newAccount.trim();
    if (!name) return;
    setPending("add");
    try {
      await call(`/api/web/${encodeURIComponent(provider)}/accounts`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ account: name }),
      });
      setNewAccount("");
      toast.success(isHttp ? "已添加 token" : "已添加账号,首次使用前请打开登录窗口登录");
      await refresh();
      await refreshSites();
    } catch (e) {
      toast.error(`添加失败: ${(e as Error).message}`);
    } finally {
      setPending(null);
    }
  };

  const removeAccount = async (account: string) => {
    const what = isHttp ? "token" : "账号";
    if (!window.confirm(`删除${what}?${isHttp ? "" : "浏览器会被关闭,profile 目录保留在磁盘上。"}`)) return;
    setPending(`delete:${account}`);
    try {
      await call(
        `/api/web/${encodeURIComponent(provider)}/accounts?account=${encodeURIComponent(account)}`,
        { method: "DELETE" },
      );
      toast.success(`已删除${what}`);
      await refresh();
      await refreshSites();
    } catch (e) {
      toast.error(`删除失败: ${(e as Error).message}`);
    } finally {
      setPending(null);
    }
  };

  const accounts = status?.accounts ?? [];
  const active = status?.active_account ?? "";
  const activeLabel = status?.active_account_label ?? active;

  // Mounted models are maintained here now (they were taken off the Models page).
  const siteModels = useMemo(
    () => sites.find((s) => s.provider === provider)?.models ?? [],
    [sites, provider],
  );

  const removeModel = async (modelId: string) => {
    if (!config) return;
    if (!window.confirm(`删除挂载模型 ${provider}/${modelId}？已经引用它的模型映射会失效。`)) return;
    setPending(`model:${modelId}`);
    try {
      const next = { ...config };
      next.providers = { ...(next.providers ?? {}) };
      const p = { ...next.providers[provider] };
      p.models = { ...(p.models ?? {}) };
      delete p.models[modelId];
      next.providers[provider] = p;
      await saveConfig(next);
      await refreshSites();
    } finally {
      setPending(null);
    }
  };

  const removeProvider = async () => {
    if (!config || !provider) return;
    if (
      !window.confirm(
        `删除 provider 块 ${provider}？它的挂载模型和引用它的模型映射都会失效。`,
      )
    )
      return;
    setPending("delete-provider");
    try {
      const next = { ...config };
      next.providers = { ...(next.providers ?? {}) };
      delete next.providers[provider];
      await saveConfig(next);
      setProvider("");
      setStatus(null);
      await refreshSites();
    } finally {
      setPending(null);
    }
  };

  const hint = useMemo(() => {
    if (!status) {
      return isUnregistered
        ? `config.json 里的 provider_type「${currentSite?.provider_type}」没有对应的网页端实现。`
        : "";
    }
    if (isHttp) {
      return `${status.label} 直接调用站点接口(config.json 里的 api_key / api_keys),不需要浏览器。token 获取方式:${status.token_hint ?? "见文档"}。`;
    }
    return `${status.provider_type} 路由通过 proxy 自己启动的浏览器驱动 ${status.home}。每个账号有独立的浏览器 profile,必须分别在那个窗口里登录;在你日常使用的浏览器里登录不会生效。只有“当前账号”会被请求使用。`;
  }, [status, isHttp, isUnregistered, currentSite]);

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-center gap-2">
        <Button size="sm" variant="outline" onClick={() => setProbeOpen((v) => !v)}>
          <Globe className="h-3.5 w-3.5" />
          {probeOpen ? "收起探测" : "探测站点（粘贴 cURL）"}
        </Button>
        {probeOpen && (
          <span className="text-xs text-muted-foreground">
            在目标站点按 F12 → Network → 右键任意一条请求 → Copy as cURL，粘贴到下面
          </span>
        )}
      </div>

      {probeOpen && (
        <div className="space-y-2 rounded-lg border border-brand-borderSubtle bg-slate-950/40 p-3">
          <textarea
            className="h-28 w-full rounded-md border border-brand-borderSubtle bg-slate-950 p-2 font-mono text-xs text-slate-200"
            placeholder="curl 'https://...' -H 'authorization: ...' --data-raw '{...}'"
            value={probeCurl}
            onChange={(e) => setProbeCurl(e.target.value)}
          />
          <div className="flex items-center gap-2">
            <Button size="sm" onClick={runProbe} disabled={!probeCurl.trim() || pending === "probe"}>
              {pending === "probe" ? "探测中…" : "发一次并看返回"}
            </Button>
            <span className="text-[11px] text-muted-foreground">
              会用你的 cookie 真发一次请求，只会读取前 4KB 返回
            </span>
          </div>
          {probeResult && (
            <pre className="max-h-64 overflow-auto rounded-md border border-brand-borderSubtle bg-black/40 p-2 text-[11px] leading-relaxed text-slate-300">
              {probeResult}
            </pre>
          )}
        </div>
      )}

      {templates.some((t) => !t.configured) && (
        <div className="flex flex-wrap items-center gap-2">
          <span className="text-xs font-mono text-slate-500">可添加的站点:</span>
          {templates
            .filter((t) => !t.configured)
            .map((t) => (
              <Button
                key={t.site}
                size="sm"
                variant="outline"
                disabled={pending === `add-site:${t.site}`}
                onClick={() => addSite(t.site)}
              >
                <Plus className="h-3.5 w-3.5" />
                {t.label}
              </Button>
            ))}
        </div>
      )}

      {sites.length > 1 && (
        <div className="flex flex-wrap gap-2">
          {sites.map((s) => (
            <Button
              key={s.provider}
              size="sm"
              variant={s.provider === provider ? "default" : "outline"}
              className={cn(s.unregistered && "border-amber-500/40 text-amber-300")}
              onClick={() => setProvider(s.provider)}
            >
              {s.transport === "http" ? (
                <KeyRound className="h-3.5 w-3.5" />
              ) : (
                <Globe className="h-3.5 w-3.5" />
              )}
              {s.label}
              {s.unregistered && <span className="text-[10px]">· 未注册</span>}
            </Button>
          ))}
        </div>
      )}

      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            {isHttp ? (
              <KeyRound className="h-5 w-5 text-emerald-400" />
            ) : (
              <Globe className={cn("h-5 w-5", isUnregistered ? "text-amber-400" : "text-sky-400")} />
            )}
            {status?.label ?? currentSite?.label ?? "网页端"}
            <span className="font-mono text-xs font-normal text-slate-500">
              {currentSite?.provider}
            </span>
          </CardTitle>
          <CardDescription>{hint}</CardDescription>
        </CardHeader>
        <CardContent className="space-y-3">
          {isUnregistered && (
            <div className="space-y-2 rounded-lg border border-amber-500/40 bg-amber-500/10 p-3">
              <p className="text-sm text-amber-300">
                provider_type「{currentSite?.provider_type}」没有对应的网页端实现，请求会直接报错。
              </p>
              <p className="text-xs text-amber-200/70">
                可能是站点已经被移除，或者 provider_type 写错了。改对 provider_type（deepseek-web /
                hailuo-web / web-http）后这个站点就能用；用不上的话直接删掉这个 provider 块。
              </p>
              <Button
                size="sm"
                variant="destructive"
                onClick={removeProvider}
                disabled={pending === "delete-provider"}
              >
                <Trash2 className="h-3.5 w-3.5" />
                删除该 Provider
              </Button>
            </div>
          )}

          {status && !isHttp && (
            <div className="space-y-2 rounded-lg border border-sky-500/30 bg-sky-500/5 p-3">
              <p className="text-sm text-slate-100">
                用 cookie 免登录{status?.cookies_configured ? "（已保存）" : ""}
              </p>
              <p className="text-xs text-muted-foreground">
                登录页有验证码时，在自己日常浏览器里登录该站点 → F12 → Network → 任意请求 → 复制 Request
                Headers 里的 cookie 整段粘到下面。保存后 proxy 打开页面时就是登录态，不再需要点登录。
              </p>
              <textarea
                className="h-20 w-full rounded-md border border-brand-borderSubtle bg-slate-950 p-2 font-mono text-[11px] text-slate-200"
                placeholder="refresh_token=...; cna=...; acw_tc=..."
                value={cookies}
                onChange={(e) => setCookies(e.target.value)}
              />
              <p className="text-xs text-muted-foreground">
                有的站点把 access token 存在 localStorage（只给 cookie 仍然显示未登录）。在已登录的那个浏览器里
                F12 → Console 执行 <code className="font-mono">copy(JSON.stringify(localStorage))</code>，粘到下框。
              </p>
              <textarea
                className="h-16 w-full rounded-md border border-brand-borderSubtle bg-slate-950 p-2 font-mono text-[11px] text-slate-200"
                placeholder='{"token":"eyJ...","theme":"dark"}'
                value={storage}
                onChange={(e) => setStorage(e.target.value)}
              />
              <Button
                size="sm"
                onClick={saveCookies}
                disabled={(!cookies.trim() && !storage.trim()) || pending === "cookies"}
              >
                {pending === "cookies" ? "保存中…" : "保存登录态"}
              </Button>
            </div>
          )}

          {isHttp && (
            <div className="flex flex-wrap items-center gap-3 rounded-lg border border-emerald-500/30 bg-emerald-500/5 p-3">
              <div className="min-w-0 flex-1">
                <p className="text-sm text-slate-100">自动抓取 token</p>
                <p className="text-xs text-muted-foreground">
                  点右侧按钮会打开一个浏览器窗口：登录该站点、在页面里发一条消息，proxy 会读取页面自己发出的
                  Authorization 头并写进 config.json。不用开 devtools、不用复制粘贴。
                </p>
              </div>
              <Button size="sm" onClick={captureToken} disabled={pending === "capture"}>
                <KeyRound className="h-3.5 w-3.5" />
                {pending === "capture" ? "等待登录…" : "登录并抓取 token"}
              </Button>
            </div>
          )}
          {currentSite && (
            <div className="space-y-2 rounded-lg border border-brand-borderSubtle bg-slate-950/40 p-3">
              <div className="flex flex-wrap items-center gap-2">
                <Route className="h-3.5 w-3.5 text-purple-400" />
                <span className="font-mono text-xs text-slate-300">
                  挂载模型 ({siteModels.length})
                </span>
                <span className="font-mono text-[11px] text-slate-500">
                  这些路由常驻在「模型映射」的下拉框里
                </span>
                <Button
                  variant="ghost"
                  size="sm"
                  className="ml-auto h-6 px-2 text-[11px] text-purple-300 hover:text-purple-200"
                  onClick={() => setNav("mappings")}
                >
                  去模型映射
                  <ChevronRight className="h-3 w-3" />
                </Button>
              </div>
              {siteModels.length === 0 ? (
                <p className="rounded border border-dashed border-brand-borderSubtle py-2 text-center font-mono text-xs text-slate-500">
                  该站点尚未挂载模型
                </p>
              ) : (
                <div className="flex flex-wrap gap-2">
                  {siteModels.map((modelId) => (
                    <span
                      key={modelId}
                      className="group flex items-center gap-1.5 rounded-md border border-brand-borderSubtle bg-slate-950/60 py-1 pl-2 pr-1 font-mono text-[11px] text-slate-200"
                    >
                      {provider}/{modelId}
                      <button
                        onClick={() => removeModel(modelId)}
                        disabled={pending === `model:${modelId}`}
                        className="text-slate-500 transition-colors hover:text-rose-400 disabled:opacity-50"
                        title="删除挂载模型"
                      >
                        <Trash2 className="h-3 w-3" />
                      </button>
                    </span>
                  ))}
                </div>
              )}
            </div>
          )}

          {sites.length === 0 && (
            <p className="text-sm text-muted-foreground">
              还没有配置网页端 provider。点上方「可添加的站点」一键添加 deepseek-web / hailuo-web,
              或在 config.json 里声明一个 provider_type 为 web-http 的站点。
            </p>
          )}

          {accounts.length === 0 && sites.length > 0 && (
            <p className="text-sm text-muted-foreground">
              尚未配置{isHttp ? " token" : "账号"},在下方添加一个。
            </p>
          )}

          {accounts.map((a) => {
            const d = describe(a);
            const isActive = a.account === active;
            return (
              <div
                key={a.account || "__default__"}
                className={cn(
                  "rounded-lg border p-3 transition-colors",
                  isActive
                    ? "border-sky-500/40 bg-sky-500/5"
                    : "border-brand-borderSubtle bg-slate-950/40",
                )}
              >
                <div className="flex flex-wrap items-center gap-2">
                  <span className="font-mono text-sm text-slate-100">{accountLabel(a)}</span>
                  <span className={cn("text-xs font-semibold", d.tone)}>{d.label}</span>
                  {isActive && (
                    <span className="rounded-md border border-sky-500/40 px-1.5 py-0.5 text-[10px] font-mono text-sky-300">
                      当前账号
                    </span>
                  )}
                  <div className="ml-auto flex flex-wrap gap-2">
                    {!isHttp && (
                      <Button
                        size="sm"
                        variant="outline"
                        onClick={() => openLogin(a.account)}
                        disabled={pending === `login:${a.account}` || status?.headless}
                      >
                        <ExternalLink className="h-3.5 w-3.5" />
                        {pending === `login:${a.account}` ? "打开中…" : "打开登录窗口"}
                      </Button>
                    )}
                    <Button
                      size="sm"
                      onClick={() => setActive(a.account)}
                      disabled={isActive || pending === `active:${a.account}`}
                    >
                      <Check className="h-3.5 w-3.5" />
                      {isActive ? "使用中" : "设为当前"}
                    </Button>
                    <Button
                      size="sm"
                      variant="destructive"
                      onClick={() => removeAccount(a.account)}
                      disabled={pending === `delete:${a.account}`}
                    >
                      <Trash2 className="h-3.5 w-3.5" />
                      删除
                    </Button>
                  </div>
                </div>
                {d.hint && <p className="mt-1 text-xs text-muted-foreground">{d.hint}</p>}
                {a.url && <p className="mt-1 truncate text-[11px] text-muted-foreground">{a.url}</p>}
                {a.profile_dir && (
                  <p className="mt-1 text-[11px] text-muted-foreground">Profile:{a.profile_dir}</p>
                )}
              </div>
            );
          })}

          {status && (
            <div className="flex items-end gap-2 pt-2">
              <div className="flex-1">
                <Label htmlFor="web-account">{isHttp ? "新增 token" : "新增账号 ID"}</Label>
                <Input
                  id="web-account"
                  value={newAccount}
                  placeholder={isHttp ? "粘贴站点请求里的 token / key" : "例如 alice / bob"}
                  onChange={(e) => setNewAccount(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === "Enter") addAccount();
                  }}
                />
              </div>
              <Button onClick={addAccount} disabled={!newAccount.trim() || pending === "add"}>
                <Plus className="h-4 w-4" />
                添加
              </Button>
              <Button variant="outline" onClick={refresh}>
                <RefreshCw className="h-4 w-4" />
                刷新
              </Button>
            </div>
          )}

          {status?.headless && !isHttp && (
            <p className="text-sm text-amber-400">
              当前该站点设置了 headless 模式,浏览器没有窗口,无法在此登录。取消对应的环境变量并重启 proxy。
            </p>
          )}
          {isHttp && activeLabel && (
            <p className="text-xs text-muted-foreground">当前使用的 token:{activeLabel}</p>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
