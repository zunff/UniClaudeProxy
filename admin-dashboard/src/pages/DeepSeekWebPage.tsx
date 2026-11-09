import { useCallback, useEffect, useState } from "react";
import { toast } from "sonner";
import { Check, ExternalLink, Globe, Plus, RefreshCw, Trash2 } from "lucide-react";
import { Button } from "@/components/ui/button";
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
  profile_dir: string;
  headless: boolean;
  browser_running: boolean;
  busy: boolean;
  logged_in: boolean | null;
  url: string | null;
}

interface DeepSeekWebStatus {
  active_account: string;
  headless: boolean;
  accounts: AccountStatus[];
}

async function call<T>(url: string, init?: RequestInit): Promise<T> {
  const res = await fetch(url, init);
  const text = await res.text();
  const data = text ? JSON.parse(text) : null;
  if (!res.ok) throw new Error(data?.error || `${res.status} ${res.statusText}`);
  return data as T;
}

function accountLabel(account: string): string {
  return account || "默认账号";
}

function describe(s: AccountStatus): { label: string; tone: string; hint: string } {
  if (s.logged_in === true)
    return { label: "已登录", tone: "text-emerald-400", hint: "可以通过 deepseek-web 路由使用网页端。" };
  if (s.logged_in === false)
    return { label: "未登录", tone: "text-amber-400", hint: "点击“打开登录窗口”,在弹出的浏览器里登录 DeepSeek。" };
  if (s.busy)
    return {
      label: "请求处理中",
      tone: "text-sky-400",
      hint: "当前有请求在使用该账号的浏览器,登录状态暂时无法读取。若窗口里出现验证码,请直接在窗口中完成。",
    };
  if (!s.browser_running)
    return {
      label: "浏览器未启动",
      tone: "text-muted-foreground",
      hint: "首次请求或点击“打开登录窗口”时会启动浏览器;profile 中保存过的登录态会被沿用。",
    };
  return { label: "状态未知", tone: "text-muted-foreground", hint: "浏览器当前不在 chat.deepseek.com。" };
}

export default function DeepSeekWebPage() {
  const [status, setStatus] = useState<DeepSeekWebStatus | null>(null);
  const [pending, setPending] = useState<string | null>(null);
  const [newAccount, setNewAccount] = useState("");

  const refresh = useCallback(async () => {
    try {
      setStatus(await call<DeepSeekWebStatus>("/api/deepseek-web/status"));
    } catch (e) {
      toast.error(`读取状态失败: ${(e as Error).message}`);
    }
  }, []);

  useEffect(() => {
    refresh();
    const t = setInterval(refresh, 3000);
    return () => clearInterval(t);
  }, [refresh]);

  const openLogin = async (account: string) => {
    setPending(`login:${account}`);
    try {
      await call<AccountStatus>("/api/deepseek-web/login", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ account }),
      });
      toast.success(`已打开 ${accountLabel(account)} 的浏览器窗口,请在其中登录 DeepSeek`);
      await refresh();
    } catch (e) {
      toast.error(`打开失败: ${(e as Error).message}`);
    } finally {
      setPending(null);
    }
  };

  const setActive = async (account: string) => {
    setPending(`active:${account}`);
    try {
      await call("/api/deepseek-web/active-account", {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ account }),
      });
      toast.success(`已切换到 ${accountLabel(account)}`);
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
      await call("/api/deepseek-web/accounts", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ account: name }),
      });
      setNewAccount("");
      toast.success(`已添加账号 ${name},首次使用前请打开登录窗口登录`);
      await refresh();
    } catch (e) {
      toast.error(`添加失败: ${(e as Error).message}`);
    } finally {
      setPending(null);
    }
  };

  const removeAccount = async (account: string) => {
    if (!window.confirm(`删除账号 ${accountLabel(account)}?浏览器会被关闭,profile 目录保留在磁盘上。`)) return;
    setPending(`delete:${account}`);
    try {
      await call(`/api/deepseek-web/accounts/${encodeURIComponent(account)}`, { method: "DELETE" });
      toast.success(`已删除账号 ${accountLabel(account)}`);
      await refresh();
    } catch (e) {
      toast.error(`删除失败: ${(e as Error).message}`);
    } finally {
      setPending(null);
    }
  };

  const accounts = status?.accounts ?? [];
  const active = status?.active_account ?? "";

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Globe className="h-5 w-5 text-sky-400" />
            DeepSeek 网页端
          </CardTitle>
          <CardDescription>
            deepseek-web 路由通过 proxy 自己启动的浏览器驱动 chat.deepseek.com。每个账号有独立的浏览器 profile,
            必须分别在那个窗口里登录;在你日常使用的浏览器里登录不会生效。只有“当前账号”会被请求使用。
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-3">
          {accounts.length === 0 && (
            <p className="text-sm text-muted-foreground">尚未配置账号,在下方添加一个。</p>
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
                  <span className="font-mono text-sm text-slate-100">{accountLabel(a.account)}</span>
                  <span className={cn("text-xs font-semibold", d.tone)}>{d.label}</span>
                  {isActive && (
                    <span className="rounded-md border border-sky-500/40 px-1.5 py-0.5 text-[10px] font-mono text-sky-300">
                      当前账号
                    </span>
                  )}
                  <div className="ml-auto flex flex-wrap gap-2">
                    <Button
                      size="sm"
                      variant="outline"
                      onClick={() => openLogin(a.account)}
                      disabled={pending === `login:${a.account}` || status?.headless}
                    >
                      <ExternalLink className="h-3.5 w-3.5" />
                      {pending === `login:${a.account}` ? "打开中…" : "打开登录窗口"}
                    </Button>
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
                <p className="mt-1 text-[11px] text-muted-foreground">Profile:{a.profile_dir}</p>
              </div>
            );
          })}

          <div className="flex items-end gap-2 pt-2">
            <div className="flex-1">
              <Label htmlFor="ds-account">新增账号 ID</Label>
              <Input
                id="ds-account"
                value={newAccount}
                placeholder="例如 alice / bob"
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

          {status?.headless && (
            <p className="text-sm text-amber-400">
              当前设置了 DEEPSEEK_WEB_HEADLESS=1,浏览器没有窗口,无法在此登录。取消该变量并重启 proxy。
            </p>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
