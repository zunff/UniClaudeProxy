import { create } from "zustand";
import { toast } from "sonner";

// --- Types ---
export interface RawAppConfig {
  server?: { host?: string; port?: string | number; local_only?: boolean };
  proxy?: ProxySection;
  upstream?: unknown;
  models: Record<string, string | string[] | Record<string, number>>;
  providers: Record<string, ProviderConfig>;
  billing?: BillingSection;
  [k: string]: unknown;
}

export interface ProxySection {
  enabled?: boolean;
  url?: string;
}

export interface ProviderConfig {
  provider_type?: string;
  api_key?: string;
  base_url?: string;
  headers?: Record<string, string>;
  models: Record<string, Record<string, unknown>>;
  [k: string]: unknown;
}

export interface BillingSection {
  enabled?: boolean;
  log_file?: string;
  prices: Record<string, PriceTableEntry>;
}

export interface PriceTableEntry {
  currency?: string;
  model?: string;
  peak?: PriceTier;
  offpeak?: PriceTier;
  peak_hours?: number[][];
  // Flat-tier compatibility
  input?: number;
  input_cached?: number;
  output?: number;
}

export interface PriceTier {
  input: number;
  input_cached: number;
  output: number;
}

export interface PricesResponse {
  prices: Record<string, PriceTableEntry>;
  price_bindings: Record<string, string>; // route_key -> price_name
  fx_to_cny?: Record<string, number>;
  bound_routes: Record<string, string[]>; // price_name -> [route_keys]
  route_to_claude: Record<string, string[]>; // route_key -> [claude_models]
  all_routes: string[];
  models: Record<string, string | string[] | Record<string, number>>;
  providers: Record<string, ProviderConfig>;
}

export interface StatsTotals {
  requests: number;
  success: number;
  input_tokens: number;
  output_tokens: number;
  cache_read_tokens: number;
  cache_miss_tokens: number;
  cost: number;
  hit_requests: number;
}

export interface StatsBucket {
  totals: StatsTotals;
  cache: { hit_requests: number; hit_rate: number; cached_token_ratio: number };
  by_model: Record<string, StatsTotals>;
  source?: "memory" | "jsonl" | "sqlite";
}

export interface RecentRecord {
  id: number;
  ts: string;
  date: string;
  provider: string;
  model: string;
  anthropic_model: string;
  is_stream: boolean;
  success: boolean;
  input_tokens: number;
  output_tokens: number;
  cache_read_tokens: number;
  cache_creation_tokens: number;
  cache_miss_tokens: number;
  cost: number | null;
  currency: string | null;
  latency_ms: number | null;
}

export interface StatsResponse {
  range: string;
  date_keys: string[];
  total: StatsBucket;
  per_day: Record<string, StatsBucket>;
  recent: RecentRecord[];
}

// --- Store ---
export type NavKey = "stats" | "models" | "mappings" | "prices";

export type ModelRoute = string | string[] | Record<string, number>;

interface AdminState {
  nav: NavKey;
  setNav: (k: NavKey) => void;

  loading: boolean;
  config: RawAppConfig | null;
  pricesResp: PricesResponse | null;
  stats: StatsResponse | null;
  statsRange: "today" | "yesterday" | "7d" | "30d" | "custom";
  statsCustom: { start: string; end: string };

  fetchAll: () => Promise<void>;
  fetchPrices: () => Promise<void>;
  fetchStats: () => Promise<void>;

  upsertPrice: (name: string, entry: PriceTableEntry) => Promise<boolean>;
  deletePrice: (name: string) => Promise<boolean>;
  setBinding: (routeKey: string, priceName: string) => Promise<boolean>;
  deleteBinding: (routeKey: string) => Promise<boolean>;
  setFxToCny: (fx: Record<string, number>) => Promise<boolean>;

  setModelMapping: (claudeModel: string, routes: ModelRoute) => Promise<boolean>;
  deleteModelMapping: (claudeModel: string) => Promise<boolean>;

  saveConfig: (next: RawAppConfig) => Promise<boolean>;
  setStatsRange: (
    r: "today" | "yesterday" | "7d" | "30d" | "custom",
    custom?: { start: string; end: string },
  ) => void;
}

async function http<T>(
  input: RequestInfo,
  init?: RequestInit,
): Promise<T> {
  const res = await fetch(input, {
    ...init,
    headers: { "Content-Type": "application/json", ...(init?.headers ?? {}) },
  });
  const text = await res.text();
  let data: any = text ? JSON.parse(text) : null;
  if (!res.ok) {
    const msg = data?.error || data?.message || `${res.status} ${res.statusText}`;
    throw new Error(msg);
  }
  return data as T;
}

export const useAdmin = create<AdminState>((set, get) => ({
  nav: "stats",
  setNav: (k) => set({ nav: k }),

  loading: false,
  config: null,
  pricesResp: null,
  stats: null,
  statsRange: "7d",
  statsCustom: { start: "", end: "" },

  fetchAll: async () => {
    set({ loading: true });
    try {
      await Promise.all([get().fetchPrices(), get().fetchStats()]);
      const cfg = await http<RawAppConfig>("/api/config");
      set({ config: cfg });
    } catch (e: any) {
      toast.error("加载失败", {
        description: e?.message || "无法从 UniClaudeProxy 服务端拉取数据",
      });
    } finally {
      set({ loading: false });
    }
  },

  fetchPrices: async () => {
    try {
      const data = await http<PricesResponse>("/api/billing/prices");
      set({ pricesResp: data });
    } catch (e: any) {
      toast.error("价格表加载失败", { description: e?.message });
    }
  },

  fetchStats: async () => {
    const { statsRange, statsCustom } = get();
    const params = new URLSearchParams({ range: statsRange });
    if (statsRange === "custom" && statsCustom.start && statsCustom.end) {
      params.set("start_date", statsCustom.start);
      params.set("end_date", statsCustom.end);
    }
    set({ loading: true });
    try {
      const data = await http<StatsResponse>(`/api/stats?${params.toString()}`);
      set({ stats: data });
    } catch (e: any) {
      toast.error("统计加载失败", { description: e?.message });
    } finally {
      set({ loading: false });
    }
  },

  setStatsRange: (r, custom) => {
    set({
      statsRange: r,
      statsCustom: custom ?? get().statsCustom,
    });
    get().fetchStats();
  },

  upsertPrice: async (name, entry) => {
    try {
      await http<{ ok: boolean }>(`/api/billing/prices/${encodeURIComponent(name)}`, {
        method: "PUT",
        body: JSON.stringify(entry),
      });
      toast.success("价格表已保存", { description: name });
      await get().fetchPrices();
      const cfg = await http<RawAppConfig>("/api/config");
      set({ config: cfg });
      return true;
    } catch (e: any) {
      toast.error("保存价格表失败", { description: e?.message });
      return false;
    }
  },

  deletePrice: async (name) => {
    try {
      await http<{ ok: boolean }>(`/api/billing/prices/${encodeURIComponent(name)}`, {
        method: "DELETE",
      });
      toast.success("价格表已删除", { description: name });
      await get().fetchPrices();
      const cfg = await http<RawAppConfig>("/api/config");
      set({ config: cfg });
      return true;
    } catch (e: any) {
      toast.error("删除价格表失败", { description: e?.message });
      return false;
    }
  },

  setBinding: async (routeKey, priceName) => {
    try {
      await http<{ ok: boolean }>(`/api/billing/bindings/${encodeURIComponent(routeKey)}`, {
        method: "PUT",
        body: JSON.stringify({ price_name: priceName }),
      });
      toast.success("绑定已更新", {
        description: `${routeKey} → ${priceName}`,
      });
      await get().fetchPrices();
      return true;
    } catch (e: any) {
      toast.error("绑定失败", { description: e?.message });
      return false;
    }
  },

  deleteBinding: async (routeKey) => {
    try {
      await http<{ ok: boolean }>(`/api/billing/bindings/${encodeURIComponent(routeKey)}`, {
        method: "DELETE",
      });
      toast.success("已解绑", { description: routeKey });
      await get().fetchPrices();
      return true;
    } catch (e: any) {
      toast.error("解绑失败", { description: e?.message });
      return false;
    }
  },

  setFxToCny: async (fx) => {
    try {
      await http<{ ok: boolean }>("/api/billing/fx", {
        method: "PUT",
        body: JSON.stringify(fx),
      });
      toast.success("汇率已更新");
      await get().fetchPrices();
      return true;
    } catch (e: any) {
      toast.error("汇率保存失败", { description: e?.message });
      return false;
    }
  },

  setModelMapping: async (claudeModel, routes) => {
    try {
      await http<{ ok: boolean }>(`/api/models/${encodeURIComponent(claudeModel)}`, {
        method: "PUT",
        body: JSON.stringify({ routes }),
      });
      toast.success("模型映射已保存", { description: claudeModel });
      const cfg = await http<RawAppConfig>("/api/config");
      set({ config: cfg });
      await get().fetchPrices();
      return true;
    } catch (e: any) {
      toast.error("保存映射失败", { description: e?.message });
      return false;
    }
  },

  deleteModelMapping: async (claudeModel) => {
    try {
      await http<{ ok: boolean }>(`/api/models/${encodeURIComponent(claudeModel)}`, {
        method: "DELETE",
      });
      toast.success("模型映射已删除", { description: claudeModel });
      const cfg = await http<RawAppConfig>("/api/config");
      set({ config: cfg });
      await get().fetchPrices();
      return true;
    } catch (e: any) {
      toast.error("删除映射失败", { description: e?.message });
      return false;
    }
  },

  saveConfig: async (next) => {
    try {
      await http<{ ok: boolean }>("/api/config", {
        method: "PUT",
        body: JSON.stringify(next),
      });
      toast.success("配置已保存", { description: "运行时已热重载" });
      set({ config: next });
      await get().fetchPrices();
      return true;
    } catch (e: any) {
      toast.error("保存配置失败", { description: e?.message });
      return false;
    }
  },
}));
