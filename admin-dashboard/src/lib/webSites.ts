import { useCallback, useEffect, useMemo, useState } from "react";

/** One configured web provider (drives the 网页端 page and provider filters). */
export interface WebSiteEntry {
  provider: string;
  provider_type: string;
  label: string;
  transport: "browser" | "http";
  home: string;
  /** Mounted model ids (the routes live only on the 网页端 page). */
  models?: string[];
  accounts: number;
}

/** A known site the dashboard can add in one click. */
export interface WebSiteTemplate {
  site: string;
  label: string;
  transport: "browser" | "http";
  home: string;
  configured: boolean;
  provider?: string;
  /** Routes the site would serve once added. */
  default_models?: Record<string, string>;
}

export interface WebSitesResponse {
  sites: WebSiteEntry[];
  templates: WebSiteTemplate[];
}

const EMPTY: WebSitesResponse = { sites: [], templates: [] };

/**
 * `provider_type` values the proxy serves through the web pipeline.
 * Used only as a fallback before `/api/web/sites` answers (or if it fails), so
 * a web provider never leaks into the Models page.
 */
export function looksLikeWebProviderType(type: string | undefined): boolean {
  if (!type) return false;
  return type === "web-http" || type.endsWith("-web");
}

/**
 * Every `provider/model` route served by a web site, configured or not.
 * These are kept permanently in the model-mapping dropdown.
 */
export function webRoutes(data: WebSitesResponse): string[] {
  const out: string[] = [];
  for (const s of data.sites) {
    for (const mid of s.models ?? []) out.push(`${s.provider}/${mid}`);
  }
  for (const t of data.templates ?? []) {
    const provider = t.provider ?? t.site;
    for (const mid of Object.keys(t.default_models ?? {})) {
      out.push(`${provider}/${mid}`);
    }
  }
  return Array.from(new Set(out));
}

export async function fetchWebSites(): Promise<WebSitesResponse> {
  const res = await fetch("/api/web/sites");
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
  const data = (await res.json()) as WebSitesResponse;
  return { sites: data.sites ?? [], templates: data.templates ?? [] };
}

export function useWebSites() {
  const [data, setData] = useState<WebSitesResponse>(EMPTY);

  const refresh = useCallback(async () => {
    try {
      setData(await fetchWebSites());
    } catch {
      setData(EMPTY); // the list is a convenience; callers fall back to provider_type
    }
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  const providerNames = useMemo(
    () => new Set(data.sites.map((s) => s.provider)),
    [data],
  );
  const isWebProvider = useCallback(
    (name: string, type?: string) =>
      providerNames.has(name) || looksLikeWebProviderType(type),
    [providerNames],
  );
  const routes = useMemo(() => webRoutes(data), [data]);

  return { ...data, refresh, isWebProvider, routes };
}
