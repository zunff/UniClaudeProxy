import { type ClassValue, clsx } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export function formatNumber(n: number | null | undefined): string {
  if (n == null || Number.isNaN(n)) return "0";
  return new Intl.NumberFormat("zh-CN").format(Math.round(n));
}

export function formatMoney(n: number | null | undefined, currency = "CNY"): string {
  if (n == null || Number.isNaN(n)) return "—";
  const abs = Math.abs(n);
  const digits = abs > 0 && abs < 0.01 ? 4 : 2;
  return new Intl.NumberFormat("zh-CN", {
    style: "currency",
    currency,
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  }).format(n);
}

export const DEFAULT_FX_TO_CNY: Record<string, number> = { USD: 7.2 };

export function toCny(
  amount: number,
  currency?: string | null,
  fxToCny?: Record<string, number> | null,
): number {
  const code = (currency || "CNY").toUpperCase();
  if (code === "CNY") return amount;
  const rate = fxToCny?.[code] ?? DEFAULT_FX_TO_CNY[code];
  if (rate == null) return amount;
  return amount * rate;
}

export function formatShort(n: number | null | undefined): string {
  if (n == null || Number.isNaN(n)) return "0";
  const abs = Math.abs(n);
  if (abs >= 1e9) return (n / 1e9).toFixed(2) + "B";
  if (abs >= 1e6) return (n / 1e6).toFixed(2) + "M";
  if (abs >= 1e3) return (n / 1e3).toFixed(2) + "K";
  return Math.round(n).toString();
}

export function todayStr(offsetDays = 0): string {
  const d = new Date();
  d.setDate(d.getDate() + offsetDays);
  return d.toISOString().slice(0, 10);
}
