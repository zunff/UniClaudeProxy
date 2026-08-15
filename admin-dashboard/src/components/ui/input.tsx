import * as React from "react";
import { cn } from "@/lib/utils";

export const Input = React.forwardRef<
  HTMLInputElement,
  React.InputHTMLAttributes<HTMLInputElement>
>(({ className, type = "text", ...props }, ref) => (
  <input
    ref={ref}
    type={type}
    className={cn(
      "flex h-9 w-full rounded-lg border border-brand-borderSubtle bg-slate-950 px-3 py-1 text-xs text-slate-100 placeholder:text-slate-500 shadow-inner transition-colors focus-visible:outline-none focus-visible:border-cyan-500 focus-visible:ring-1 focus-visible:ring-cyan-500/30 disabled:cursor-not-allowed disabled:opacity-50 font-mono",
      className,
    )}
    {...props}
  />
));
Input.displayName = "Input";

export const Label = React.forwardRef<
  HTMLLabelElement,
  React.LabelHTMLAttributes<HTMLLabelElement>
>(({ className, ...props }, ref) => (
  <label
    ref={ref}
    className={cn("text-xs font-mono font-medium text-slate-400 mb-1 block", className)}
    {...props}
  />
));
Label.displayName = "Label";
