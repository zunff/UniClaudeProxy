import * as React from "react";
import { type VariantProps, cva } from "class-variance-authority";
import { Slot } from "@radix-ui/react-slot";
import { cn } from "@/lib/utils";

const buttonVariants = cva(
  "inline-flex items-center justify-center gap-2 whitespace-nowrap rounded-lg text-sm font-medium transition-all duration-150 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-cyan-500/50 disabled:pointer-events-none disabled:opacity-50 active:scale-[0.98] select-none",
  {
    variants: {
      variant: {
        default:
          "bg-cyan-500/15 text-cyan-400 border border-cyan-500/40 hover:bg-cyan-500/25",
        primary:
          "bg-cyan-500 text-slate-950 font-semibold hover:bg-cyan-400 border border-cyan-400 shadow-sm",
        secondary:
          "bg-brand-panel2 text-slate-200 border border-brand-borderSubtle hover:bg-brand-panel3 hover:border-slate-600 hover:text-white",
        ghost:
          "text-slate-400 hover:bg-slate-800/60 hover:text-slate-100",
        destructive:
          "bg-rose-500/15 text-rose-300 border border-rose-500/40 hover:bg-rose-500/25",
        outline:
          "border border-brand-borderSubtle bg-slate-950/40 text-slate-200 hover:bg-slate-800/60",
      },
      size: {
        sm: "h-8 px-3 text-xs rounded-md",
        default: "h-9 px-4 text-xs",
        lg: "h-10 px-5 text-sm",
        icon: "h-8 w-8 p-0",
      },
    },
    defaultVariants: { variant: "default", size: "default" },
  },
);

export interface ButtonProps
  extends React.ButtonHTMLAttributes<HTMLButtonElement>,
    VariantProps<typeof buttonVariants> {
  asChild?: boolean;
}

export const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(
  ({ className, variant, size, asChild = false, ...props }, ref) => {
    const Cmp = asChild ? Slot : "button";
    return (
      <Cmp
        ref={ref}
        className={cn(buttonVariants({ variant, size }), className)}
        {...props}
      />
    );
  },
);
Button.displayName = "Button";

export { buttonVariants };
