import * as React from "react";
import { type VariantProps, cva } from "class-variance-authority";
import { Slot } from "@radix-ui/react-slot";
import { cn } from "@/lib/utils";

const buttonVariants = cva(
  "inline-flex items-center justify-center gap-2 whitespace-nowrap rounded-md text-sm font-medium transition-all focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-brand-cyan/60 disabled:pointer-events-none disabled:opacity-50",
  {
    variants: {
      variant: {
        default:
          "bg-brand-cyan/15 text-brand-cyan border border-brand-cyan/40 hover:bg-brand-cyan/25 hover:shadow-glow",
        primary:
          "bg-brand-cyan text-slate-950 hover:bg-cyan-300 shadow-[0_0_24px_-6px_rgba(34,211,238,0.5)]",
        secondary:
          "bg-brand-panel2 text-slate-100 border border-brand-borderSubtle hover:bg-brand-panel2/80",
        ghost: "text-slate-300 hover:bg-white/5 hover:text-white",
        destructive:
          "bg-rose-500/15 text-rose-300 border border-rose-500/40 hover:bg-rose-500/25",
        outline:
          "border border-brand-borderSubtle bg-transparent text-slate-200 hover:bg-white/5",
      },
      size: {
        sm: "h-8 px-3 text-xs",
        default: "h-9 px-4",
        lg: "h-11 px-6 text-base",
        icon: "h-9 w-9",
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
