import type { ButtonHTMLAttributes } from "react";

type ButtonVariant = "primary" | "secondary" | "outline" | "ghost" | "danger";

const variants: Record<ButtonVariant, string> = {
  primary: "bg-teal-700 text-white hover:bg-teal-800 focus-visible:ring-teal-600",
  secondary: "bg-teal-50 text-teal-800 hover:bg-teal-100 focus-visible:ring-teal-600",
  outline: "border border-slate-200 bg-white text-slate-700 hover:bg-slate-50 focus-visible:ring-teal-600",
  ghost: "text-slate-600 hover:bg-slate-100 focus-visible:ring-teal-600",
  danger: "bg-rose-600 text-white hover:bg-rose-700 focus-visible:ring-rose-500",
};

export function Button({
  className = "",
  variant = "primary",
  size = "default",
  ...props
}: ButtonHTMLAttributes<HTMLButtonElement> & { variant?: ButtonVariant; size?: "default" | "icon" }) {
  return (
    <button
      className={`inline-flex items-center justify-center gap-2 rounded-xl font-semibold transition-colors focus-visible:outline-none focus-visible:ring-4 focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50 ${size === "icon" ? "h-11 w-11" : "min-h-11 px-4 py-2.5"} ${variants[variant]} ${className}`}
      {...props}
    />
  );
}
