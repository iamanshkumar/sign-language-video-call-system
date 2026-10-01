import type { HTMLAttributes } from "react";

export function Card({ className = "", ...props }: HTMLAttributes<HTMLDivElement>) {
  return <div className={`rounded-3xl border border-slate-200/80 bg-white shadow-[0_16px_55px_-28px_rgba(15,23,42,.24)] ${className}`} {...props} />;
}
