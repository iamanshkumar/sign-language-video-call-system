import type { LabelHTMLAttributes } from "react";

export function FieldLabel({ className = "", ...props }: LabelHTMLAttributes<HTMLLabelElement>) {
  return <label className={`mb-2 block text-sm font-semibold text-slate-700 ${className}`} {...props} />;
}
