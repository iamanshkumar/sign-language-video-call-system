"use client";

import { FormEvent, useState } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { FieldLabel } from "@/components/ui/field";
import { Input } from "@/components/ui/input";

type AuthMode = "login" | "register";

function Mark({ className = "" }: { className?: string }) {
  return (
    <svg viewBox="0 0 44 44" fill="none" className={className} aria-hidden="true">
      <rect width="44" height="44" rx="14" fill="currentColor" />
      <path d="M13 23.5 18.2 18l4.2 4.2L30.8 14M30.8 14v7m0-7h-7" stroke="white" strokeWidth="2.7" strokeLinecap="round" strokeLinejoin="round" />
      <circle cx="15" cy="30" r="2" fill="white" /><circle cx="29" cy="30" r="2" fill="white" />
      <path d="M17 30h10" stroke="white" strokeWidth="2" strokeLinecap="round" />
    </svg>
  );
}

export default function Home() {
  const router = useRouter();
  const [mode, setMode] = useState<AuthMode>("login");
  const [name, setName] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [room, setRoom] = useState("");
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const [loading, setLoading] = useState(false);

  async function handleAuth(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setError("");
    setNotice("");
    setLoading(true);
    try {
      const response = await fetch(`/api/auth/${mode === "login" ? "login" : "register"}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(mode === "login" ? { email, password } : { name, email, password }),
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || "Something went wrong.");
      if (mode === "register") {
        setNotice("Your account is ready. Sign in to start a call.");
        setMode("login");
        setPassword("");
      } else {
        setNotice(`Welcome, ${data.user.name.split(" ")[0]}. Pick a room to continue.`);
      }
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : "Something went wrong.");
    } finally {
      setLoading(false);
    }
  }

  function handleJoin(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const entry = room.trim();
    const linkMatch = entry.match(/^(?:https?:\/\/[^/]+)?\/?channel\/([a-zA-Z0-9_-]{1,64})\/?(?:[?#].*)?$/);
    const safeName = linkMatch?.[1] ?? entry;
    if (!/^[a-zA-Z0-9_-]{1,64}$/.test(safeName)) {
      setError("Enter a room ID or a valid Signcall room link.");
      return;
    }
    router.push(`/channel/${encodeURIComponent(safeName)}`);
  }

  function handleCreateRoom() {
    const roomId = `call-${crypto.randomUUID()}`;
    router.push(`/channel/${roomId}`);
  }

  return (
    <main className="min-h-screen overflow-hidden bg-[#f6f8f7] text-slate-900">
      <div className="absolute inset-x-0 top-0 h-[440px] bg-[radial-gradient(ellipse_at_top,_rgba(204,231,221,.55),_transparent_67%)]" />
      <div className="relative mx-auto flex min-h-screen max-w-[1320px] flex-col px-5 sm:px-8">
        <header className="flex h-20 items-center justify-between border-b border-slate-200/70">
          <Link href="/" className="flex items-center gap-3" aria-label="Signcall home">
            <Mark className="h-10 w-10 text-teal-700" />
            <span className="text-lg font-bold tracking-tight">signcall<span className="text-teal-700">.</span></span>
          </Link>
          <span className="hidden items-center gap-2 text-sm text-slate-500 sm:flex">
            <span className="h-2 w-2 rounded-full bg-emerald-500" /> Made for clearer conversations
          </span>
        </header>

        <section className="grid flex-1 items-center gap-12 py-12 lg:grid-cols-[1fr_460px] lg:gap-20 lg:py-16">
          <div className="max-w-2xl">
            <div className="mb-7 inline-flex items-center gap-2 rounded-full border border-teal-100 bg-white/80 px-3.5 py-2 text-xs font-semibold text-teal-800 shadow-sm">
              <span className="flex h-5 w-5 items-center justify-center rounded-full bg-teal-100 text-[11px]">✳</span>
              Communication without barriers
            </div>
            <h1 className="max-w-xl text-5xl font-semibold leading-[1.06] tracking-[-0.045em] sm:text-6xl lg:text-[68px]">
              Every gesture<br />brings us <span className="font-serif font-medium italic text-teal-700">closer.</span>
            </h1>
            <p className="mt-6 max-w-lg text-base leading-7 text-slate-600 sm:text-lg sm:leading-8">
              A calmer space to connect face to face. Start a room, invite someone in, and let every conversation feel more human.
            </p>
            <div className="mt-10 flex flex-wrap gap-x-7 gap-y-3 text-sm font-medium text-slate-600">
              <span className="flex items-center gap-2"><span className="text-teal-700">✓</span> One click to connect</span>
              <span className="flex items-center gap-2"><span className="text-teal-700">✓</span> Private rooms</span>
              <span className="flex items-center gap-2"><span className="text-teal-700">✓</span> Designed for everyone</span>
            </div>

            <div className="mt-14 hidden items-center gap-4 border-t border-slate-200/80 pt-7 sm:flex">
              <div className="flex -space-x-2">
                <div className="flex h-9 w-9 items-center justify-center rounded-full border-2 border-[#f6f8f7] bg-amber-100 text-xs font-bold text-amber-800">A</div>
                <div className="flex h-9 w-9 items-center justify-center rounded-full border-2 border-[#f6f8f7] bg-rose-100 text-xs font-bold text-rose-800">M</div>
                <div className="flex h-9 w-9 items-center justify-center rounded-full border-2 border-[#f6f8f7] bg-sky-100 text-xs font-bold text-sky-800">J</div>
              </div>
              <p className="text-xs leading-5 text-slate-500"><strong className="text-slate-700">Connection, made more accessible.</strong><br />A little more understanding goes a long way.</p>
            </div>
          </div>

          <Card className="relative p-6 sm:p-8">
            <div className="absolute -right-3 -top-3 flex h-12 w-12 items-center justify-center rounded-2xl bg-[#f2d9ba] text-xl shadow-sm" aria-hidden="true">✋</div>
            <div className="mb-6 flex rounded-xl bg-slate-100 p-1" role="tablist" aria-label="Account access">
              <button type="button" role="tab" aria-selected={mode === "login"} onClick={() => { setMode("login"); setError(""); setNotice(""); }} className={`min-h-10 flex-1 rounded-lg text-sm font-semibold transition ${mode === "login" ? "bg-white text-slate-900 shadow-sm" : "text-slate-500 hover:text-slate-800"}`}>Sign in</button>
              <button type="button" role="tab" aria-selected={mode === "register"} onClick={() => { setMode("register"); setError(""); setNotice(""); }} className={`min-h-10 flex-1 rounded-lg text-sm font-semibold transition ${mode === "register" ? "bg-white text-slate-900 shadow-sm" : "text-slate-500 hover:text-slate-800"}`}>Create account</button>
            </div>
            <h2 className="text-2xl font-semibold tracking-tight">{mode === "login" ? "Welcome back" : "Join the conversation"}</h2>
            <p className="mt-1 text-sm leading-6 text-slate-500">{mode === "login" ? "Sign in to start or join a video call." : "A few details and you’ll be ready to connect."}</p>

            <form onSubmit={handleAuth} className="mt-6 space-y-4">
              {mode === "register" && <div><FieldLabel htmlFor="name">Your name</FieldLabel><Input id="name" name="name" autoComplete="name" placeholder="Alex Morgan" value={name} onChange={(event) => setName(event.target.value)} required minLength={2} maxLength={80} /></div>}
              <div><FieldLabel htmlFor="email">Email address</FieldLabel><Input id="email" name="email" type="email" autoComplete="email" placeholder="you@example.com" value={email} onChange={(event) => setEmail(event.target.value)} required maxLength={254} /></div>
              <div><FieldLabel htmlFor="password">Password</FieldLabel><Input id="password" name="password" type="password" autoComplete={mode === "login" ? "current-password" : "new-password"} placeholder="At least 8 characters" value={password} onChange={(event) => setPassword(event.target.value)} required minLength={8} maxLength={72} /></div>
              {error && <p role="alert" className="rounded-xl bg-rose-50 px-3.5 py-3 text-sm text-rose-700">{error}</p>}
              {notice && <p role="status" className="rounded-xl bg-emerald-50 px-3.5 py-3 text-sm text-emerald-800">{notice}</p>}
              <Button className="w-full" type="submit" disabled={loading}>{loading ? "Please wait…" : mode === "login" ? "Sign in" : "Create account"}<span aria-hidden="true">→</span></Button>
            </form>

            <div className="my-6 flex items-center gap-3"><span className="h-px flex-1 bg-slate-200"/><span className="text-xs font-medium uppercase tracking-wider text-slate-400">or jump into a room</span><span className="h-px flex-1 bg-slate-200"/></div>
            <Button type="button" className="w-full" onClick={handleCreateRoom}>Create a room <span aria-hidden="true">✳</span></Button>
            <p className="mt-2 text-center text-xs text-slate-400">We’ll make a private room link for you to share.</p>
            <div className="my-5 flex items-center gap-3"><span className="h-px flex-1 bg-slate-200"/><span className="text-xs font-medium uppercase tracking-wider text-slate-400">or join an existing room</span><span className="h-px flex-1 bg-slate-200"/></div>
            <form onSubmit={handleJoin} className="space-y-3">
              <div><FieldLabel htmlFor="room">Room ID or invite link</FieldLabel><Input id="room" name="room" placeholder="Paste the room link or ID" value={room} onChange={(event) => { setRoom(event.target.value); if (error) setError(""); }} maxLength={512} /></div>
              <Button type="submit" variant="outline" className="w-full">Join a call <span className="text-teal-700" aria-hidden="true">↗</span></Button>
            </form>
            <p className="mt-5 text-center text-xs leading-5 text-slate-400">By continuing, you agree to use Signcall respectfully and keep conversations private.</p>
          </Card>
        </section>
        <footer className="flex flex-col gap-2 border-t border-slate-200/70 py-5 text-xs text-slate-400 sm:flex-row sm:items-center sm:justify-between"><span>© {new Date().getFullYear()} Signcall</span><span>Made for conversations that matter</span></footer>
      </div>
    </main>
  );
}
