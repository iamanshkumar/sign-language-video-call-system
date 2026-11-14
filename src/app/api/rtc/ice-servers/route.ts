import { createHmac } from "node:crypto";
import { cookies } from "next/headers";
import { NextResponse } from "next/server";
import { sessionCookie, verifySessionToken } from "@/lib/auth";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

const defaultIceServers = [
  { urls: "stun:stun.l.google.com:19302" },
  { urls: "stun:stun1.l.google.com:19302" },
];

export async function GET() {
  const turnUrls = (process.env.TURN_URLS ?? "")
    .split(",")
    .map((url) => url.trim())
    .filter(Boolean);
  const turnSecret = process.env.TURN_SHARED_SECRET;

  if (!turnUrls.length) {
    return NextResponse.json({ iceServers: defaultIceServers });
  }
  if (!turnSecret) {
    return NextResponse.json(
      { error: "TURN_URLS is set, but TURN_SHARED_SECRET is missing." },
      { status: 503 },
    );
  }

  const configuredTtl = Number(process.env.TURN_CREDENTIAL_TTL_SECONDS ?? 3600);
  const ttlSeconds = Number.isFinite(configuredTtl)
    ? Math.min(Math.max(Math.floor(configuredTtl), 60), 86_400)
    : 3600;
  const sessionToken = (await cookies()).get(sessionCookie.name)?.value;
  const userId = verifySessionToken(sessionToken) ?? "guest";
  const username = `${Math.floor(Date.now() / 1000) + ttlSeconds}:${userId}`;
  const credential = createHmac("sha1", turnSecret).update(username).digest("base64");

  return NextResponse.json({
    iceServers: [
      ...defaultIceServers,
      { urls: turnUrls, username, credential },
    ],
  }, {
    headers: { "Cache-Control": "no-store, max-age=0" },
  });
}
