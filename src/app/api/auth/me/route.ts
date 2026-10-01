import { cookies } from "next/headers";
import { NextResponse } from "next/server";
import { dbConnect } from "@/lib/dbConnect";
import { sessionCookie, verifySessionToken } from "@/lib/auth";
import { User } from "@/models/User";

export async function GET() {
  const token = (await cookies()).get(sessionCookie.name)?.value;
  const userId = verifySessionToken(token);
  if (!userId) return NextResponse.json({ user: null }, { status: 200 });

  try {
    await dbConnect();
    const user = await User.findById(userId).select("name email role").lean();
    return NextResponse.json({ user: user ? { id: String(user._id), name: user.name, email: user.email, role: user.role } : null });
  } catch (error) {
    console.error("Session lookup failed", error);
    return NextResponse.json({ error: "Unable to load your session." }, { status: 500 });
  }
}
