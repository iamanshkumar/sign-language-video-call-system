import bcrypt from "bcryptjs";
import { NextResponse } from "next/server";
import { dbConnect } from "@/lib/dbConnect";
import { createSessionToken, sessionCookie } from "@/lib/auth";
import { User } from "@/models/User";

export async function POST(request: Request) {
  try {
    const body = await request.json();
    const email = typeof body.email === "string" ? body.email.trim().toLowerCase() : "";
    const password = typeof body.password === "string" ? body.password : "";
    if (!email || !password) return NextResponse.json({ error: "Enter your email and password." }, { status: 400 });

    await dbConnect();
    const user = await User.findOne({ email }).select("+password");
    if (!user || !(await bcrypt.compare(password, user.password))) {
      return NextResponse.json({ error: "Email or password is incorrect." }, { status: 401 });
    }

    const response = NextResponse.json({
      message: "Welcome back.",
      user: { id: user.id, name: user.name, email: user.email, role: user.role },
    });
    response.cookies.set(sessionCookie.name, createSessionToken(user.id), {
      httpOnly: true,
      secure: process.env.NODE_ENV === "production",
      sameSite: "lax",
      path: "/",
      maxAge: sessionCookie.maxAge,
    });
    return response;
  } catch (error) {
    if (error instanceof SyntaxError) return NextResponse.json({ error: "Invalid request body." }, { status: 400 });
    console.error("Login failed", error);
    if (error instanceof Error && (error.name === "MongooseServerSelectionError" || error.message.includes("ECONNREFUSED"))) {
      return NextResponse.json(
        { error: "The database is unavailable. Start MongoDB, then try signing in again." },
        { status: 503 },
      );
    }
    if (error instanceof Error && error.message.includes("SESSION_SECRET")) {
      return NextResponse.json(
        { error: "Session setup is missing. Add SESSION_SECRET to .env.local and restart the app." },
        { status: 503 },
      );
    }
    return NextResponse.json({ error: "Unable to sign in right now." }, { status: 500 });
  }
}
