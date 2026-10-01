import bcrypt from "bcryptjs";
import { NextResponse } from "next/server";
import { dbConnect } from "@/lib/dbConnect";
import { User } from "@/models/User";

export async function POST(request: Request) {
  try {
    const body = await request.json();
    const name = typeof body.name === "string" ? body.name.trim() : "";
    const email = typeof body.email === "string" ? body.email.trim().toLowerCase() : "";
    const password = typeof body.password === "string" ? body.password : "";

    if (name.length < 2 || name.length > 80 || !/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email)) {
      return NextResponse.json({ error: "Enter a valid name and email address." }, { status: 400 });
    }
    if (password.length < 8 || password.length > 72) {
      return NextResponse.json({ error: "Password must be between 8 and 72 characters." }, { status: 400 });
    }

    await dbConnect();
    const existingUser = await User.findOne({ email }).select("_id").lean();
    if (existingUser) return NextResponse.json({ error: "Email already registered." }, { status: 409 });

    const user = await User.create({ name, email, password: await bcrypt.hash(password, 12) });
    return NextResponse.json(
      { message: "Account created. You can now sign in.", user: { id: user.id, name: user.name, email: user.email, role: user.role } },
      { status: 201 },
    );
  } catch (error) {
    if (error instanceof SyntaxError) return NextResponse.json({ error: "Invalid request body." }, { status: 400 });
    if (error && typeof error === "object" && "code" in error && error.code === 11000) {
      return NextResponse.json({ error: "Email already registered." }, { status: 409 });
    }
    console.error("Registration failed", error);
    return NextResponse.json({ error: "Unable to create your account right now." }, { status: 500 });
  }
}
