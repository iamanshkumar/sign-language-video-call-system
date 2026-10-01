import mongoose from "mongoose";
import { NextResponse } from "next/server";
import { dbConnect } from "@/lib/dbConnect";
import { RoomPresence } from "@/models/RoomPresence";
import { RoomSignal, type RoomSignalType } from "@/models/RoomSignal";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

const ROOM_ID_PATTERN = /^[a-zA-Z0-9_-]{1,64}$/;
const CLIENT_ID_PATTERN = /^[a-f0-9-]{36}$/i;
const SIGNAL_TYPES: RoomSignalType[] = ["offer", "answer", "ice", "leave"];
const PRESENCE_TTL_MS = 30_000;

interface RouteContext {
  params: Promise<{ roomId: string }>;
}

function validIds(roomId: string, clientId: string) {
  return ROOM_ID_PATTERN.test(roomId) && CLIENT_ID_PATTERN.test(clientId);
}

function errorResponse(error: unknown) {
  console.error("WebRTC signaling request failed", error);
  return NextResponse.json(
    { error: "Room signaling is unavailable. Check that MongoDB is running and MONGODB_URI is configured." },
    { status: 503 },
  );
}

export async function GET(request: Request, { params }: RouteContext) {
  const { roomId } = await params;
  const url = new URL(request.url);
  const clientId = url.searchParams.get("clientId") ?? "";
  const after = url.searchParams.get("after");
  if (!validIds(roomId, clientId)) {
    return NextResponse.json({ error: "Invalid room or participant ID." }, { status: 400 });
  }
  if (after && !mongoose.isValidObjectId(after)) {
    return NextResponse.json({ error: "Invalid signaling cursor." }, { status: 400 });
  }

  try {
    await dbConnect();
    const now = new Date();
    await RoomPresence.findOneAndUpdate(
      { roomId, clientId },
      { $set: { expiresAt: new Date(now.getTime() + PRESENCE_TTL_MS) } },
      { upsert: true, setDefaultsOnInsert: true },
    );

    const [participants, messages] = await Promise.all([
      RoomPresence.find({ roomId, updatedAt: { $gte: new Date(now.getTime() - PRESENCE_TTL_MS) } })
        .select("clientId")
        .lean(),
      RoomSignal.find({
        roomId,
        toClientId: clientId,
        ...(after ? { _id: { $gt: new mongoose.Types.ObjectId(after) } } : {}),
      })
        .sort({ _id: 1 })
        .limit(100)
        .lean(),
    ]);

    return NextResponse.json({
      participants: participants.map((participant) => participant.clientId),
      messages: messages.map((message) => ({
        id: String(message._id),
        fromClientId: message.fromClientId,
        type: message.type,
        payload: message.payload,
      })),
    });
  } catch (error) {
    return errorResponse(error);
  }
}

export async function POST(request: Request, { params }: RouteContext) {
  const { roomId } = await params;
  try {
    const body = await request.json();
    const fromClientId = typeof body.fromClientId === "string" ? body.fromClientId : "";
    const toClientId = typeof body.toClientId === "string" ? body.toClientId : "";
    const type = body.type as RoomSignalType;
    const payload = body.payload && typeof body.payload === "object" ? body.payload : {};

    if (!validIds(roomId, fromClientId) || !CLIENT_ID_PATTERN.test(toClientId) || fromClientId === toClientId) {
      return NextResponse.json({ error: "Invalid room or participant ID." }, { status: 400 });
    }
    if (!SIGNAL_TYPES.includes(type)) {
      return NextResponse.json({ error: "Invalid WebRTC signal type." }, { status: 400 });
    }
    if (Buffer.byteLength(JSON.stringify(payload), "utf8") > 64_000) {
      return NextResponse.json({ error: "WebRTC signal is too large." }, { status: 413 });
    }

    await dbConnect();
    await RoomSignal.create({ roomId, fromClientId, toClientId, type, payload });
    return NextResponse.json({ delivered: true }, { status: 202 });
  } catch (error) {
    if (error instanceof SyntaxError) {
      return NextResponse.json({ error: "Invalid request body." }, { status: 400 });
    }
    return errorResponse(error);
  }
}

export async function DELETE(request: Request, { params }: RouteContext) {
  const { roomId } = await params;
  const clientId = new URL(request.url).searchParams.get("clientId") ?? "";
  if (!validIds(roomId, clientId)) {
    return NextResponse.json({ error: "Invalid room or participant ID." }, { status: 400 });
  }

  try {
    await dbConnect();
    const others = await RoomPresence.find({
      roomId,
      clientId: { $ne: clientId },
      updatedAt: { $gte: new Date(Date.now() - PRESENCE_TTL_MS) },
    })
      .select("clientId")
      .lean();
    if (others.length) {
      await RoomSignal.insertMany(
        others.map((other) => ({
          roomId,
          fromClientId: clientId,
          toClientId: other.clientId,
          type: "leave" as const,
          payload: {},
        })),
      );
    }
    await RoomPresence.deleteOne({ roomId, clientId });
    return NextResponse.json({ left: true });
  } catch (error) {
    return errorResponse(error);
  }
}
