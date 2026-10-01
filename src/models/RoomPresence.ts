import mongoose, { type Document, type Model } from "mongoose";

export interface IRoomPresence extends Document {
  roomId: string;
  clientId: string;
  expiresAt: Date;
  createdAt: Date;
  updatedAt: Date;
}

const RoomPresenceSchema = new mongoose.Schema<IRoomPresence>(
  {
    roomId: { type: String, required: true, maxlength: 64 },
    clientId: { type: String, required: true, maxlength: 64 },
    expiresAt: { type: Date, required: true, expires: 0 },
  },
  { timestamps: true },
);

RoomPresenceSchema.index({ roomId: 1, clientId: 1 }, { unique: true });
RoomPresenceSchema.index({ roomId: 1, updatedAt: 1 });

export const RoomPresence =
  (mongoose.models.RoomPresence as Model<IRoomPresence> | undefined) ??
  mongoose.model<IRoomPresence>("RoomPresence", RoomPresenceSchema);
