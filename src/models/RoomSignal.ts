import mongoose, { type Document, type Model } from "mongoose";

export type RoomSignalType = "offer" | "answer" | "ice" | "leave";

export interface IRoomSignal extends Document {
  roomId: string;
  fromClientId: string;
  toClientId: string;
  type: RoomSignalType;
  payload: Record<string, unknown>;
  expiresAt: Date;
  createdAt: Date;
}

const RoomSignalSchema = new mongoose.Schema<IRoomSignal>(
  {
    roomId: { type: String, required: true, maxlength: 64 },
    fromClientId: { type: String, required: true, maxlength: 64 },
    toClientId: { type: String, required: true, maxlength: 64 },
    type: { type: String, enum: ["offer", "answer", "ice", "leave"], required: true },
    payload: { type: mongoose.Schema.Types.Mixed, default: {} },
    expiresAt: {
      type: Date,
      required: true,
      default: () => new Date(Date.now() + 10 * 60 * 1000),
      expires: 0,
    },
  },
  { timestamps: { createdAt: true, updatedAt: false } },
);

RoomSignalSchema.index({ roomId: 1, toClientId: 1, _id: 1 });

export const RoomSignal =
  (mongoose.models.RoomSignal as Model<IRoomSignal> | undefined) ??
  mongoose.model<IRoomSignal>("RoomSignal", RoomSignalSchema);
