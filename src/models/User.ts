import mongoose, { type Document, type Model } from "mongoose";

export type UserRole = "disabled" | "normal";

export interface IUser extends Document {
  name: string;
  email: string;
  password: string;
  role: UserRole;
  createdAt: Date;
}

const UserSchema = new mongoose.Schema<IUser>(
  {
    name: { type: String, required: true, trim: true, minlength: 2, maxlength: 80 },
    email: {
      type: String,
      required: true,
      unique: true,
      lowercase: true,
      trim: true,
      maxlength: 254,
    },
    password: { type: String, required: true, minlength: 8, select: false },
    role: { type: String, enum: ["disabled", "normal"], default: "normal", required: true },
  },
  { timestamps: { createdAt: true, updatedAt: false } },
);

export const User = (mongoose.models.User as Model<IUser> | undefined) ??
  mongoose.model<IUser>("User", UserSchema);
