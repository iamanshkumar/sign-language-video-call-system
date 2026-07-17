import mongoose, { mongo } from "mongoose";
import { unique } from "next/dist/build/utils";

export type UserRole = 'disabled' | 'normal';

export interface IUser extends mongoose.Document{
    name : string;
    email : string;
    password : string;
    role : UserRole;
    createdAt : Date
}

const UserSchema = new mongoose.Schema({
  name: {
    type: String,
    required: [true, "Name is required"],
    trim: true,
  },
  email: {
    type: String,
    requird: [true, "Email is required"],
    unique: true,
    lowercase: true,
    trim: true,
  },
  password: {
    type: String,
    required: [true, "Password is required"],
    minlength: [6, "Password must be at least 6 characters long"],
  },
  role: {
    type: String,
    enum: ["disabled", "normal"],
    required: [true, 'Role must be specified as either "disabled" or "normal"'],
  },
  createdAt: {
    type: Date,
    default: Date.now,
  },
});

export const User = mongoose.models.User || mongoose.model("User", UserSchema);