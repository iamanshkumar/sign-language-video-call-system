import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // The app is opened from a 192.168.x.x address during local testing.
  // Next otherwise blocks dev asset/HMR requests from a hostname other than localhost.
  allowedDevOrigins: ["192.168.*.*"],
};

export default nextConfig;
