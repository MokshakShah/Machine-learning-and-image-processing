import path from "node:path"
import { fileURLToPath } from "node:url"

const configDir = path.dirname(fileURLToPath(import.meta.url))

/** @type {import('next').NextConfig} */
const nextConfig = {
  typescript: {
    ignoreBuildErrors: true,
  },
  turbopack: {
    root: configDir,
  },
  images: {
    unoptimized: true,
  },
}

export default nextConfig
