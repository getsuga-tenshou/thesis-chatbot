/** @type {import('next').NextConfig} */
const nextConfig = {
  experimental: {
    serverActions: true,
  },
  env: {
    HUGGING_FACE_API_KEY: process.env.HUGGINGFACE_API_KEY,
    MONGODB_URI: process.env.MONGODB_URI,
  },
};

module.exports = nextConfig;
