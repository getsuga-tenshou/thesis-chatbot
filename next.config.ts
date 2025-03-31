/** @type {import('next').NextConfig} */
const nextConfig = {
  experimental: {
    serverActions: true,
  },
  env: {
    HUGGING_FACE_API_KEY: process.env.HUGGINGFACE_API_KEY,
  },
};

module.exports = nextConfig;
