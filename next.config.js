/** @type {import('next').NextConfig} */
const nextConfig = {
  devIndicators: false,
  async rewrites() {
    return [
      {
        source: "/api/bounds",
        destination: "http://127.0.0.1:5000/api/bounds",
      },
    ];
  },
};

module.exports = nextConfig;