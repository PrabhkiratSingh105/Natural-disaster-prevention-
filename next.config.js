/** @type {import('next').NextConfig} */
const nextConfig = {
  devIndicators: false,
  async rewrites() {
    return [
      {
        source: "/api/:path*",
        destination: `${process.env.FLASK_API_URL || "http://127.0.0.1:5000"}/api/:path*`,
      },
    ];
  },
};

module.exports = nextConfig;