import "./globals.css";

export const metadata = {
  title: "OneHack — Disaster Map",
  description: "Google Maps disaster planning interface",
};

export default function RootLayout({ children }) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body>{children}</body>
    </html>
  );
}
