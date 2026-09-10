import "./globals.css";

export const metadata = {
  title: "OneHack — Disaster Area Selector",
  description: "Google Maps disaster area selection interface",
};

export default function RootLayout({ children }) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body>{children}</body>
    </html>
  );
}
