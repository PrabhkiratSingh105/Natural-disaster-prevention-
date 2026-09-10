"use client";
import dynamic from "next/dynamic";

const DisasterMap = dynamic(() => import("./DisasterMap"), {
  ssr: false,
});

export default function Home() {
  return <DisasterMap />;
}
