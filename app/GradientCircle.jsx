"use client";

import { useEffect } from "react";
import { useMap } from "@vis.gl/react-google-maps";

const EARTH_RADIUS_METERS = 6371008.8;

function destinationDueNorth(latitude, longitude, distanceMeters) {
  return {
    lat: latitude + (distanceMeters / EARTH_RADIUS_METERS) * (180 / Math.PI),
    lng: longitude,
  };
}

/**
 * Renders a borderless radial risk gradient centered at the supplied
 * coordinates. Radius is measured in meters, matching Google Maps circles.
 */
export function GradientCircle({ latitude, longitude, radius }) {
  const map = useMap();

  useEffect(() => {
    const centerLatitude = Number(latitude);
    const centerLongitude = Number(longitude);
    const radiusMeters = Number(radius);

    if (
      !map ||
      !window.google?.maps?.OverlayView ||
      !Number.isFinite(centerLatitude) ||
      !Number.isFinite(centerLongitude) ||
      !Number.isFinite(radiusMeters) ||
      radiusMeters <= 0
    ) {
      return undefined;
    }

    const overlay = new window.google.maps.OverlayView();
    const gradientId = `gradient-circle-${Math.random().toString(36).slice(2)}`;
    let container;
    let svg;
    let circle;

    overlay.onAdd = () => {
      container = document.createElement("div");
      container.style.position = "absolute";
      container.style.pointerEvents = "none";

      svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
      svg.setAttribute("aria-hidden", "true");
      svg.style.display = "block";
      svg.style.overflow = "visible";

      const defs = document.createElementNS("http://www.w3.org/2000/svg", "defs");
      const gradient = document.createElementNS(
        "http://www.w3.org/2000/svg",
        "radialGradient"
      );
      gradient.setAttribute("id", gradientId);
      [
        ["0%", "#ef0000", "1"],
        ["38%", "#ff4d00", "0.86"],
        ["72%", "#ffd000", "0.45"],
        ["100%", "#fff200", "0"],
      ].forEach(([offset, color, opacity]) => {
        const stop = document.createElementNS("http://www.w3.org/2000/svg", "stop");
        stop.setAttribute("offset", offset);
        stop.setAttribute("stop-color", color);
        stop.setAttribute("stop-opacity", opacity);
        gradient.appendChild(stop);
      });
      defs.appendChild(gradient);
      svg.appendChild(defs);

      circle = document.createElementNS("http://www.w3.org/2000/svg", "circle");
      circle.setAttribute("fill", `url(#${gradientId})`);
      circle.setAttribute("stroke", "none");
      svg.appendChild(circle);
      container.appendChild(svg);
      overlay.getPanes().overlayLayer.appendChild(container);
    };

    overlay.draw = () => {
      if (!container || !svg || !circle) return;

      const projection = overlay.getProjection();
      if (!projection) return;

      const center = projection.fromLatLngToDivPixel(
        new window.google.maps.LatLng(centerLatitude, centerLongitude)
      );
      const edge = projection.fromLatLngToDivPixel(
        new window.google.maps.LatLng(
          destinationDueNorth(centerLatitude, centerLongitude, radiusMeters)
        )
      );
      if (!center || !edge) return;

      const pixelRadius = Math.max(1, Math.abs(center.y - edge.y));
      const diameter = pixelRadius * 2;

      container.style.left = `${center.x - pixelRadius}px`;
      container.style.top = `${center.y - pixelRadius}px`;
      svg.setAttribute("width", String(diameter));
      svg.setAttribute("height", String(diameter));
      svg.setAttribute("viewBox", `0 0 ${diameter} ${diameter}`);
      circle.setAttribute("cx", String(pixelRadius));
      circle.setAttribute("cy", String(pixelRadius));
      circle.setAttribute("r", String(pixelRadius));
    };

    overlay.onRemove = () => {
      container?.remove();
      container = undefined;
      svg = undefined;
      circle = undefined;
    };

    overlay.setMap(map);

    return () => {
      overlay.setMap(null);
    };
  }, [latitude, longitude, map, radius]);

  return null;
}
export default GradientCircle;
