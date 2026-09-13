"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import { APIProvider, Map, useMap } from "@vis.gl/react-google-maps";
import GradientCircle from "./GradientCircle";

const INDIA_CENTER = { lat: 22.5937, lng: 78.9629 };
const WEATHER_TILE_MAX_ZOOM = 18;
const CLOUD_OPACITY = 100;

function CloudTileLayer({ enabled, apiKey, onTileError }) {
  const map = useMap();

  useEffect(() => {
    if (!map || !enabled || !apiKey || !window.google?.maps?.Size) {
      return undefined;
    }

    const transparentTile = (ownerDocument) => {
      const tile = ownerDocument.createElement("div");
      tile.style.width = "256px";
      tile.style.height = "256px";
      tile.style.background = "transparent";
      return tile;
    };

    const tileLayer = {
      tileSize: new window.google.maps.Size(256, 256),
      minZoom: 0,
      maxZoom: WEATHER_TILE_MAX_ZOOM,
      name: "OpenWeather clouds",
      alt: "OpenWeather cloud cover",
      getTile: (coord, zoom, ownerDocument) => {
        const tilesPerAxis = 2 ** zoom;
        if (
          zoom < 0 ||
          zoom > WEATHER_TILE_MAX_ZOOM ||
          !coord ||
          coord.y < 0 ||
          coord.y >= tilesPerAxis
        ) {
          return transparentTile(ownerDocument);
        }

        const wrappedX = ((coord.x % tilesPerAxis) + tilesPerAxis) % tilesPerAxis;
        const tile = ownerDocument.createElement("img");
        tile.alt = "";
        tile.width = 256;
        tile.height = 256;
        tile.referrerPolicy = "no-referrer-when-downgrade";
        tile.style.opacity = String(CLOUD_OPACITY / 100);
        tile.onerror = () => {
          tile.style.display = "none";
          onTileError("provider");
        };
        tile.src = `https://tile.openweathermap.org/map/clouds_new/${zoom}/${wrappedX}/${coord.y}.png?appid=${encodeURIComponent(apiKey)}`;
        return tile;
      },
    };

    map.overlayMapTypes.insertAt(0, tileLayer);

    return () => {
      const overlays = map.overlayMapTypes;
      for (let index = overlays.getLength() - 1; index >= 0; index -= 1) {
        if (overlays.getAt(index) === tileLayer) {
          overlays.removeAt(index);
        }
      }
    };
  }, [apiKey, enabled, map, onTileError]);

  return null;
}

function PredictionView({ prediction }) {
  const map = useMap();

  useEffect(() => {
    if (!map || !prediction?.grid) return;
    map.fitBounds({
      north: prediction.grid.north_lat,
      south: prediction.grid.south_lat,
      east: prediction.grid.east_lon,
      west: prediction.grid.west_lon,
    }, 48);
  }, [map, prediction]);

  if (!prediction?.center || !prediction.radius_meters) return null;
  return (
    <GradientCircle
      latitude={prediction.center.lat}
      longitude={prediction.center.lon}
      radius={prediction.radius_meters}
    />
  );
}

export default function DisasterMap() {
  const apiKey = process.env.NEXT_PUBLIC_GOOGLE_MAPS_API_KEY;
  const configuredMapId = process.env.NEXT_PUBLIC_GOOGLE_MAPS_MAP_ID?.trim();
  const weatherTileApiKey = process.env.NEXT_PUBLIC_WEATHER_MAP_API_KEY;
  const [cloudsEnabled, setCloudsEnabled] = useState(false);
  const [cloudTileError, setCloudTileError] = useState("");
  const [cyclones, setCyclones] = useState([]);
  const [selectedCyclone, setSelectedCyclone] = useState("");
  const [prediction, setPrediction] = useState(null);
  const [predictionError, setPredictionError] = useState("");
  const [isLoadingPrediction, setIsLoadingPrediction] = useState(false);

  useEffect(() => {
    fetch("/api/cyclones")
      .then((response) => response.json())
      .then((items) => {
        setCyclones(items);
        if (items.length > 0) setSelectedCyclone(items[0].id);
      })
      .catch(() => setPredictionError("Could not load cyclone options."));
  }, []);

  async function selectCyclone(event) {
    const cycloneId = event.target.value;
    setSelectedCyclone(cycloneId);
    setPrediction(null);
    setPredictionError("");
    if (!cycloneId) return;

    const cyclone = cyclones.find((item) => item.id === cycloneId);
    if (!cyclone) return;

    setIsLoadingPrediction(true);
    try {
      const response = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ cyclone: cyclone.id, center: cyclone.center }),
      });
      const payload = await response.json().catch(() => ({}));
      if (!response.ok || payload.error) {
        throw new Error(payload.error || "Could not calculate cyclone damage.");
      }
      setPrediction(payload);
    } catch (error) {
      setPredictionError(error.message || "Could not calculate cyclone damage.");
    } finally {
      setIsLoadingPrediction(false);
    }
  }

  const mapOptions = useMemo(
    () => ({
      mapTypeControl: true,
      streetViewControl: true,
      fullscreenControl: true,
      zoomControl: true,
      clickableIcons: false,
    }),
    []
  );

  const handleCloudTileError = useCallback((reason) => {
    setCloudTileError(reason);
  }, []);

  if (!apiKey) {
    return (
      <main className="setup-screen">
        <div className="setup-card">
          <h1>Google Maps API key missing</h1>
          <p>Add your key to <code>.env.local</code>:</p>
          <pre>NEXT_PUBLIC_GOOGLE_MAPS_API_KEY=YOUR_KEY_HERE</pre>
          <p>Then restart the Next.js development server.</p>
        </div>
      </main>
    );
  }

  return (
    <APIProvider apiKey={apiKey}>
      <main className="app map-only-view">
        <section className="map-section">
          <div className="map-search-panel cyclone-picker">
            <div className="cyclone-picker-heading">
              <span className="cyclone-picker-mark" aria-hidden="true">✦</span>
              <div>
                <span className="cyclone-picker-kicker">Impact viewer</span>
                <strong>Select a cyclone</strong>
              </div>
            </div>
            <label className="sr-only" htmlFor="cyclone-select">Cyclone</label>
            <div className="cyclone-select-wrap">
              <select id="cyclone-select" value={selectedCyclone} onChange={selectCyclone}>
              <option value="">Select a cyclone</option>
              {cyclones.map((cyclone) => (
                <option key={cyclone.id} value={cyclone.id}>{cyclone.name}</option>
              ))}
              </select>
              <span className="cyclone-select-chevron" aria-hidden="true">⌄</span>
            </div>
            {isLoadingPrediction && <span className="cloud-help">Calculating damage heatmap...</span>}
            {predictionError && <span className="cloud-help cloud-error">{predictionError}</span>}
          </div>
          <div className="cloud-control">
            <button
              type="button"
              className={`cloud-toggle ${cloudsEnabled ? "active" : ""}`}
              aria-pressed={cloudsEnabled}
              onClick={() => {
                setCloudsEnabled((enabled) => !enabled);
                setCloudTileError("");
              }}
            >
              <span aria-hidden="true">☁️</span>
              <span>Clouds</span>
              <span className="cloud-state">{cloudsEnabled ? "ON" : "OFF"}</span>
            </button>
            {!weatherTileApiKey && (
              <span className="cloud-help cloud-error">
                Add NEXT_PUBLIC_WEATHER_MAP_API_KEY, then restart Next.js.
              </span>
            )}
            {cloudsEnabled && cloudTileError === "provider" && (
              <span className="cloud-help cloud-error">
                Cloud tiles were rejected. Check that the OpenWeather key is valid and enabled.
              </span>
            )}
          </div>

          <Map
            className="map"
            {...(configuredMapId ? { mapId: configuredMapId } : {})}
            defaultCenter={INDIA_CENTER}
            defaultZoom={5}
            gestureHandling="auto"
            {...mapOptions}
          >
            <CloudTileLayer
              enabled={cloudsEnabled}
              apiKey={weatherTileApiKey}
              onTileError={handleCloudTileError}
            />
            <PredictionView prediction={prediction} />
          </Map>
        </section>
      </main>
    </APIProvider>
  );
}
