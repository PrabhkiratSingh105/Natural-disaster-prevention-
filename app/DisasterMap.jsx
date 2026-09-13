"use client";

import React, { useCallback, useMemo, useState, useEffect } from "react";
import {
  APIProvider,
  Map,
  Marker,
  Circle,
  Rectangle,
  useMapsLibrary,
  useMap,
} from "@vis.gl/react-google-maps";
import GradientCircle from "./GradientCircle";

const INDIA_CENTER = { lat: 22.5937, lng: 78.9629 };
const INDIA_BOUNDS = {
  north: 37.6,
  south: 6.2,
  east: 97.7,
  west: 68.0,
};
const WEATHER_TILE_MAX_ZOOM = 18;
// Change this constant in code to tune cloud transparency.
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

function toRad(value) {
  return (value * Math.PI) / 180;
}

function distanceMeters(a, b) {
  if (!a || !b) return 0;
  const earthRadius = 6371008.8;
  const lat1 = toRad(a.lat);
  const lat2 = toRad(b.lat);
  const dLat = toRad(b.lat - a.lat);
  const dLng = toRad(b.lng - a.lng);
  const haversine =
    Math.sin(dLat / 2) ** 2 +
    Math.cos(lat1) * Math.cos(lat2) * Math.sin(dLng / 2) ** 2;
  return 2 * earthRadius * Math.atan2(Math.sqrt(haversine), Math.sqrt(1 - haversine));
}

function areaFromRadius(radiusMeters) {
  return radiusMeters > 0 ? Math.PI * radiusMeters ** 2 : 0;
}

function areaFromViewport(viewport) {
  if (!viewport) return 0;

  const northSouth = distanceMeters(
    { lat: viewport.north, lng: viewport.west },
    { lat: viewport.south, lng: viewport.west }
  );
  const eastWest = distanceMeters(
    { lat: viewport.north, lng: viewport.west },
    { lat: viewport.north, lng: viewport.east }
  );
  return northSouth * eastWest;
}

function normalizeBounds(first, second) {
  if (!first || !second) return null;
  return {
    north: Math.max(first.lat, second.lat),
    south: Math.min(first.lat, second.lat),
    east: Math.max(first.lng, second.lng),
    west: Math.min(first.lng, second.lng),
  };
}

function areaFromBounds(bounds) {
  if (!bounds) return 0;
  return areaFromViewport(bounds);
}

function centerFromBounds(bounds) {
  if (!bounds) return null;
  return {
    lat: (bounds.north + bounds.south) / 2,
    lng: (bounds.east + bounds.west) / 2,
  };
}

function formatArea(areaSquareMeters) {
  return {
    km: areaSquareMeters > 0 ? (areaSquareMeters / 1_000_000).toFixed(3) : "",
    meters: areaSquareMeters > 0 ? areaSquareMeters.toFixed(2) : "",
  };
}

function getAreaSquareMeters(area) {
  if (!area) return 0;
  if (area.areaSquareMeters != null) return area.areaSquareMeters;
  if (area.bounds) return areaFromBounds(area.bounds);
  if (area.viewport) return areaFromViewport(area.viewport);
  return areaFromRadius(area.radiusMeters);
}

function findAreaAtDrop(projection, areas, dropLatLng, dropPixel) {
  let bestId = null;
  let bestScore = Infinity;

  areas.forEach((area) => {
    const dist = distanceMeters(area.center, dropLatLng);
    const containsCircle = dist <= (area.radiusMeters || 0);
    const containsRectangle =
      area.bounds &&
      dropLatLng.lat >= area.bounds.south &&
      dropLatLng.lat <= area.bounds.north &&
      dropLatLng.lng >= area.bounds.west &&
      dropLatLng.lng <= area.bounds.east;
    const pinPixel = projection.fromLatLngToContainerPixel(
      new google.maps.LatLng(area.center.lat, area.center.lng)
    );
    const pixelDist = pinPixel
      ? Math.hypot(dropPixel.x - pinPixel.x, dropPixel.y - pinPixel.y)
      : Infinity;
    const onPin = pixelDist <= 28;

    if (!containsCircle && !containsRectangle && !onPin) return;

    const score = containsRectangle ? 0 : containsCircle ? area.radiusMeters : dist;
    if (score < bestScore) {
      bestScore = score;
      bestId = area.id;
    }
  });

  return bestId;
}

function todayString() {
  const now = new Date();
  return formatLocalDate(now);
}

function todayTimeString() {
  const now = new Date();
  return formatLocalTime(now);
}

function formatLocalDate(date) {
  const year = date.getFullYear();
  const month = String(date.getMonth() + 1).padStart(2, "0");
  const day = String(date.getDate()).padStart(2, "0");
  return `${year}-${month}-${day}`;
}

function formatLocalTime(date) {
  const hours = String(date.getHours()).padStart(2, "0");
  const minutes = String(date.getMinutes()).padStart(2, "0");
  return `${hours}:${minutes}`;
}

function PlaceSearch({ onPlaceSelect }) {
  const places = useMapsLibrary("places");
  const map = useMap();
  const [query, setQuery] = useState("");
  const [predictions, setPredictions] = useState([]);
  const [isSearching, setIsSearching] = useState(false);
  const [selectedPrediction, setSelectedPrediction] = useState(null);
  const [highlightedPredictionIndex, setHighlightedPredictionIndex] = useState(-1);
  const [isSubmitting, setIsSubmitting] = useState(false);

  useEffect(() => {
    if (!places || query.trim().length < 2) {
      setPredictions([]);
      return undefined;
    }

    // Don't fetch if the user just selected this exact text from the dropdown
    if (
      selectedPrediction &&
      (query === selectedPrediction.text?.toString() ||
       query === selectedPrediction.mainText?.toString())
    ) {
      return undefined;
    }

    const timeoutId = window.setTimeout(() => {
      setIsSearching(true);
      places.AutocompleteSuggestion.fetchAutocompleteSuggestions({
        input: query.trim(),
        includedRegionCodes: ["in"],
      })
        .then(({ suggestions }) => {
          setPredictions(
            (suggestions || [])
              .map((suggestion) => suggestion.placePrediction)
              .filter(Boolean)
          );
          setHighlightedPredictionIndex(-1);
        })
        .catch(() => {
          setPredictions([]);
          setHighlightedPredictionIndex(-1);
        })
        .finally(() => setIsSearching(false));
    }, 250);

    return () => window.clearTimeout(timeoutId);
  }, [places, query, selectedPrediction]);

  function choosePrediction(prediction) {
    const displayName = prediction.text?.toString() || prediction.mainText?.toString() || "";
    setQuery(displayName);
    setSelectedPrediction(prediction);
    setPredictions([]);
    setHighlightedPredictionIndex(-1);
  }

  async function handleSearch(prediction = selectedPrediction, displayName = query) {
    if (!prediction || !places || !map) return;

    const placeId = prediction.placeId;
    const resourceName = prediction.resourceName || (placeId ? `places/${placeId}` : null);
    const resolvedPlaceId = placeId || resourceName?.split("/").pop();
    if (!placeId && !resourceName) {
      console.warn("Selected place prediction does not include a Google place identifier.");
      return;
    }

    setIsSubmitting(true);
    try {
      // Use the new Place class to fetch location details
      const place = new places.Place(
        placeId ? { id: placeId } : { resourceName }
      );
      await place.fetchFields({ fields: ["location", "viewport"] });

      const location = place.location;
      if (location) {
        const coords = { lat: location.lat(), lng: location.lng() };
        const viewport = place.viewport?.toJSON?.() || null;

        // Fit the map to the searched place when Google returns its bounds.
        if (place.viewport) {
          map.fitBounds(place.viewport);
        } else {
          map.panTo(coords);
          map.setZoom(10);
        }

        // Notify parent to add the area
        onPlaceSelect({
          name: displayName,
          center: coords,
          viewport,
          placeId: resolvedPlaceId,
          areaSquareMeters: areaFromViewport(viewport),
        });
      }
    } catch (err) {
      console.error("Failed to fetch place details:", err);
    } finally {
      setIsSubmitting(false);
    }
  }

  return (
    <div className="place-search">
      <div className="place-search-input-wrap">
        <span aria-hidden="true">⌕</span>
        <input
          type="search"
          value={query}
          placeholder="Search a state, district, city..."
          aria-label="Search for a geographic area"
          onChange={(event) => {
            setQuery(event.target.value);
            if (selectedPrediction) setSelectedPrediction(null);
            setHighlightedPredictionIndex(-1);
          }}
          onKeyDown={(event) => {
            if (event.key === "ArrowDown" && predictions.length > 0) {
              event.preventDefault();
              setHighlightedPredictionIndex((currentIndex) =>
                currentIndex < predictions.length - 1 ? currentIndex + 1 : 0
              );
            } else if (event.key === "ArrowUp" && predictions.length > 0) {
              event.preventDefault();
              setHighlightedPredictionIndex((currentIndex) =>
                currentIndex > 0 ? currentIndex - 1 : predictions.length - 1
              );
            } else if (event.key === "Enter") {
              const highlightedPrediction = predictions[highlightedPredictionIndex];
              if (highlightedPrediction) {
                event.preventDefault();
                const displayName =
                  highlightedPrediction.text?.toString() ||
                  highlightedPrediction.mainText?.toString() ||
                  "";
                choosePrediction(highlightedPrediction);
                handleSearch(highlightedPrediction, displayName);
              } else if (selectedPrediction) {
                event.preventDefault();
                handleSearch();
              }
            }
          }}
        />
        {isSearching && <span className="search-spinner" aria-label="Searching" />}
      </div>
      {predictions.length > 0 && (
        <ul className="place-results">
          {predictions.map((prediction, index) => (
            <li key={prediction.placeId}>
              <button
                type="button"
                className={index === highlightedPredictionIndex ? "highlighted" : ""}
                onClick={() => {
                  const displayName =
                    prediction.text?.toString() || prediction.mainText?.toString() || "";
                  choosePrediction(prediction);
                  handleSearch(prediction, displayName);
                }}
              >
                <strong>{prediction.mainText?.toString() || prediction.text?.toString()}</strong>
                <span>{prediction.secondaryText?.toString() || ""}</span>
              </button>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

function MapEvents({
  mode,
  center,
  setCenter,
  setLiveRadius,
  rectangleStart,
  setRectangleStart,
  setLiveBounds,
  setMode,
  selectedDate,
  setAreas,
  selectedAreaId,
  setSelectedAreaId,
  areas,
}) {
  const map = useMap();

  useEffect(() => {
    if (!map) return;

    const clickListener = google.maps.event.addListener(map, "click", (event) => {
      const latLng = event.latLng;
      if (!latLng) return;

      const coords = { lat: latLng.lat(), lng: latLng.lng() };

      if (mode === "PLACING") {
        setCenter(coords);
        setLiveRadius(0);
        setMode("DRAWING");
      } else if (mode === "DRAWING") {
        if (center) {
          const radiusMeters = distanceMeters(center, coords);
          setAreas((prev) => [
            ...prev,
            {
              id: crypto.randomUUID(),
              date: selectedDate,
              center: { ...center },
              radiusMeters,
              areaSquareMeters: areaFromRadius(radiusMeters),
            },
          ]);
          setCenter(null);
          setLiveRadius(0);
          setMode("IDLE");
        }
      } else if (mode === "RECTANGLE_DRAWING" || mode === "RECTANGLE_EDITING") {
        if (!rectangleStart) {
          setRectangleStart(coords);
          setLiveBounds(normalizeBounds(coords, coords));
          return;
        }

        const bounds = normalizeBounds(rectangleStart, coords);
        const areaSquareMeters = areaFromBounds(bounds);
        const centerPoint = centerFromBounds(bounds);
        if (mode === "RECTANGLE_EDITING") {
          setAreas((prev) =>
            prev.map((area) =>
              area.id === selectedAreaId
                ? { ...area, bounds, center: centerPoint, areaSquareMeters }
                : area
            )
          );
          setSelectedAreaId(null);
        } else {
          setAreas((prev) => [
            ...prev,
            {
              id: crypto.randomUUID(),
              name: "",
              shape: "rectangle",
              date: selectedDate,
              center: centerPoint,
              bounds,
              radiusMeters: null,
              areaSquareMeters,
            },
          ]);
        }
        setRectangleStart(null);
        setLiveBounds(null);
        setMode("IDLE");
      } else if (mode === "EDITING") {
        const areaToEdit = areas.find((a) => a.id === selectedAreaId);
        if (areaToEdit) {
          const radiusMeters = distanceMeters(areaToEdit.center, coords);
          setAreas((prev) =>
              prev.map((a) => (a.id === selectedAreaId
                ? { ...a, radiusMeters, areaSquareMeters: areaFromRadius(radiusMeters) }
                : a))
          );
          setLiveRadius(0);
          setMode("IDLE");
          setSelectedAreaId(null);
        }
      }
    });

    const moveListener = google.maps.event.addListener(map, "mousemove", (event) => {
      const latLng = event.latLng;
      if (!latLng) return;
      const cursor = { lat: latLng.lat(), lng: latLng.lng() };

      if (mode === "DRAWING" && center) {
        setLiveRadius(distanceMeters(center, cursor));
      } else if (
        (mode === "RECTANGLE_DRAWING" || mode === "RECTANGLE_EDITING") &&
        rectangleStart
      ) {
        setLiveBounds(normalizeBounds(rectangleStart, cursor));
      } else if (mode === "EDITING") {
        const areaToEdit = areas.find((a) => a.id === selectedAreaId);
        if (areaToEdit) {
          setLiveRadius(distanceMeters(areaToEdit.center, cursor));
        }
      }
    });

    return () => {
      google.maps.event.removeListener(clickListener);
      google.maps.event.removeListener(moveListener);
    };
  }, [
    map,
    mode,
    center,
    selectedDate,
    setCenter,
    setLiveRadius,
    rectangleStart,
    setRectangleStart,
    setLiveBounds,
    setMode,
    setAreas,
    selectedAreaId,
    setSelectedAreaId,
    areas,
  ]);

  return null;
}

function DragDropManager({ isDraggingDelete, setIsDraggingDelete, areas, deleteArea }) {
  const map = useMap();
  const overlayRef = React.useRef(null);

  useEffect(() => {
    if (!map) return;

    const overlay = new google.maps.OverlayView();
    overlay.draw = () => {};
    overlay.setMap(map);
    overlayRef.current = overlay;

    return () => {
      overlay.setMap(null);
      overlayRef.current = null;
    };
  }, [map]);

  useEffect(() => {
    if (!isDraggingDelete) return;

    const handleMouseUp = (e) => {
      try {
        const projection = overlayRef.current?.getProjection();
        const mapDiv = map?.getDiv();
        if (projection && mapDiv) {
          const rect = mapDiv.getBoundingClientRect();
          const x = e.clientX - rect.left;
          const y = e.clientY - rect.top;
          const insideMap = x >= 0 && y >= 0 && x <= rect.width && y <= rect.height;

          if (insideMap) {
            const latLng = projection.fromContainerPixelToLatLng(
              new google.maps.Point(x, y)
            );
            if (latLng) {
              const dropLatLng = { lat: latLng.lat(), lng: latLng.lng() };
              const areaToDelete = findAreaAtDrop(projection, areas, dropLatLng, { x, y });
              if (areaToDelete) {
                deleteArea(areaToDelete);
              }
            }
          }
        }
      } finally {
        setIsDraggingDelete(false);
      }
    };

    window.addEventListener("mouseup", handleMouseUp);
    return () => window.removeEventListener("mouseup", handleMouseUp);
  }, [isDraggingDelete, map, areas, deleteArea, setIsDraggingDelete]);

  return null;
}

function SelectedBoundary({ placeIds, mapId }) {
  const map = useMap();

  useEffect(() => {
    if (!map || !mapId || placeIds.length === 0 || !google.maps.FeatureType) return undefined;
    const selectedPlaceIds = new Set(placeIds);

    const featureTypes = [
      google.maps.FeatureType.ADMINISTRATIVE_AREA_LEVEL_1,
      google.maps.FeatureType.ADMINISTRATIVE_AREA_LEVEL_2,
      google.maps.FeatureType.LOCALITY,
    ].filter(Boolean);
    let layers;

    try {
      layers = featureTypes
        .map((featureType) => map.getFeatureLayer(featureType))
        .filter(Boolean);
    } catch {
      return undefined;
    }

    const style = ({ feature }) => {
      if (!selectedPlaceIds.has(feature.placeId)) return null;

      return {
        fillColor: "#ef4444",
        fillOpacity: 0.13,
        strokeColor: "#b91c1c",
        strokeOpacity: 1,
        strokeWeight: 3,
      };
    };

    try {
      layers.forEach((layer) => {
        layer.style = style;
      });
    } catch {
      return undefined;
    }

    return () => {
      layers.forEach((layer) => {
        layer.style = null;
      });
    };
  }, [map, mapId, placeIds]);

  return null;
}

export default function DisasterMap() {
  const apiKey = process.env.NEXT_PUBLIC_GOOGLE_MAPS_API_KEY;
  const mapId = process.env.NEXT_PUBLIC_GOOGLE_MAPS_MAP_ID;
  const boundariesEnabled = process.env.NEXT_PUBLIC_GOOGLE_MAPS_BOUNDARIES_ENABLED === "true";
  const weatherTileApiKey = process.env.NEXT_PUBLIC_WEATHER_MAP_API_KEY;

  // State Machine: 'IDLE' | 'PLACING' | 'DRAWING' | 'RECTANGLE_DRAWING' | 'RECTANGLE_EDITING' | 'EDITING'
  const [mode, setMode] = useState("IDLE");
  const [center, setCenter] = useState(null);
  const [liveRadius, setLiveRadius] = useState(0);
  const [rectangleStart, setRectangleStart] = useState(null);
  const [liveBounds, setLiveBounds] = useState(null);
  const [selectedAreaId, setSelectedAreaId] = useState(null);
  const [selectedDate, setSelectedDate] = useState("");
  const [selectedTime, setSelectedTime] = useState("");
  const [areas, setAreas] = useState([]);
  const [isTableCollapsed, setIsTableCollapsed] = useState(false);
  const [isDraggingDelete, setIsDraggingDelete] = useState(false);
  const [dragPos, setDragPos] = useState({ x: 0, y: 0 });
  const [showRadiusNotice, setShowRadiusNotice] = useState(false);
  const [boundsError, setBoundsError] = useState("");
  const [isSubmittingAreas, setIsSubmittingAreas] = useState(false);
  const [cloudsEnabled, setCloudsEnabled] = useState(false);
  const [cloudTileError, setCloudTileError] = useState("");
  const boundaryPlaceIds = useMemo(
    () => areas.map((area) => area.placeId).filter(Boolean),
    [areas]
  );

  useEffect(() => {
    setSelectedDate(todayString());
    setSelectedTime(todayTimeString());
  }, []);

  useEffect(() => {
    if (mode === "PLACING" || mode === "DRAWING" || mode === "RECTANGLE_DRAWING") {
      setSelectedAreaId(null);
      setRectangleStart(null);
      setLiveBounds(null);
    }
    if (mode === "IDLE") {
      setRectangleStart(null);
      setLiveBounds(null);
    }
  }, [mode]);

  const handleCloudTileError = useCallback((reason) => {
    setCloudTileError(reason);
  }, []);

  useEffect(() => {
    const handleMouseMove = (e) => {
      if (isDraggingDelete) {
        setDragPos({ x: e.clientX, y: e.clientY });
      }
    };
    window.addEventListener("mousemove", handleMouseMove);
    return () => window.removeEventListener("mousemove", handleMouseMove);
  }, [isDraggingDelete]);

  function updateArea(id, updates) {
    setAreas((prev) =>
      prev.map((area) => (area.id === id ? { ...area, ...updates } : area))
    );
  }

  const mapOptions = useMemo(
    () => ({
      // Map restriction is disabled for now to allow users to explore outside India, but you can enable it if needed.
      // restriction: { latLngBounds: INDIA_BOUNDS, strictBounds: false },
      mapTypeControl: true,
      streetViewControl: true,
      fullscreenControl: true,
      zoomControl: true,
      clickableIcons: false,
    }),
    []
  );

  function deleteArea(id) {
    setAreas((prev) => prev.filter((area) => area.id !== id));
    if (selectedAreaId === id) {
      setSelectedAreaId(null);
    }
  }

  async function handlePlaceSelect({ name, center: placeCenter, viewport, placeId, areaSquareMeters }) {
    setBoundsError("");
    const newId = crypto.randomUUID();
    setAreas((prev) => [
      ...prev,
      {
        id: newId,
        name,
        date: selectedDate || todayString(),
        center: placeCenter,
        radiusMeters: null,
        viewport,
        areaSquareMeters,
        placeId,
      },
    ]);
    setShowRadiusNotice(true);
  }

  async function submitSelectedAreas() {
    if (areas.length === 0) {
      setBoundsError("Search and select an area before submitting.");
      return;
    }

    setIsSubmittingAreas(true);
    setBoundsError("");
    try {
      const results = await Promise.all(
        areas.map(async (area) => {
          const response = await fetch("/api/bounds", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ location: area.name }),
          });
          const payload = await response.json().catch(() => ({}));
          if (!response.ok || payload.error) {
            throw new Error(payload.error || `Could not find the boundary for ${area.name}.`);
          }
          return { id: area.id, payload };
        })
      );

      setAreas((currentAreas) => currentAreas.map((area) => {
        const result = results.find((item) => item.id === area.id);
        return result
          ? {
              ...area,
              extremePoints: result.payload.extreme_points,
              coordinateLimits: result.payload.limits,
              boundaryGeoJson: result.payload.geojson,
            }
          : area;
      }));
    } catch (error) {
      setBoundsError(error.message || "Could not calculate the selected area's extreme points.");
    } finally {
      setIsSubmittingAreas(false);
    }
  }

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
    <APIProvider apiKey={apiKey} libraries={["places"]}>
      <main className="app">
        <section className="map-section">
          <div className="map-search-panel">
            <PlaceSearch onPlaceSelect={handlePlaceSelect} />
          </div>

          {showRadiusNotice && (
            <div className="radius-notice" role="status">
              <span>Adjust the selected area on the map to change its size.</span>
              <button
                type="button"
                className="radius-notice-dismiss"
                aria-label="Dismiss radius notice"
                onClick={() => setShowRadiusNotice(false)}
              >
                &times;
              </button>
            </div>
          )}

          <div className="floating-menu">
            <button
              className={`menu-item ${
                ["PLACING", "DRAWING", "EDITING"].includes(mode) ? "active" : ""
              }`}
              onClick={() => {
                if (mode === "IDLE") setMode("PLACING");
                else setMode("IDLE");
              }}
              title="Add Pin"
            >
              📍
            </button>
            <button
              className={`menu-item ${
                mode === "RECTANGLE_DRAWING" ? "active" : ""
              }`}
              onClick={() => {
                if (mode === "RECTANGLE_DRAWING") {
                  setMode("IDLE");
                  setRectangleStart(null);
                  setLiveBounds(null);
                } else {
                  setMode("RECTANGLE_DRAWING");
                  setCenter(null);
                  setLiveRadius(0);
                }
              }}
              title="Add rectangle"
            >
              <svg
                className="rectangle-tool-icon"
                viewBox="0 0 24 24"
                aria-hidden="true"
              >
                <rect x="4" y="5" width="16" height="14" rx="1.5" />
                <path d="M4 9h16M8 5v14" />
              </svg>
            </button>
            <button
              type="button"
              className="menu-item"
              onClick={submitSelectedAreas}
              disabled={isSubmittingAreas}
              title="Submit selected area and show extreme points"
            >
              ✅
            </button>
            <button
              type="button"
              className={`menu-item delete-drag-handle ${isDraggingDelete ? "dragging" : ""}`}
              draggable={false}
              onMouseDown={(e) => {
                e.preventDefault();
                setIsDraggingDelete(true);
                setDragPos({ x: e.clientX, y: e.clientY });
              }}
              title="Drag to delete a pin"
            >
              🗑️
            </button>
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
            {/*<label className="cloud-timeline">
              <span>Timeline date and time</span>
              <input
                type="date"
                value={selectedDate}
                onChange={(event) => {
                  setSelectedDate(event.target.value);
                }}
                aria-label="Timeline date"
              />
              <input
                type="time"
                value={selectedTime}
                onChange={(event) => {
                  setSelectedTime(event.target.value);
                }}
                aria-label="Timeline time"
              />
            </label>*/}
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

          {isDraggingDelete && (
            <div
              className="dragging-delete-icon"
              style={{ left: dragPos.x, top: dragPos.y }}
            >
              🗑️
            </div>
          )}

          <Map
            className="map"
            mapId={mapId}
            defaultCenter={INDIA_CENTER}
            defaultZoom={5}
            gestureHandling={
              ["DRAWING", "RECTANGLE_DRAWING", "RECTANGLE_EDITING"].includes(mode)
                ? "none"
                : "auto"
            }
            {...mapOptions}
          >
            <CloudTileLayer
              enabled={cloudsEnabled}
              apiKey={weatherTileApiKey}
              onTileError={handleCloudTileError}
            />
            {boundariesEnabled && (
              <SelectedBoundary placeIds={boundaryPlaceIds} mapId={mapId} />
            )}
            <DragDropManager
              isDraggingDelete={isDraggingDelete}
              setIsDraggingDelete={setIsDraggingDelete}
              areas={areas}
              deleteArea={deleteArea}
            />
            <MapEvents
              mode={mode}
              center={center}
              setCenter={setCenter}
              setLiveRadius={setLiveRadius}
              rectangleStart={rectangleStart}
              setRectangleStart={setRectangleStart}
              setLiveBounds={setLiveBounds}
              setMode={setMode}
              selectedDate={selectedDate}
              setAreas={setAreas}
              selectedAreaId={selectedAreaId}
              setSelectedAreaId={setSelectedAreaId}
              areas={areas}
            />
            <GradientCircle
              latitude={INDIA_CENTER.lat}
              longitude={INDIA_CENTER.lng}
              radius={250000}
            />
            {center && (
              <>
                <Marker position={center} />
                {liveRadius > 0 && (
                  <Circle
                    center={center}
                    radius={liveRadius}
                    strokeColor="#b91c1c"
                    strokeOpacity={0.95}
                    strokeWeight={2}
                    fillColor="#ef4444"
                    fillOpacity={0.16}
                    clickable={false}
                  />
                )}
              </>
            )}
            {liveBounds && (
              <Rectangle
                bounds={liveBounds}
                strokeColor="#b91c1c"
                strokeOpacity={0.95}
                strokeWeight={2}
                fillColor="#ef4444"
                fillOpacity={0.16}
                clickable={false}
              />
            )}
            {areas.map((area, index) => (
              <React.Fragment key={area.id}>
                {area.shape !== "rectangle" && <Marker
                  position={area.center}
                  title={`Pin ${index + 1}`}
                  draggable={mode === "EDITING" && selectedAreaId === area.id}
                  onDragEnd={(e) => {
                    const newPos = e.latLng;
                    if (newPos) {
                      updateArea(area.id, {
                        center: { lat: newPos.lat(), lng: newPos.lng() },
                      });
                    }
                  }}
                    onClick={() => {
                      setSelectedAreaId(area.id);
                      setMode("EDITING");
                    }}
                  />}
                {area.extremePoints && (
                  <>
                    {Object.entries(area.extremePoints).map(([direction, point]) => (
                      <Marker
                        key={`${area.id}-${direction}`}
                        position={{ lat: point.lat, lng: point.lon }}
                        title={`${direction} extreme: ${point.lat}, ${point.lon}`}
                      />
                    ))}
                  </>
                )}
                {(() => {
                  const isEditing = mode === "EDITING" && selectedAreaId === area.id;
                  const displayRadius = isEditing && liveRadius > 0 ? liveRadius : area.radiusMeters;
                  
                  return displayRadius != null && displayRadius > 0 ? (
                    <Circle
                      center={area.center}
                      radius={displayRadius}
                      strokeColor={isEditing ? "#3b82f6" : "#10b981"}
                      strokeOpacity={0.9}
                      strokeWeight={isEditing ? 3 : 2}
                      fillColor={isEditing ? "#3b82f6" : "#10b981"}
                      fillOpacity={0.13}
                      clickable={false}
                    />
                  ) : null;
                })()}
                {area.shape === "rectangle" && area.bounds && (
                  <Rectangle
                    bounds={area.bounds}
                    strokeColor={
                      mode === "RECTANGLE_EDITING" && selectedAreaId === area.id
                        ? "#3b82f6"
                        : "#10b981"
                    }
                    strokeOpacity={0.9}
                    strokeWeight={
                      mode === "RECTANGLE_EDITING" && selectedAreaId === area.id
                        ? 3
                        : 2
                    }
                    fillColor={
                      mode === "RECTANGLE_EDITING" && selectedAreaId === area.id
                        ? "#3b82f6"
                        : "#10b981"
                    }
                    fillOpacity={0.13}
                    clickable
                    onClick={() => {
                      setSelectedAreaId(area.id);
                      setRectangleStart(null);
                      setLiveBounds(area.bounds);
                      setMode("RECTANGLE_EDITING");
                    }}
                  />
                )}
              </React.Fragment>
            ))}
          </Map>
        </section>

        <section className="panel">
          {boundsError && (
            <div className="status" role="alert">
              {boundsError}
            </div>
          )}
          {(mode === "DRAWING" || mode === "EDITING") && (
            <div className="creation-controls">
              <div className="live-stat">
                <span>Center</span>
                <strong>
                  {mode === "EDITING"
                    ? (areas.find(a => a.id === selectedAreaId)?.center
                        ? `${areas.find(a => a.id === selectedAreaId).center.lat.toFixed(6)}, ${areas.find(a => a.id === selectedAreaId).center.lng.toFixed(6)}`
                        : "Selecting...")
                    : (center ? `${center.lat.toFixed(6)}, ${center.lng.toFixed(6)}` : "Selecting...")}
                </strong>
              </div>
              <div className="live-stat">
                <span>Area</span>
                <strong>
                  {formatArea(
                    mode === "EDITING" && liveRadius === 0
                      ? getAreaSquareMeters(areas.find((area) => area.id === selectedAreaId))
                      : areaFromRadius(liveRadius)
                  ).km || "0.000"} km²
                </strong>
              </div>
            </div>
          )}
          {(mode === "RECTANGLE_DRAWING" || mode === "RECTANGLE_EDITING") && (
            <div className="creation-controls">
              <div className="live-stat">
                <span>Rectangle</span>
                <strong>
                  {rectangleStart
                    ? liveBounds
                      ? `${liveBounds.south.toFixed(6)}, ${liveBounds.west.toFixed(6)} to ${liveBounds.north.toFixed(6)}, ${liveBounds.east.toFixed(6)}`
                      : "Selecting second corner..."
                    : "Select first corner"}
                </strong>
              </div>
              <div className="live-stat">
                <span>Area</span>
                <strong>{formatArea(areaFromBounds(liveBounds)).km || "0.000"} km²</strong>
              </div>
            </div>
          )}

          <div className={`table-wrap ${isTableCollapsed ? "collapsed" : ""}`}>
            <table>
              <thead onClick={() => setIsTableCollapsed(!isTableCollapsed)} style={{ cursor: "pointer" }}>
                <tr className="collapsible-header">
                  <th>Area</th>
                  <th>Type</th>
                  <th>Date</th>
                  <th>Center Latitude</th>
                  <th>Center Longitude</th>
                  <th>Area (km²)</th>
                  <th>Radius (km)</th>
                  <th>North Point</th>
                  <th>South Point</th>
                  <th>East Point</th>
                  <th>West Point</th>
                  <th>Action</th>
                  <th className="collapse-indicator">{isTableCollapsed ? "Expand" : "Collapse"}</th>
                </tr>
              </thead>
              {!isTableCollapsed && (
                <tbody>
                  {areas.length === 0 ? (
                    <tr>
                      <td colSpan="13" className="empty">No disaster areas created yet.</td>
                    </tr>
                  ) : (
                    areas.map((area, index) => (
                      <tr
                        key={area.id}
                        style={{
                          backgroundColor: selectedAreaId === area.id ? "#e0f2fe" : "transparent",
                          cursor: "default",
                        }}
                      >
                        <td>
                          <input
                            type="text"
                            className="table-text-input"
                            value={
                              area.name ||
                              `${area.shape === "rectangle" ? "Rectangle" : "Pin"} ${index + 1}`
                            }
                            onClick={(e) => e.stopPropagation()}
                            onChange={(e) => updateArea(area.id, { name: e.target.value })}
                          />
                        </td>
                        <td>{area.shape === "rectangle" ? "Rectangle" : "Pin"}</td>
                        <td>
                          <input
                            type="date"
                            className="table-date-input"
                            value={area.date}
                            onClick={(e) => e.stopPropagation()}
                            onChange={(e) => updateArea(area.id, { date: e.target.value })}
                          />
                        </td>
                        <td>
                          <input
                            type="number"
                            className="table-num-input"
                            value={area.center?.lat?.toFixed(6) ?? ""}
                            onClick={(e) => e.stopPropagation()}
                            onChange={(e) => {
                              const val = parseFloat(e.target.value);
                              if (!isNaN(val)) {
                                updateArea(area.id, { center: { ...area.center, lat: val } });
                              }
                            }}
                          />
                        </td>
                        <td>
                          <input
                            type="number"
                            className="table-num-input"
                            value={area.center?.lng?.toFixed(6) ?? ""}
                            onClick={(e) => e.stopPropagation()}
                            onChange={(e) => {
                              const val = parseFloat(e.target.value);
                              if (!isNaN(val)) {
                                updateArea(area.id, { center: { ...area.center, lng: val } });
                              }
                            }}
                          />
                        </td>
                        <td>{formatArea(getAreaSquareMeters(area)).km}</td>
                        <td>
                          {area.radiusMeters > 0 ? (area.radiusMeters / 1000).toFixed(3) : ""}
                        </td>
                        {[
                          area.extremePoints?.north,
                          area.extremePoints?.south,
                          area.extremePoints?.east,
                          area.extremePoints?.west,
                        ].map((point, pointIndex) => (
                          <td className="coordinate-limits" key={`${area.id}-point-${pointIndex}`}>
                            {point ? `${point.lat.toFixed(6)}, ${point.lon.toFixed(6)}` : "-"}
                          </td>
                        ))}
                        <td className="action-cell">
                          {area.shape === "rectangle" && (
                            <button
                              type="button"
                              className="delete-btn"
                              title="Edit rectangle"
                              onClick={(e) => {
                                e.stopPropagation();
                                setSelectedAreaId(area.id);
                                setRectangleStart(null);
                                setLiveBounds(area.bounds);
                                setMode("RECTANGLE_EDITING");
                              }}
                            >
                              ✏️
                            </button>
                          )}
                          <button
                            className="delete-btn"
                            onClick={(e) => {
                              e.stopPropagation();
                              deleteArea(area.id);
                            }}
                            style={{
                              background: "none",
                              border: "none",
                              cursor: "pointer",
                              fontSize: "14px",
                            }}
                          >
                            🗑️
                          </button>
                        </td>
                      </tr>
                    ))
                  )}
                </tbody>
              )}
            </table>
          </div>
        </section>
      </main>
    </APIProvider>
  );
}
