"use client";

import React, { useMemo, useState, useEffect } from "react";
import {
  APIProvider,
  Map,
  Marker,
  Circle,
  useMapsLibrary,
  useMap,
} from "@vis.gl/react-google-maps";

const INDIA_CENTER = { lat: 22.5937, lng: 78.9629 };
const INDIA_BOUNDS = {
  north: 37.6,
  south: 6.2,
  east: 97.7,
  west: 68.0,
};

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

function findAreaAtDrop(projection, areas, dropLatLng, dropPixel) {
  let bestId = null;
  let bestScore = Infinity;

  areas.forEach((area) => {
    const dist = distanceMeters(area.center, dropLatLng);
    const contains = dist <= (area.radiusMeters || 0);
    const pinPixel = projection.fromLatLngToContainerPixel(
      new google.maps.LatLng(area.center.lat, area.center.lng)
    );
    const pixelDist = pinPixel
      ? Math.hypot(dropPixel.x - pinPixel.x, dropPixel.y - pinPixel.y)
      : Infinity;
    const onPin = pixelDist <= 28;

    if (!contains && !onPin) return;

    const score = contains ? area.radiusMeters : dist;
    if (score < bestScore) {
      bestScore = score;
      bestId = area.id;
    }
  });

  return bestId;
}

function todayString() {
  const now = new Date();
  const year = now.getFullYear();
  const month = String(now.getMonth() + 1).padStart(2, "0");
  const day = String(now.getDate()).padStart(2, "0");
  return `${year}-${month}-${day}`;
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

    setIsSubmitting(true);
    try {
      const placeId = prediction.placeId;

      // Use the new Place class to fetch location details
      const place = new places.Place({ id: placeId });
      await place.fetchFields({ fields: ["location", "viewport"] });

      const location = place.location;
      if (location) {
        const coords = { lat: location.lat(), lng: location.lng() };

        // Always pan and zoom to the searched place reliably
        map.panTo(coords);
        map.setZoom(10);

        // Notify parent to add the area
        onPlaceSelect({
          name: displayName,
          center: coords,
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
        <button
          type="button"
          className="search-btn"
          disabled={!selectedPrediction || isSubmitting}
          onClick={handleSearch}
          title="Search and add to map"
        >
          {isSubmitting ? "…" : "Search"}
        </button>
      </div>
      {predictions.length > 0 && (
        <ul className="place-results">
          {predictions.map((prediction, index) => (
            <li key={prediction.placeId}>
              <button
                type="button"
                className={index === highlightedPredictionIndex ? "highlighted" : ""}
                onClick={() => choosePrediction(prediction)}
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
            },
          ]);
          setCenter(null);
          setLiveRadius(0);
          setMode("IDLE");
        }
      } else if (mode === "EDITING") {
        const areaToEdit = areas.find((a) => a.id === selectedAreaId);
        if (areaToEdit) {
          const radiusMeters = distanceMeters(areaToEdit.center, coords);
          setAreas((prev) =>
            prev.map((a) => (a.id === selectedAreaId ? { ...a, radiusMeters } : a))
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
  }, [map, mode, center, selectedDate, setCenter, setLiveRadius, setMode, setAreas, selectedAreaId, setSelectedAreaId, areas]);

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

export default function DisasterMap() {
  const apiKey = process.env.NEXT_PUBLIC_GOOGLE_MAPS_API_KEY;

  // State Machine: 'IDLE' | 'PLACING' | 'DRAWING' | 'EDITING'
  const [mode, setMode] = useState("IDLE");
  const [center, setCenter] = useState(null);
  const [liveRadius, setLiveRadius] = useState(0);
  const [selectedAreaId, setSelectedAreaId] = useState(null);
  const [selectedDate, setSelectedDate] = useState("");
  const [areas, setAreas] = useState([]);
  const [isTableCollapsed, setIsTableCollapsed] = useState(false);
  const [isDraggingDelete, setIsDraggingDelete] = useState(false);
  const [dragPos, setDragPos] = useState({ x: 0, y: 0 });
  const [showRadiusNotice, setShowRadiusNotice] = useState(false);

  useEffect(() => {
    setSelectedDate(todayString());
  }, []);

  useEffect(() => {
    if (mode === "PLACING" || mode === "DRAWING") {
      setSelectedAreaId(null);
    }
  }, [mode]);

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
            <PlaceSearch
              onPlaceSelect={({ name, center: placeCenter }) => {
                const newId = crypto.randomUUID();
                setAreas((prev) => [
                  ...prev,
                  {
                    id: newId,
                    name,
                    date: selectedDate || todayString(),
                    center: placeCenter,
                    radiusMeters: null,
                  },
                ]);
                setShowRadiusNotice(true);
              }}
            />
          </div>

          {showRadiusNotice && (
            <div className="radius-notice" role="status">
              <span>Increase the area using the radius size/length option below.</span>
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
              className={`menu-item ${mode !== "IDLE" ? "active" : ""}`}
              onClick={() => {
                if (mode === "IDLE") setMode("PLACING");
                else setMode("IDLE");
              }}
              title="Add Pin"
            >
              📍
            </button>
            <button className="menu-item" onClick={() => alert("Submission feature coming soon!")} title="Submit">
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
            defaultCenter={INDIA_CENTER}
            defaultZoom={5}
            gestureHandling={mode === "DRAWING" ? "none" : "auto"}
            {...mapOptions}
          >
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
              setMode={setMode}
              selectedDate={selectedDate}
              setAreas={setAreas}
              selectedAreaId={selectedAreaId}
              setSelectedAreaId={setSelectedAreaId}
              areas={areas}
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
            {areas.map((area, index) => (
              <React.Fragment key={area.id}>
                <Marker
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
                />
                {(() => {
                  const isEditing = mode === "EDITING" && selectedAreaId === area.id;
                  const displayRadius = isEditing && liveRadius > 0 ? liveRadius : area.radiusMeters;
                  
                  return displayRadius != null && displayRadius > 0 ? (
                    <Circle
                      center={area.center}
                      radius={displayRadius}
                      strokeColor={isEditing ? "#3b82f6" : "#b91c1c"}
                      strokeOpacity={0.9}
                      strokeWeight={isEditing ? 3 : 2}
                      fillColor={isEditing ? "#3b82f6" : "#ef4444"}
                      fillOpacity={0.13}
                      clickable={false}
                    />
                  ) : null;
                })()}
              </React.Fragment>
            ))}
          </Map>
        </section>

        <section className="panel">
          

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
                <span>Radius</span>
                <strong>{(liveRadius / 1000).toFixed(3)} km</strong>
              </div>
            </div>
          )}

          <div className={`table-wrap ${isTableCollapsed ? "collapsed" : ""}`}>
            <table>
              <thead onClick={() => setIsTableCollapsed(!isTableCollapsed)} style={{ cursor: "pointer" }}>
                <tr className="collapsible-header">
                  <th>Pin</th>
                  <th>Date</th>
                  <th>Center Latitude</th>
                  <th>Center Longitude</th>
                  <th>Radius (km)</th>
                  <th>Radius (m)</th>
                  <th>Action</th>
                  <th className="collapse-indicator">{isTableCollapsed ? "Expand" : "Collapse"}</th>
                </tr>
              </thead>
              {!isTableCollapsed && (
                <tbody>
                  {areas.length === 0 ? (
                    <tr>
                      <td colSpan="7" className="empty">No disaster areas created yet.</td>
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
                            value={area.name || `Pin ${index + 1}`}
                            onClick={(e) => e.stopPropagation()}
                            onChange={(e) => updateArea(area.id, { name: e.target.value })}
                          />
                        </td>
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
                        <td>
                          <input
                            type="number"
                            className="table-num-input"
                            value={area.radiusMeters != null && area.radiusMeters !== 0 ? (area.radiusMeters / 1000).toFixed(3) : ""}
                            onClick={(e) => e.stopPropagation()}
                            onChange={(e) => {
                              const val = parseFloat(e.target.value);
                              if (!isNaN(val)) {
                                updateArea(area.id, { radiusMeters: val * 1000 });
                              }
                            }}
                          />
                        </td>
                        <td>
                          <input
                            type="number"
                            className="table-num-input"
                            value={area.radiusMeters?.toFixed(2) ?? ""}
                            onClick={(e) => e.stopPropagation()}
                            onChange={(e) => {
                              const val = parseFloat(e.target.value);
                              if (!isNaN(val)) {
                                updateArea(area.id, { radiusMeters: val });
                              }
                            }}
                          />
                        </td>
                        <td>
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
