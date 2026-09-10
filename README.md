# OneHack Next.js + Google Maps

This project is a Next.js App Router starter for the OneHack disaster-prediction UI.

It uses:

- Next.js 16
- React 19
- `@vis.gl/react-google-maps`
- Google Maps JavaScript API

The current `@vis.gl/react-google-maps` documentation supports Next.js/App Router when the map component is a client component, because Google Maps cannot be server-rendered. The project therefore keeps the interactive map in `app/DisasterMap.jsx` with `"use client"`.

## Setup

### 1. Add your Google Maps API key

Copy:

```text
.env.example
```

to:

```text
.env.local
```

Then set:

```env
NEXT_PUBLIC_GOOGLE_MAPS_API_KEY=YOUR_GOOGLE_MAPS_API_KEY
```

Because the Google Maps JavaScript API runs in the browser, this key is necessarily exposed to the browser. Use a browser-restricted API key; do not put a server secret/service-account credential in `NEXT_PUBLIC_*`.

### 2. Install

```bash
npm install
```

### 3. Run

```bash
npm run dev
```

Open:

```text
http://localhost:3000
```

## Google Cloud setup

Make sure your Google Cloud project has billing configured and the required Google Maps Platform services enabled for the JavaScript map you use.

For deployment, restrict the browser key to the domains/origins used by your app.

## Current interaction

1. Click `📍 Add Pin`.
2. Click the map to place the center.
3. The map locks while you are sizing the area.
4. Move the cursor away from the center to grow/shrink the circle.
5. Pick any past, present, or future date.
6. Click again to save.
7. The map unlocks automatically.
8. Create Pin 2, Pin 3, etc.
9. Every saved area appears as another table row.

Saved data in the browser currently has this structure:

```json
{
  "date": "2026-09-10",
  "center": {
    "lat": 28.6139,
    "lng": 77.209
  },
  "radiusMeters": 24500
}
```

## Next Flask integration

When you are ready, each saved area can be POSTed to your Flask API:

```http
POST /api/disaster/predict
Content-Type: application/json
```

with:

```json
{
  "date": "2026-09-10",
  "center": {
    "lat": 28.6139,
    "lng": 77.209
  },
  "radius_km": 24.5
}
```


## Important implementation note

`@vis.gl/react-google-maps` exposes the map click/mouse event position through
`event.detail.latLng`. In this app it is consumed as a plain `{ lat, lng }`
literal, so the code uses:

```js
event.detail.lat
event.detail.lng
```

rather than the older Google Maps class-style:

```js
event.detail.latLng.lat()
event.detail.latLng.lng()
```
