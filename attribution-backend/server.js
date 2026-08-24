/**
 * AIS Vessel Attribution Backend
 * ================================
 * Exposes POST /attribute — given a detected oil spill's location, time,
 * and (optionally) a hindcast drift trajectory from OpenDrift, this scores
 * nearby vessels from historic AIS data and returns ranked candidates.
 *
 * Scoring = weighted combination of:
 *   1. Proximity      - how close the vessel was to the spill location around the spill time
 *   2. Trajectory match - how well the vessel's AIS track lines up with the
 *                         OpenDrift hindcast (backtracked) drift path
 *   3. Anomaly score   - AIS signal gaps / erratic speed or course changes
 *                        near the spill time (red flags for illegal discharge)
 *
 * Install:
 *   npm install express csv-parser
 *
 * Run:
 *   node server.js
 *
 * AIS data:
 *   Download a daily AIS CSV file from MarineCadastre (free, no login):
 *   https://coast.noaa.gov/htdata/CMSP/AISDataHandler/2024/index.html
 *   e.g. https://coast.noaa.gov/htdata/CMSP/AISDataHandler/2024/AIS_2024_03_15.zip
 *   Unzip and point AIS_CSV_PATH below (or via env var) at the .csv file.
 *
 *   Interactive alternative (draw a bounding box + date range in browser):
 *   https://marinecadastre.gov/accessais/
 */

const express = require("express");
const cors = require("cors");
const fs = require("fs");
const csv = require("csv-parser");

const app = express();

app.use(
  cors({
    origin: "https://oiltrace-e2da1.web.app",
  })
);

app.use(express.json());

const PORT = process.env.PORT || 4000;
const AIS_CSV_PATH = process.env.AIS_CSV_PATH || "./AIS_2024_11_07.csv";

// Scoring weights - tune these based on demo results
const WEIGHTS = {
  proximity: 0.45,
  trajectory: 0.35,
  anomaly: 0.20,
};

// Only consider AIS pings within this window around the spill time (minutes)
const TIME_WINDOW_MINUTES = 180;
// Only consider vessels within this radius of the spill location (km)
const SEARCH_RADIUS_KM = 50;

// ----------------------------------------------------------------------
// Geo helpers
// ----------------------------------------------------------------------
function haversineKm(lat1, lon1, lat2, lon2) {
  const R = 6371; // Earth radius in km
  const dLat = ((lat2 - lat1) * Math.PI) / 180;
  const dLon = ((lon2 - lon1) * Math.PI) / 180;
  const a =
    Math.sin(dLat / 2) ** 2 +
    Math.cos((lat1 * Math.PI) / 180) *
      Math.cos((lat2 * Math.PI) / 180) *
      Math.sin(dLon / 2) ** 2;
  const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
  return R * c;
}

function minutesBetween(t1, t2) {
  return Math.abs(new Date(t1) - new Date(t2)) / 60000;
}

// ----------------------------------------------------------------------
// Load & group AIS pings by vessel (MMSI), once at startup
// ----------------------------------------------------------------------
let vesselTracks = {}; // { mmsi: [{ lat, lon, timestamp, speed, course, name }, ...] }
let dataLoaded = false;

function loadAisData(csvPath) {
  return new Promise((resolve, reject) => {
    const tracks = {};
    if (!fs.existsSync(csvPath)) {
      console.warn(
        `AIS data file not found at ${csvPath}. /attribute will run with an empty dataset until a file is provided.`
      );
      return resolve(tracks);
    }
    fs.createReadStream(csvPath)
      .pipe(csv())
      .on("data", (row) => {
        // MarineCadastre AIS CSV columns: MMSI, BaseDateTime, LAT, LON, SOG, COG, VesselName, ...
        const mmsi = row.MMSI;
        if (!mmsi) return;
        const lat = parseFloat(row.LAT);
        const lon = parseFloat(row.LON);
        if (Number.isNaN(lat) || Number.isNaN(lon)) return;

        if (!tracks[mmsi]) tracks[mmsi] = [];
        tracks[mmsi].push({
          lat,
          lon,
          timestamp: row.BaseDateTime,
          speed: parseFloat(row.SOG) || 0,
          course: parseFloat(row.COG) || 0,
          name: row.VesselName || "Unknown",
          imo: row.IMO || "N/A",
          vesselType: row.VesselType || "N/A",
        });
      })
      .on("end", () => {
        // sort each vessel's pings chronologically
        Object.values(tracks).forEach((pings) =>
          pings.sort((a, b) => new Date(a.timestamp) - new Date(b.timestamp))
        );
        resolve(tracks);
      })
      .on("error", reject);
  });
}

// ----------------------------------------------------------------------
// Scoring
// ----------------------------------------------------------------------

/** Proximity score: closer distance at the closest-in-time ping = higher score (0-1) */
function proximityScore(pings, spillLat, spillLon, spillTime) {
  let best = null;
  for (const p of pings) {
    if (minutesBetween(p.timestamp, spillTime) > TIME_WINDOW_MINUTES) continue;
    const dist = haversineKm(spillLat, spillLon, p.lat, p.lon);
    if (best === null || dist < best) best = dist;
  }
  if (best === null) return { score: 0, distanceKm: null };
  const score = Math.max(0, 1 - best / SEARCH_RADIUS_KM);
  return { score, distanceKm: best };
}

/** Trajectory score: average distance from vessel pings to the hindcast drift path (0-1) */
function trajectoryScore(pings, driftPath) {
  if (!driftPath || driftPath.length === 0) return { score: 0.5, avgDistKm: null }; // neutral if no drift path given
  const relevant = pings.filter((p) => true); // could further filter by time window
  if (relevant.length === 0) return { score: 0, avgDistKm: null };

  let totalMinDist = 0;
  let count = 0;
  for (const p of relevant) {
    let minDist = Infinity;
    for (const dp of driftPath) {
      const d = haversineKm(p.lat, p.lon, dp.lat, dp.lon);
      if (d < minDist) minDist = d;
    }
    totalMinDist += minDist;
    count++;
  }
  const avgDistKm = totalMinDist / count;
  const score = Math.max(0, 1 - avgDistKm / SEARCH_RADIUS_KM);
  return { score, avgDistKm };
}

/** Anomaly score: AIS gaps and erratic speed/course changes near spill time (0-1, higher = more suspicious) */
function anomalyScore(pings, spillTime) {
  const nearby = pings.filter(
    (p) => minutesBetween(p.timestamp, spillTime) <= TIME_WINDOW_MINUTES
  );
  if (nearby.length < 2) return { score: 0, flags: [] };

  const flags = [];
  let anomalyPoints = 0;

  for (let i = 1; i < nearby.length; i++) {
    const gapMin = minutesBetween(nearby[i].timestamp, nearby[i - 1].timestamp);
    if (gapMin > 30) {
      flags.push(`AIS signal gap of ${Math.round(gapMin)} min`);
      anomalyPoints += 1;
    }
    const speedDelta = Math.abs(nearby[i].speed - nearby[i - 1].speed);
    if (speedDelta > 8) {
      flags.push(`Sudden speed change of ${speedDelta.toFixed(1)} knots`);
      anomalyPoints += 1;
    }
    const courseDelta = Math.abs(nearby[i].course - nearby[i - 1].course);
    if (courseDelta > 45 && courseDelta < 315) {
      flags.push(`Sharp course change of ${courseDelta.toFixed(0)} degrees`);
      anomalyPoints += 1;
    }
  }

  const score = Math.min(1, anomalyPoints / 3); // cap at 1
  return { score, flags: [...new Set(flags)] };
}

// ----------------------------------------------------------------------
// POST /attribute
// ----------------------------------------------------------------------
app.post("/attribute", async (req, res) => {
  try {
    const { spill_lat, spill_lon, spill_time, drift_path, top_k } = req.body;

    if (spill_lat === undefined || spill_lon === undefined || !spill_time) {
      return res.status(400).json({
        error: "spill_lat, spill_lon, and spill_time are required",
      });
    }

    const candidates = [];

    for (const [mmsi, pings] of Object.entries(vesselTracks)) {
      const prox = proximityScore(pings, spill_lat, spill_lon, spill_time);
      if (prox.distanceKm === null) continue; // vessel wasn't near the area in the time window at all

      const traj = trajectoryScore(pings, drift_path);
      const anomaly = anomalyScore(pings, spill_time);

      const finalScore =
        WEIGHTS.proximity * prox.score +
        WEIGHTS.trajectory * traj.score +
        WEIGHTS.anomaly * anomaly.score;

      const latestPing = pings[pings.length - 1];

      candidates.push({
        mmsi,
        vesselName: latestPing.name,
        imo: latestPing.imo,
        vesselType: latestPing.vesselType,
        confidence: Math.round(finalScore * 100),
        breakdown: {
          proximityKm: prox.distanceKm !== null ? Number(prox.distanceKm.toFixed(2)) : null,
          trajectoryAvgDistKm:
            traj.avgDistKm !== null ? Number(traj.avgDistKm.toFixed(2)) : null,
          anomalyFlags: anomaly.flags,
        },
      });
    }

    candidates.sort((a, b) => b.confidence - a.confidence);
    const limit = top_k && Number.isInteger(top_k) ? top_k : 3;

    res.json({
      spill: { lat: spill_lat, lon: spill_lon, time: spill_time },
      candidateCount: candidates.length,
      topVessels: candidates.slice(0, limit),
    });
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: "Internal error scoring vessels" });
  }
});

app.get("/health", (req, res) => {
  res.json({ status: "ok", dataLoaded, vesselCount: Object.keys(vesselTracks).length });
});

// ----------------------------------------------------------------------
// Startup
// ----------------------------------------------------------------------
loadAisData(AIS_CSV_PATH).then((tracks) => {
  vesselTracks = tracks;
  dataLoaded = true;
  console.log(
    `Loaded AIS tracks for ${Object.keys(vesselTracks).length} vessels from ${AIS_CSV_PATH}`
  );
  app.listen(PORT, () => {
    console.log(`Attribution service running on http://localhost:${PORT}`);
  });
});

module.exports = app;
