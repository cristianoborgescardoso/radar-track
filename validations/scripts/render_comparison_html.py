"""
Interactive Leaflet HTML Map Generator for Validation Comparison.

Responsibility: Render a standalone, interactive Leaflet.js HTML map
comparing Our Tracker T1 against Fortem R20 #919 and RTK GPS Ground Truth,
built on the clean, reliable design architecture of interactive_radar_map.html:
- Working Esri World Imagery (Satellite), CartoDB Dark, and OSM layers
- Leaflet Layer Switcher Control (base maps + overlays)
- Clean dark theme top toolbar with play/pause, time slider, speed slider
- Instantaneous Telemetry HUD table with color indicators and metrics
- Toggleable rejected clutter echo markers
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from geo_metrics import parse_ts


def generate_comparison_html(
    track_t1: Dict[str, Any],
    t1_metrics: Dict[str, Any],
    t1_errors: List[float],
    r20_points: List[Dict[str, Any]],
    r20_metrics: Dict[str, Any],
    r20_errors: List[float],
    gps_points: List[Dict[str, Any]],
    clutter_plots: Optional[List[Dict[str, Any]]],
    t0_s: float,
    max_sec: int,
    output_html: Path,
) -> None:
    """Create the interactive playable comparison HTML map matching interactive_radar_map.html structure."""
    t1_js_pts = []
    for p, err in zip(track_t1["points"], t1_errors):
        tp = parse_ts(p["timestamp"]) - t0_s
        t1_js_pts.append(
            {
                "plot_id": int(p["plot_id"]),
                "rel_t": round(tp, 2),
                "latitude": round(float(p["latitude"]), 7),
                "longitude": round(float(p["longitude"]), 7),
                "altitude": round(float(p["altitude"]), 1),
                "raw_latitude": round(float(p.get("raw_latitude", p["latitude"])), 7),
                "raw_longitude": round(float(p.get("raw_longitude", p["longitude"])), 7),
                "raw_altitude": round(float(p.get("raw_altitude", p["altitude"])), 1),
                "speed_mps": p["speed_mps"],
                "heading_deg": p["heading_deg"],
                "certainty": p.get("certainty", 100.0),
                "confidence": p.get("confidence", 1.0),
                "err_gps_m": round(float(err), 2) if err is not None else None,
            }
        )

    r20_js_pts = []
    for idx, (p, err) in enumerate(zip(r20_points, r20_errors), start=1):
        tp = parse_ts(p.get("epoch_s", p["timestamp"])) - t0_s
        r20_js_pts.append(
            {
                "step": idx,
                "rel_t": round(tp, 2),
                "latitude": round(float(p["latitude"]), 7),
                "longitude": round(float(p["longitude"]), 7),
                "altitude": round(float(p["altitude"]), 1),
                "speed_mps": p["speed_mps"],
                "heading_deg": p["heading_deg"],
                "err_gps_m": round(float(err), 2) if err is not None else None,
            }
        )

    gps_js_pts = []
    for idx, p in enumerate(gps_points, start=1):
        tp = parse_ts(p.get("epoch_s", p["timestamp"])) - t0_s
        gps_js_pts.append(
            {
                "seq": int(p.get("sequence", idx)),
                "rel_t": round(tp, 2),
                "latitude": round(float(p["latitude"]), 7),
                "longitude": round(float(p["longitude"]), 7),
                "altitude": round(float(p["altitude_gps"]), 1),
                "speed_mps": p["speed_mps"],
                "heading_deg": p["heading_deg"],
            }
        )

    clutter_js_pts = []
    if clutter_plots:
        for p in clutter_plots:
            tp = parse_ts(p["timestamp"]) - t0_s
            clutter_js_pts.append(
                {
                    "plot_id": int(p["id"]),
                    "rel_t": round(tp, 2),
                    "latitude": round(float(p["latitude"]), 7),
                    "longitude": round(float(p["longitude"]), 7),
                    "altitude": round(float(p["altitude"]), 1),
                }
            )

    center_lat = float(np.mean([p["latitude"] for p in gps_js_pts]))
    center_lon = float(np.mean([p["longitude"] for p in gps_js_pts]))
    gps_valid_spds = [p["speed_mps"] for p in gps_js_pts if p["speed_mps"] is not None]
    gps_mean_spd = float(np.mean(gps_valid_spds)) if gps_valid_spds else 7.55
    num_clutter = len(clutter_js_pts)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>AERPAW / Fortem TrueView R20 Validation Corridor — Interactive Map</title>
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
  <style>
    body {{ margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #0f172a; color: #f8fafc; }}
    #map {{ height: calc(100vh - 74px); width: 100vw; }}
    .toolbar {{
      height: 74px; display: flex; align-items: center; justify-content: space-between;
      padding: 0 20px; background: #1e293b; border-bottom: 1px solid #334155; box-sizing: border-box;
    }}
    .title-box h1 {{ margin: 0; font-size: 15.5px; color: #38bdf8; }}
    .title-box p {{ margin: 2px 0 0 0; font-size: 11.5px; color: #94a3b8; }}
    .controls {{ display: flex; align-items: center; gap: 12px; }}
    .speed-control {{ display: flex; align-items: center; gap: 6px; font-size: 12px; color: #94a3b8; font-weight: 500; }}
    .speed-control input[type=range] {{ width: 80px; accent-color: #38bdf8; }}
    .speed-control span {{ font-weight: 700; color: #38bdf8; min-width: 28px; text-align: left; }}
    button {{
      background: #0284c7; color: white; border: none; padding: 7px 13px;
      border-radius: 6px; cursor: pointer; font-weight: 600; font-size: 12.5px;
    }}
    button:hover {{ background: #0369a1; }}
    .btn-secondary {{ background: #475569; }}
    .btn-secondary:hover {{ background: #334155; }}
    input[type=range] {{ width: 220px; cursor: pointer; accent-color: #38bdf8; }}
    .legend {{
      background: rgba(15, 23, 42, 0.92); padding: 10px 14px; border-radius: 8px;
      border: 1px solid #334155; color: #e2e8f0; font-size: 12px; line-height: 1.6;
      box-shadow: 0 4px 14px rgba(0,0,0,0.45);
    }}
    .hud-box {{
      background: rgba(15, 23, 42, 0.94); padding: 12px 16px; border-radius: 8px;
      border: 1px solid #334155; color: #f8fafc; font-size: 12px; min-width: 480px;
      box-shadow: 0 4px 14px rgba(0,0,0,0.5); backdrop-filter: blur(4px);
    }}
    .hud-box h3 {{ margin: 0 0 8px 0; font-size: 13px; color: #38bdf8; border-bottom: 1px solid #334155; padding-bottom: 5px; }}
    .hud-table {{ width: 100%; border-collapse: collapse; font-variant-numeric: tabular-nums; }}
    .hud-table th {{ text-align: left; color: #94a3b8; font-weight: 600; padding: 4px 8px; border-bottom: 1px solid #334155; font-size: 11.5px; }}
    .hud-table td {{ padding: 4px 8px; border-bottom: 1px solid #1e293b; font-size: 12px; }}
    .swatch {{ display: inline-block; width: 10px; height: 10px; border-radius: 2px; vertical-align: middle; margin-right: 6px; }}
    .dot {{ display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 6px; }}
  </style>
</head>
<body>
  <div class="toolbar">
    <div class="title-box">
      <h1>AERPAW / Fortem TrueView R20 Validation Corridor (Opt2 Flight)</h1>
      <p>Our Tracker (T1) vs. Fortem R20 (#919) vs. RTK GPS Ground Truth &bull; {num_clutter} Rejected Clutter Echoes</p>
    </div>
    <div class="controls">
      <button id="playBtn" onclick="togglePlay()">&#9654; Play Comparison</button>
      <button class="btn-secondary" onclick="resetFull()">Show All</button>
      <input type="range" id="timeSlider" min="0" max="{max_sec}" step="0.5" value="{max_sec}" oninput="onSlider(this.value)" />
      <span id="timeLabel" style="min-width: 90px; font-variant-numeric: tabular-nums; font-size: 13px;">t = {max_sec}.0 s</span>
      <div class="speed-control">
        <label for="speedSlider">Speed:</label>
        <input type="range" id="speedSlider" min="1" max="10" step="1" value="1" oninput="onSpeedChange(this.value)" />
        <span id="speedLabel">1x</span>
      </div>
    </div>
  </div>

  <div id="map"></div>

  <script>
    const t1Pts = {json.dumps(t1_js_pts)};
    const r20Pts = {json.dumps(r20_js_pts)};
    const gpsPts = {json.dumps(gps_js_pts)};
    const clutterPts = {json.dumps(clutter_js_pts)};
    const maxSec = {max_sec};

    const streetLayer = L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
      maxZoom: 19, attribution: '&copy; OpenStreetMap'
    }});
    const satelliteLayer = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{{z}}/{{y}}/{{x}}', {{
      maxZoom: 19, attribution: '&copy; Esri World Imagery (Satellite)'
    }});
    const darkRadarLayer = L.tileLayer('https://{{s}}.basemaps.cartocdn.com/dark_all/{{z}}/{{x}}/{{y}}{{r}}.png', {{
      maxZoom: 19, attribution: '&copy; CartoDB Dark Matter'
    }});

    const map = L.map('map', {{
      center: [{center_lat}, {center_lon}],
      zoom: 17,
      layers: [satelliteLayer]
    }});

    const baseMaps = {{
      "Satellite (Esri World Imagery)": satelliteLayer,
      "Dark Radar Scope (Tactical)": darkRadarLayer,
      "Street Map (OpenStreetMap)": streetLayer
    }};

    const dynamicLayer = L.layerGroup().addTo(map);
    const clutterLayer = L.layerGroup().addTo(map);

    L.control.layers(baseMaps, {{
      "Reconstructed Trajectories": dynamicLayer,
      "Rejected Clutter Echoes": clutterLayer
    }}).addTo(map);

    // Static radar tripod marker
    L.circleMarker([35.724976, -78.691933], {{
      radius: 9, color: '#f59e0b', fillColor: '#fbbf24', fillOpacity: 0.95, weight: 2
    }}).addTo(map).bindPopup('<b>Fortem TrueView R20 Radar Tripod</b><br/>Lake Wheeler Field Lab<br/>Lat: 35.724976°, Lon: -78.691933°<br/>Elevation: 10.0 m MSL<br/>Yaw Alignment: +2.0° CCW ENU');

    const legend = L.control({{ position: 'bottomright' }});
    legend.onAdd = function() {{
      const div = L.DomUtil.create('div', 'legend');
      div.innerHTML = '<b>Track Comparison (Opt2)</b><br>' +
        '<span class="dot" style="background:#38bdf8"></span><b>Our Tracker T1</b> (RMSE: {t1_metrics["rmse_xy_m"]:.2f} m)<br>' +
        '<span class="dot" style="background:#c084fc"></span><b>Fortem R20 #919</b> (RMSE: {r20_metrics["rmse_xy_m"]:.2f} m)<br>' +
        '<span class="dot" style="background:#f8fafc"></span><b>UAV RTK GPS</b> (Ground Truth)<br>' +
        '<span class="dot" style="background:#ef4444"></span><b>Rejected Clutter</b> ({num_clutter} echoes)';
      return div;
    }};
    legend.addTo(map);

    const hud = L.control({{ position: 'topright' }});
    hud.onAdd = function() {{
      const div = L.DomUtil.create('div', 'hud-box');
      div.id = 'telemetryHud';
      div.innerHTML = '<h3>Instantaneous Radar Telemetry</h3><p style="color:#94a3b8;margin:0;">Play comparison or drag time slider to inspect live target kinematics.</p>';
      return div;
    }};
    hud.addTo(map);

    function findClosest(pts, t) {{
      let best = pts[0];
      let minDt = Math.abs(pts[0].rel_t - t);
      for (let i = 1; i < pts.length; i++) {{
        const dt = Math.abs(pts[i].rel_t - t);
        if (dt < minDt) {{ minDt = dt; best = pts[i]; }}
      }}
      return {{ item: best, dt: minDt }};
    }}

    function renderAtTime(maxRelSec) {{
      dynamicLayer.clearLayers();
      clutterLayer.clearLayers();
      document.getElementById('timeLabel').innerText = 't = ' + Number(maxRelSec).toFixed(1) + ' s';

      // 1. Clutter echoes up to current time (active window fade)
      clutterPts.forEach(c => {{
        const dt = maxRelSec - c.rel_t;
        if (dt >= -0.2 && dt <= 1.2) {{
          const fade = Math.max(0.2, 1.0 - (dt / 1.2));
          L.circleMarker([c.latitude, c.longitude], {{
            radius: 4, color: '#dc2626', fillColor: '#ef4444', fillOpacity: fade * 0.85, opacity: fade, weight: 1.5
          }}).bindPopup(`<b>Rejected Clutter Echo</b><br>Plot ID: ${{c.plot_id}}<br>Alt: ${{c.altitude.toFixed(1)}} m<br>t: +${{c.rel_t.toFixed(2)}} s`).addTo(clutterLayer);
        }}
      }});

      // 2. Visible points
      const visGps = gpsPts.filter(p => p.rel_t <= maxRelSec);
      const visR20 = r20Pts.filter(p => p.rel_t <= maxRelSec);
      const visT1  = t1Pts.filter(p => p.rel_t <= maxRelSec);

      // Full trail polylines
      if (visGps.length > 1) {{
        L.polyline(visGps.map(p => [p.latitude, p.longitude]), {{ color: '#f8fafc', weight: 4, opacity: 0.85, dashArray: '2, 4' }}).addTo(dynamicLayer);
      }}
      if (visR20.length > 1) {{
        L.polyline(visR20.map(p => [p.latitude, p.longitude]), {{ color: '#a855f7', weight: 3, opacity: 0.8, dashArray: '6, 6' }}).addTo(dynamicLayer);
      }}
      if (visT1.length > 1) {{
        L.polyline(visT1.map(p => [p.latitude, p.longitude]), {{ color: '#0284c7', weight: 3.5, opacity: 0.95 }}).addTo(dynamicLayer);
      }}

      // Current head indicators
      if (visGps.length > 0) {{
        const g = visGps[visGps.length - 1];
        L.circleMarker([g.latitude, g.longitude], {{ radius: 8, color: '#000000', fillColor: '#f8fafc', fillOpacity: 1, weight: 2.5 }})
          .bindPopup(`<b>UAV RTK GPS (Ground Truth)</b><br>Time: +${{g.rel_t.toFixed(2)}} s<br>Speed: ${{g.speed_mps !== null ? g.speed_mps.toFixed(2) + ' m/s' : 'N/A'}}<br>Alt: ${{g.altitude.toFixed(1)}} m`).addTo(dynamicLayer);
      }}
      if (visR20.length > 0) {{
        const r = visR20[visR20.length - 1];
        L.circleMarker([r.latitude, r.longitude], {{ radius: 7, color: '#6b21a8', fillColor: '#c084fc', fillOpacity: 0.95, weight: 2 }})
          .bindPopup(`<b>Fortem R20 Onboard (#919)</b><br>Time: +${{r.rel_t.toFixed(2)}} s<br>Error vs GPS: ${{r.err_gps_m !== null ? r.err_gps_m.toFixed(2) + ' m' : 'N/A'}}<br>Speed: ${{r.speed_mps !== null ? r.speed_mps.toFixed(2) + ' m/s' : 'N/A'}}`).addTo(dynamicLayer);
      }}
      if (visT1.length > 0) {{
        const t = visT1[visT1.length - 1];
        L.circleMarker([t.latitude, t.longitude], {{ radius: 7, color: '#0369a1', fillColor: '#38bdf8', fillOpacity: 0.95, weight: 2 }})
          .bindPopup(`<b>Our Tracker (T1)</b><br>Time: +${{t.rel_t.toFixed(2)}} s<br>Error vs GPS: ${{t.err_gps_m !== null ? t.err_gps_m.toFixed(2) + ' m' : 'N/A'}}<br>Certainty: ${{t.certainty.toFixed(0)}}%<br>Speed: ${{t.speed_mps !== null ? t.speed_mps.toFixed(2) + ' m/s' : 'N/A'}}`).addTo(dynamicLayer);
      }}

      // Telemetry table row updates
      const curT1 = findClosest(t1Pts, maxRelSec);
      const curR20 = findClosest(r20Pts, maxRelSec);
      const curGps = findClosest(gpsPts, maxRelSec);

      const t1ErrStr = (curT1.item && curT1.dt <= 1.5 && curT1.item.err_gps_m !== null) ? `${{curT1.item.err_gps_m.toFixed(2)}} m` : '--';
      const t1SpdStr = (curT1.item && curT1.dt <= 1.5 && curT1.item.speed_mps !== null) ? `${{curT1.item.speed_mps.toFixed(2)}} m/s` : '--';
      const t1CertStr = (curT1.item && curT1.dt <= 1.5) ? `${{curT1.item.certainty.toFixed(0)}}%` : '--';
      const t1Status = (curT1.item && curT1.dt <= 1.5) ? '<span style="color:#22c55e;font-weight:600;">Active</span>' : '<span style="color:#f59e0b;font-weight:600;">Coast</span>';

      const r20ErrStr = (curR20.item && curR20.dt <= 1.5 && curR20.item.err_gps_m !== null) ? `${{curR20.item.err_gps_m.toFixed(2)}} m` : '--';
      const r20SpdStr = (curR20.item && curR20.dt <= 1.5 && curR20.item.speed_mps !== null) ? `${{curR20.item.speed_mps.toFixed(2)}} m/s` : '--';

      const gpsSpdStr = (curGps.item && curGps.item.speed_mps !== null) ? `${{curGps.item.speed_mps.toFixed(2)}} m/s` : '--';
      const gpsAltStr = curGps.item ? `${{curGps.item.altitude.toFixed(1)}} m` : '--';

      const hudEl = document.getElementById('telemetryHud');
      if (hudEl) {{
        hudEl.innerHTML =
          `<h3>Instantaneous Validation Telemetry &bull; t = ${{Number(maxRelSec).toFixed(1)}} s</h3>` +
          `<table class="hud-table">` +
          `<thead><tr><th>Stream</th><th>Pos Error (vs GPS)</th><th>Speed</th><th>Altitude / Cert</th><th>Status</th></tr></thead>` +
          `<tbody>` +
          `<tr><td><span class="swatch" style="background:#0284c7;"></span><b>Our Tracker T1</b></td><td><b>${{t1ErrStr}}</b></td><td>${{t1SpdStr}}</td><td>${{t1CertStr}}</td><td>${{t1Status}}</td></tr>` +
          `<tr><td><span class="swatch" style="background:#a855f7;"></span><b>Fortem R20 #919</b></td><td><b>${{r20ErrStr}}</b></td><td>${{r20SpdStr}}</td><td>--</td><td>--</td></tr>` +
          `<tr><td><span class="swatch" style="background:#f8fafc;"></span><b>UAV RTK GPS</b></td><td>0.00 m (Ref)</td><td>${{gpsSpdStr}}</td><td>${{gpsAltStr}}</td><td><span style="color:#22c55e;font-weight:600;">Ground Truth</span></td></tr>` +
          `</tbody>` +
          `</table>`;
      }}
    }}

    let timer = null;
    let playbackSpeed = 1.0;

    function onSpeedChange(val) {{
      playbackSpeed = Number(val);
      document.getElementById('speedLabel').innerText = playbackSpeed + 'x';
      if (timer) {{
        clearInterval(timer);
        startPlaybackLoop();
      }}
    }}

    function startPlaybackLoop() {{
      const slider = document.getElementById('timeSlider');
      const step = 0.5 * playbackSpeed;
      const intervalMs = 500;
      timer = setInterval(() => {{
        let v = Number(slider.value) + step;
        if (v > maxSec) {{
          v = maxSec;
          slider.value = v;
          renderAtTime(v);
          clearInterval(timer);
          timer = null;
          document.getElementById('playBtn').innerHTML = '&#9654; Play Comparison';
          return;
        }}
        slider.value = v;
        renderAtTime(v);
      }}, intervalMs);
    }}

    function togglePlay() {{
      const slider = document.getElementById('timeSlider');
      const btn = document.getElementById('playBtn');
      if (timer) {{
        clearInterval(timer);
        timer = null;
        btn.innerHTML = '&#9654; Play Comparison';
      }} else {{
        if (Number(slider.value) >= maxSec - 0.5) slider.value = 0;
        btn.innerHTML = '&#9208; Pause';
        startPlaybackLoop();
      }}
    }}

    function onSlider(val) {{ renderAtTime(Number(val)); }}
    function resetFull() {{
      if (timer) togglePlay();
      document.getElementById('timeSlider').value = maxSec;
      renderAtTime(maxSec);
    }}

    renderAtTime(maxSec);
  </script>
</body>
</html>"""
    output_html.parent.mkdir(parents=True, exist_ok=True)
    output_html.write_text(html, encoding="utf-8")
