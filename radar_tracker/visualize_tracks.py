"""Render diagnostic overviews of reconstructed radar tracks, certainty lifecycle, and interactive HTML map."""

import argparse
import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import pymap3d as pm

from radar_tracker.config import TrackerConfig
from radar_tracker.preprocess import prepare_plots_dataframe


def _parse_iso_ts(ts: Any) -> float:
    if isinstance(ts, (int, float)):
        return float(ts)
    return datetime.fromisoformat(str(ts).replace("Z", "+00:00")).timestamp()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as input_file:
        return json.load(input_file)


def create_visualization(
    plots_path: Path,
    tracks_path: Path,
    output_path: Path,
    raw_plots: Optional[List[Dict[str, Any]]] = None,
    tracks: Optional[List[Dict[str, Any]]] = None,
) -> None:
    """Render 4-panel overview: Horizontal tracks, Altitude over time (with clutter), Speed, and Heading."""
    if raw_plots is None:
        raw_plots = load_json(plots_path)
    if tracks is None:
        tracks = load_json(tracks_path)
    if not raw_plots:
        raise ValueError("The input plots file is empty.")

    dataframe, origin = prepare_plots_dataframe(raw_plots)
    t0_s = float(dataframe["t"].min())
    dataframe["t_rel"] = dataframe["t"] - t0_s
    plot_by_id = dataframe.set_index("id")
    assigned_plot_ids = {
        int(plot_id) for track in tracks for plot_id in track["plot_ids"]
    }
    unassigned = dataframe[~dataframe["id"].isin(assigned_plot_ids)]

    figure, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)
    horizontal_axis, altitude_axis, speed_axis, heading_axis = axes.flat
    colors = plt.get_cmap("tab10")

    horizontal_axis.scatter(
        unassigned["x"],
        unassigned["y"],
        marker="x",
        color="0.65",
        alpha=0.55,
        s=22,
        label=f"Unassigned plots ({len(unassigned)})",
    )

    altitude_axis.scatter(
        unassigned["t_rel"],
        unassigned["altitude"],
        marker="x",
        color="0.65",
        alpha=0.55,
        s=22,
        label=f"Unassigned plots ({len(unassigned)})",
    )

    for track_index, track in enumerate(tracks):
        color = colors(track_index % colors.N)
        points = track["points"]
        point_ids = [int(point["plot_id"]) for point in points]
        track_frame = plot_by_id.loc[point_ids]
        times = track_frame["t_rel"].to_numpy()
        label = track["track_id"]
        
        certs = [p.get("certainty", 100.0) for p in points if "certainty" in p]
        mean_cert = float(np.mean(certs)) if certs else float(track.get("certainty", 100.0))
        track_conf = float(track.get("confidence", 1.0))
        label_with_metrics = f"{label} (Cert: {mean_cert:.1f} pts, Conf: {track_conf:.3f})"

        est_lats = np.array([float(p["latitude"]) for p in points], dtype=float)
        est_lons = np.array([float(p["longitude"]) for p in points], dtype=float)
        est_alts = np.array([float(p["altitude"]) for p in points], dtype=float)
        est_x, est_y, _ = pm.geodetic2enu(est_lats, est_lons, est_alts, *origin)
        est_x = np.asarray(est_x, dtype=float)
        est_y = np.asarray(est_y, dtype=float)

        raw_x = track_frame["x"].to_numpy(dtype=float)
        raw_y = track_frame["y"].to_numpy(dtype=float)
        raw_alt = track_frame["altitude"].to_numpy(dtype=float)
        is_smoothed = float(np.max(np.hypot(est_x - raw_x, est_y - raw_y))) > 0.5

        if is_smoothed:
            horizontal_axis.scatter(
                raw_x,
                raw_y,
                color=[color],
                alpha=0.30,
                s=16,
                label=f"{label} raw plots ({len(points)})",
            )
            altitude_axis.scatter(
                times,
                raw_alt,
                color=[color],
                alpha=0.30,
                s=16,
                label=f"{label} raw altitude",
            )

        horizontal_axis.plot(
            est_x,
            est_y,
            color=color,
            linewidth=2.0 if is_smoothed else 1.5,
            marker="." if not is_smoothed else None,
            markersize=4,
            label=f"{label_with_metrics} [{len(points)} pts]",
        )
        horizontal_axis.scatter(
            est_x[0],
            est_y[0],
            color=[color],
            marker="o",
            edgecolor="black",
            linewidth=0.6,
            s=55,
            zorder=3,
        )
        horizontal_axis.scatter(
            est_x[-1],
            est_y[-1],
            color=[color],
            marker="s",
            edgecolor="black",
            linewidth=0.6,
            s=45,
            zorder=3,
        )

        altitude_axis.plot(
            times,
            est_alts,
            color=color,
            linewidth=2.0 if is_smoothed else 1.5,
            marker="." if not is_smoothed else None,
            markersize=3.5,
            label=f"{label} ({len(points)} plots)",
        )

        speeds = [point["speed_mps"] for point in points]
        speed_axis.plot(
            times,
            speeds,
            color=color,
            linewidth=1.5,
            marker=".",
            markersize=3,
            label=label,
        )

        headings = [point["heading_deg"] for point in points]
        heading_axis.plot(
            times,
            headings,
            color=color,
            linewidth=1.5,
            marker=".",
            markersize=3,
            label=label,
        )

    horizontal_axis.set_title("Horizontal tracks (East vs. North)")
    horizontal_axis.set_xlabel("East (m)")
    horizontal_axis.set_ylabel("North (m)")
    horizontal_axis.set_aspect("equal", adjustable="datalim")
    horizontal_axis.grid(True, alpha=0.25)
    horizontal_axis.legend(loc="best", fontsize="small")

    altitude_axis.set_title("Altitude over time (Confirmed Tracks vs. Unassigned Clutter)")
    altitude_axis.set_xlabel("Elapsed time (s)")
    altitude_axis.set_ylabel("Altitude (m)")
    altitude_axis.grid(True, alpha=0.25)
    altitude_axis.legend(loc="best", fontsize="small")

    speed_axis.set_title("Estimated horizontal ground speed")
    speed_axis.set_xlabel("Elapsed time (s)")
    speed_axis.set_ylabel("Speed (m/s)")
    speed_axis.grid(True, alpha=0.25)
    speed_axis.legend(loc="best", fontsize="small")

    heading_axis.set_title("Estimated ground heading")
    heading_axis.set_xlabel("Elapsed time (s)")
    heading_axis.set_ylabel("Heading (degrees; 0° = North)")
    heading_axis.set_ylim(0, 360)
    heading_axis.set_yticks([0, 90, 180, 270, 360])
    heading_axis.grid(True, alpha=0.25)
    heading_axis.legend(loc="best", fontsize="small")

    figure.suptitle(
        f"Radar track reconstruction: {len(tracks)} confirmed tracks, "
        f"{len(unassigned)} unassigned plots",
        fontsize=15,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def create_certainty_visualization(
    confirmed_tracks: List[Any],
    rejected_tracks: List[Any],
    config: TrackerConfig,
    output_path: Path,
    t0: Optional[float] = None,
) -> None:
    """Render Track Certainty Score (0-100) evolution for confirmed tracks and rejected clutter."""
    all_tr = list(confirmed_tracks) + list(rejected_tracks)
    if t0 is None:
        t0 = min((tr.plots[0]["t"] for tr in all_tr if tr.plots), default=0.0)

    fig, ax = plt.subplots(figsize=(12, 6), dpi=150)
    colors = ["#2563eb", "#ea580c", "#16a34a", "#dc2626", "#7c3aed", "#0891b2"]

    for idx, tr in enumerate(confirmed_tracks):
        t_rel = [p["t"] - t0 for p in tr.plots]
        cert_hist = tr.certainty_history[: len(t_rel)]
        mean_cert = float(np.mean(cert_hist)) if cert_hist else float(tr.certainty)
        col = colors[idx % len(colors)]
        ax.plot(
            t_rel,
            cert_hist,
            color=col,
            lw=2.2,
            marker="o",
            ms=3.5,
            label=f"T{idx + 1} Confirmed ({len(tr.plots)} hits | Mean: {mean_cert:.1f} pts, Final: {tr.certainty:.0f} pts)",
        )

    clutter_labeled = False
    for tr in rejected_tracks:
        t_scan = tr.plots[0]["t"] - t0
        n_decay_steps = int(
            max(1, round(config.initial_certainty / max(1.0, config.certainty_miss_penalty)))
        )
        dt_step = config.nominal_scan_period_s
        ts_c = [t_scan + k * dt_step for k in range(n_decay_steps + 1)]
        cs_c = [
            max(0.0, config.initial_certainty - k * config.certainty_miss_penalty)
            for k in range(n_decay_steps + 1)
        ]
        ax.plot(
            ts_c,
            cs_c,
            color="#94a3b8",
            lw=1.0,
            ls="--",
            alpha=0.45,
            label=f"Rejected Clutter ({len(rejected_tracks)} false tracks)"
            if not clutter_labeled
            else None,
        )
        clutter_labeled = True

    ax.axhline(
        config.confirmation_certainty_threshold,
        color="#ef4444",
        ls=":",
        lw=2.0,
        label=f"Confirmation Threshold ({config.confirmation_certainty_threshold:.0f} pts)",
    )
    ax.axhline(
        config.initial_certainty,
        color="#64748b",
        ls="-.",
        lw=1.3,
        label=f"Initial Certainty ({config.initial_certainty:.0f} pts)",
    )
    ax.set_title(
        f"Track Certainty Lifecycle (0–100 Scale: +{config.certainty_hit_increment:.0f}/hit, -{config.certainty_miss_penalty:.0f}/miss)",
        fontsize=13,
        fontweight="bold",
    )
    ax.set_xlabel("Elapsed Time (s)")
    ax.set_ylabel("Certainty Score (0–100)")
    ax.set_ylim(-5, 105)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="center right", fontsize=9.5)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def create_interactive_html_map(
    raw_plots: list[dict],
    tracks: list[dict],
    output_html_path: Path,
) -> None:
    """Generate a standalone interactive Leaflet HTML map (Satellite, Dark Radar Scope, Street Map)."""
    assigned_ids = {int(pid) for tr in tracks for pid in tr.get("plot_ids", [])}
    clutter_data = [
        {
            "id": int(p["id"]),
            "latitude": float(p["latitude"]),
            "longitude": float(p["longitude"]),
            "altitude": float(p["altitude"]),
            "timestamp": p["timestamp"],
        }
        for p in raw_plots
        if int(p["id"]) not in assigned_ids
    ]

    all_lats = [float(p["latitude"]) for p in raw_plots]
    all_lons = [float(p["longitude"]) for p in raw_plots]
    center_lat = sum(all_lats) / len(all_lats) if all_lats else 0.0
    center_lon = sum(all_lons) / len(all_lons) if all_lons else 0.0

    all_ts = [_parse_iso_ts(p["timestamp"]) for p in raw_plots]
    t0_s = min(all_ts) if all_ts else 0.0
    max_sec = int(math.ceil(max(all_ts) - t0_s)) if all_ts else 100
    total_assigned = sum(len(tr.get("plot_ids", [])) for tr in tracks)

    palette = ["#3b82f6", "#f97316", "#22c55e", "#ef4444", "#a855f7", "#06b6d4", "#eab308"]
    colors_map = {
        tr["track_id"]: palette[idx % len(palette)] for idx, tr in enumerate(tracks)
    }
    legend_rows_list = []
    for tr in tracks:
        pts = tr.get("points", [])
        certs = [p.get("certainty", 100.0) for p in pts if "certainty" in p]
        mean_c = float(np.mean(certs)) if certs else float(tr.get("certainty", 100.0))
        conf_val = float(tr.get("confidence", 1.0))
        legend_rows_list.append(
            f'<span class="dot" style="background:{colors_map[tr["track_id"]]}"></span>'
            f'<b>{tr["track_id"]}</b> (avg: {tr.get("average_speed_mps", 0.0):.2f} m/s, '
            f'hdg: {tr.get("average_heading_deg", 0.0):.1f}° | '
            f'Cert Média: {mean_c:.1f} pts | Conf: {conf_val:.3f})<br>'
        )
    legend_rows = "".join(legend_rows_list)
    legend_rows += (
        f'<span class="dot" style="background:#94a3b8"></span>'
        f"Rejected Clutter ({len(clutter_data)} plots)"
    )

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Radar Track Reconstruction — Interactive Map (Google Maps / Satellite Style)</title>
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
    .title-box h1 {{ margin: 0; font-size: 16px; color: #38bdf8; }}
    .title-box p {{ margin: 2px 0 0 0; font-size: 12px; color: #94a3b8; }}
    .controls {{ display: flex; align-items: center; gap: 12px; }}
    .speed-control {{ display: flex; align-items: center; gap: 6px; font-size: 12px; color: #94a3b8; font-weight: 500; }}
    .speed-control input[type=range] {{ width: 80px; }}
    .speed-control span {{ font-weight: 700; color: #38bdf8; min-width: 28px; text-align: left; }}
    button {{
      background: #0284c7; color: white; border: none; padding: 8px 14px;
      border-radius: 6px; cursor: pointer; font-weight: 600; font-size: 13px;
    }}
    button:hover {{ background: #0369a1; }}
    input[type=range] {{ width: 260px; cursor: pointer; }}
    .legend {{
      background: rgba(15, 23, 42, 0.9); padding: 10px 14px; border-radius: 8px;
      border: 1px solid #334155; color: #e2e8f0; font-size: 12px; line-height: 1.6;
    }}
    .hud-box {{
      background: rgba(15, 23, 42, 0.94); padding: 12px 16px; border-radius: 8px;
      border: 1px solid #334155; color: #f8fafc; font-size: 12px; min-width: 520px;
      box-shadow: 0 4px 14px rgba(0,0,0,0.45);
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
      <h1>Radar Multi-Target Track Reconstruction — Interactive Map</h1>
      <p>{len(tracks)} Confirmed Tracks ({total_assigned} plots) &bull; {len(clutter_data)} Rejected False Alarms (Clutter) &bull; Click any point to inspect kinematics</p>
    </div>
    <div class="controls">
      <button id="playBtn" onclick="togglePlay()">&#9654; Play Sweep</button>
      <button onclick="resetFull()">Show All</button>
      <input type="range" id="timeSlider" min="0" max="{max_sec}" step="1" value="{max_sec}" oninput="onSlider(this.value)" />
      <span id="timeLabel" style="min-width: 95px; font-variant-numeric: tabular-nums; font-size: 13px;">t = {max_sec:.1f} s</span>
      <div class="speed-control">
        <label for="speedSlider">Speed:</label>
        <input type="range" id="speedSlider" min="1" max="10" step="1" value="1" oninput="onSpeedChange(this.value)" />
        <span id="speedLabel">1x</span>
      </div>
    </div>
  </div>
  <div id="map"></div>

  <script>
    const tracksData = {json.dumps(tracks)};
    const clutterData = {json.dumps(clutter_data)};
    const colors = {json.dumps(colors_map)};
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
      center: [{center_lat:.6f}, {center_lon:.6f}],
      zoom: 14,
      layers: [satelliteLayer]
    }});

    const baseMaps = {{
      "Satellite (Esri World Imagery)": satelliteLayer,
      "Dark Radar Scope (Tactical)": darkRadarLayer,
      "Street Map (OpenStreetMap)": streetLayer
    }};

    const t0 = {t0_s:.6f};
    const dynamicLayer = L.layerGroup().addTo(map);
    L.control.layers(baseMaps, {{"Reconstructed Tracks & Plots": dynamicLayer}}).addTo(map);

    const legend = L.control({{position: 'bottomright'}});
    legend.onAdd = function() {{
      const div = L.DomUtil.create('div', 'legend');
      div.innerHTML = '<b>Reconstructed Targets</b><br>' + {json.dumps(legend_rows)};
      return div;
    }};
    legend.addTo(map);

    const hud = L.control({{position: 'topright'}});
    hud.onAdd = function() {{
      const div = L.DomUtil.create('div', 'hud-box');
      div.id = 'telemetryHud';
      div.innerHTML = '<h3>Instantaneous Radar Telemetry</h3><p style="color:#94a3b8;margin:0;">Play sweep or drag time slider to inspect live target kinematics.</p>';
      return div;
    }};
    hud.addTo(map);

    function epochSeconds(value) {{
      return typeof value === 'number' ? value : Date.parse(value) / 1000.0;
    }}

    function renderAtTime(maxRelSec) {{
      dynamicLayer.clearLayers();
      document.getElementById('timeLabel').innerText = 't = ' + Number(maxRelSec).toFixed(1) + ' s';

      clutterData.forEach(c => {{
        const tc = epochSeconds(c.timestamp) - t0;
        if (tc <= maxRelSec) {{
          L.circleMarker([c.latitude, c.longitude], {{
            radius: 3.5, color: '#94a3b8', fillColor: '#64748b', fillOpacity: 0.6, weight: 1
          }}).bindPopup(`<b>Rejected False Alarm (CLUTTER)</b><br>Plot ID: ${{c.id}}<br>Alt: ${{c.altitude.toFixed(1)}} m<br>t: +${{tc.toFixed(2)}} s`).addTo(dynamicLayer);
        }}
      }});

      let hudRows = '';
      tracksData.forEach(tr => {{
        const col = colors[tr.track_id] || '#38bdf8';
        const visiblePts = tr.points.filter(p => (epochSeconds(p.timestamp) - t0) <= maxRelSec);
        if (visiblePts.length > 1) {{
          const latlngs = visiblePts.map(p => [p.latitude, p.longitude]);
          L.polyline(latlngs, {{ color: col, weight: 3.5, opacity: 0.9 }}).addTo(dynamicLayer);
        }}
        visiblePts.forEach((p, idx) => {{
          const isHead = (idx === visiblePts.length - 1);
          const tp = epochSeconds(p.timestamp) - t0;
          L.circleMarker([p.latitude, p.longitude], {{
            radius: isHead ? 7 : 3,
            color: isHead ? '#ffffff' : col,
            fillColor: col,
            fillOpacity: isHead ? 1.0 : 0.8,
            weight: isHead ? 2 : 1
          }}).bindPopup(
            `<b>Track ${{tr.track_id}}</b> (Plot #${{p.plot_id}})<br>` +
            `Time: +${{tp.toFixed(2)}} s<br>` +
            `Speed: ${{p.speed_mps !== null ? p.speed_mps + ' m/s' : 'N/A (1st point)'}}<br>` +
            `Heading: ${{p.heading_deg !== null ? p.heading_deg + '°' : 'N/A'}}<br>` +
            `Altitude: ${{p.altitude.toFixed(1)}} m`
          ).addTo(dynamicLayer);
        }});

        if (visiblePts.length > 0) {{
          const cur = visiblePts[visiblePts.length - 1];
          const tp = epochSeconds(cur.timestamp) - t0;
          const dtLag = maxRelSec - tp;
          const isCoasting = dtLag >= 1.0;
          const statusBadge = isCoasting
            ? `<span style="color:#f59e0b;font-weight:600;font-size:11px;">Coast (+${{dtLag.toFixed(1)}}s)</span>`
            : `<span style="color:#22c55e;font-weight:600;font-size:11px;">Active</span>`;
          const spdText = (cur.speed_mps !== null && cur.speed_mps !== undefined)
            ? `${{Number(cur.speed_mps).toFixed(1)}} m/s <span style="color:#94a3b8;font-size:11px;">(${{(cur.speed_mps * 3.6).toFixed(0)}} km/h)</span>`
            : '—';
          const hdgText = (cur.heading_deg !== null && cur.heading_deg !== undefined)
            ? `${{Number(cur.heading_deg).toFixed(1)}}°`
            : '—';
          const altText = `${{Number(cur.altitude).toFixed(1)}} m`;
          // Time-projected certainty during coasting: decay by penalty (-7 pts) per elapsed scan period
          const missedSteps = isCoasting ? Math.floor(dtLag / 1.0) : 0;
          const liveCert = cur.certainty !== undefined ? Math.max(0, cur.certainty - missedSteps * 7.0) : undefined;
          const certText = liveCert !== undefined ? `${{Number(liveCert).toFixed(0)}} pts` : '—';
          const confText = cur.confidence !== undefined ? `${{Number(cur.confidence).toFixed(3)}}` : (tr.confidence !== undefined ? `${{Number(tr.confidence).toFixed(3)}}` : '—');
          hudRows += `<tr>
            <td><span class="swatch" style="background:${{col}};"></span><b>${{tr.track_id}}</b></td>
            <td><b>${{spdText}}</b></td>
            <td>${{hdgText}}</td>
            <td>${{altText}}</td>
            <td>${{certText}}</td>
            <td>${{confText}}</td>
            <td>${{statusBadge}}</td>
          </tr>`;
        }} else {{
          hudRows += `<tr>
            <td><span class="swatch" style="background:${{col}};"></span><b>${{tr.track_id}}</b></td>
            <td colspan="6" style="color:#64748b;font-style:italic;">Not yet acquired</td>
          </tr>`;
        }}
      }});

      const hudEl = document.getElementById('telemetryHud');
      if (hudEl) {{
        hudEl.innerHTML =
          `<h3>Instantaneous Radar Telemetry &bull; t = ${{Number(maxRelSec).toFixed(1)}} s</h3>` +
          `<table class="hud-table">` +
          `<thead><tr><th>Track</th><th>Speed</th><th>Heading</th><th>Altitude</th><th>Certainty</th><th>Confidence</th><th>Status</th></tr></thead>` +
          `<tbody>${{hudRows}}</tbody>` +
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
          document.getElementById('playBtn').innerHTML = '&#9654; Play Sweep';
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
        btn.innerHTML = '&#9654; Play Sweep';
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
</html>
"""
    output_html_path.parent.mkdir(parents=True, exist_ok=True)
    output_html_path.write_text(html_content, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a diagnostic PNG and interactive HTML map for reconstructed radar tracks."
    )
    parser.add_argument("plots_file", nargs="?", default="plots.json")
    parser.add_argument("tracks_file", nargs="?", default="tracks.json")
    parser.add_argument("output_file", nargs="?", default="tracks_visualization.png")
    parser.add_argument(
        "html_file",
        nargs="?",
        default=os.path.join("diagnostics", "interactive_radar_map.html"),
    )
    arguments = parser.parse_args()

    plots_path = Path(arguments.plots_file)
    tracks_path = Path(arguments.tracks_file)
    output_path = Path(arguments.output_file)
    html_path = Path(arguments.html_file)

    raw_plots = json.loads(plots_path.read_text(encoding="utf-8"))
    tracks = json.loads(tracks_path.read_text(encoding="utf-8"))

    create_visualization(
        plots_path,
        tracks_path,
        output_path,
        raw_plots=raw_plots,
        tracks=tracks,
    )
    create_interactive_html_map(raw_plots, tracks, html_path)
    print(f"  -> PNG Overview    : {output_path}")
    print(f"  -> Interactive HTML: {html_path}")


if __name__ == "__main__":
    main()
