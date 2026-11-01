"""Capture the existing Leaflet player's timeline as a README-friendly GIF.

Requires Pillow, playwright, and an installed Chrome browser.
"""
import argparse
import io
import json
import math
import hashlib
from pathlib import Path

from PIL import Image
from playwright.sync_api import sync_playwright


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--html", type=Path, default=Path("tracks_interactive_map.html"))
    parser.add_argument("--output", type=Path, default=Path("docs/tracking-demo.gif"))
    parser.add_argument("--zoom", type=float, default=1.10,
                        help="Map magnification relative to the fitted view (default: 1.10)")
    args = parser.parse_args()
    source_bytes = args.html.read_bytes()
    source_sha256 = hashlib.sha256(source_bytes).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frames = []
    errors = []
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="chrome", headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 900},
                                device_scale_factor=4 / 3)
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.route(args.html.resolve().as_uri(), lambda route: route.fulfill(
            body=source_bytes, content_type="text/html; charset=utf-8"))
        page.goto(args.html.resolve().as_uri(), wait_until="networkidle", timeout=60000)
        page.wait_for_function("typeof renderAtTime === 'function' && typeof L !== 'undefined'")
        # Keep every trajectory visible, leaving room for the existing telemetry panel.
        page.evaluate("""() => {
            const points = typeof tracksData !== 'undefined'
                ? tracksData.flatMap(t => t.points)
                : [...t1Pts, ...r20Pts, ...gpsPts];
            map.fitBounds(L.latLngBounds(points.map(p => [p.latitude, p.longitude])),
                {paddingTopLeft:[45,210], paddingBottomRight:[45,45], animate:false});
            onSpeedChange(8);
        }""")
        page.evaluate("""delta => {
            map.options.zoomSnap = 0;
            map.setZoom(map.getZoom() + delta, {animate:false});
        }""", math.log2(args.zoom))
        page.wait_for_timeout(2500)
        tiles = page.locator(".leaflet-tile-loaded").count()
        if not tiles:
            raise RuntimeError("Basemap tiles did not load; refusing to export a blank map.")
        duration = page.evaluate("maxSec")
        counts = []
        for index in range(151):
            timestamp = duration * index / 150
            counts.append(page.evaluate("""t => {
                document.getElementById('timeSlider').value = t;
                renderAtTime(t);
                return dynamicLayer.getLayers().length;
            }""", timestamp))
            page.wait_for_timeout(20)
            shot = page.screenshot(animations="disabled", type="png")
            frame = Image.open(io.BytesIO(shot)).convert("RGB")
            frames.append(frame)
            if index in (0, 75, 150):
                frame.save(args.output.with_name(args.output.stem + f"-frame-{index:03d}.png"))
        browser.close()
    if errors:
        raise RuntimeError(f"JavaScript errors: {errors}")
    assert counts[-1] > counts[0], "Timeline did not advance"
    # A common palette avoids color flicker between consecutive GIF frames.
    palette = frames[-1].quantize(colors=240)
    # Reserve saturated colors for tracks/UI; satellite imagery otherwise
    # consumes the adaptive palette and makes different tracks look identical.
    reserved = ["3b82f6", "f97316", "22c55e", "ef4444", "ffffff", "38bdf8",
                "94a3b8", "64748b", "0f172a", "1e293b", "c084fc", "f8fafc",
                "e2e8f0", "334155", "f59e0b", "000000"]
    colors = palette.getpalette()[:720]
    colors += [int(color[i:i+2], 16) for color in reserved for i in (0, 2, 4)]
    palette.putpalette(colors)
    indexed = [frame.quantize(palette=palette, dither=Image.Dither.NONE) for frame in frames]
    indexed[0].save(args.output, save_all=True, append_images=indexed[1:],
                    duration=[160] * 150 + [2000], loop=0, optimize=True, disposal=1)
    with Image.open(args.output) as result:
        assert result.n_frames > 100
        info = {"frames": result.n_frames, "size": result.size,
                "bytes": args.output.stat().st_size, "loaded_tiles": tiles,
                "source_html": str(args.html.resolve()), "source_sha256": source_sha256,
                "source_unchanged_during_capture": hashlib.sha256(args.html.read_bytes()).hexdigest() == source_sha256,
                "zoom": args.zoom,
                "layers_start_middle_end": [counts[0], counts[75], counts[-1]],
                "javascript_errors": errors}
    args.output.with_suffix(".json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
