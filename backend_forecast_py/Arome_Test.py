"""
Flask app serving wind animations and an interactive Leaflet canvas.
- /               index with links
- /wind_animation animated quiver GIF over OSM basemap
- /wind_particles particle GIF (constant next-hour field)
- /wind_grid      JSON u/v grid for current hour around Kiel
- /wind_canvas    interactive Leaflet map with particle canvas (wind-js style)
"""

from flask import Flask, send_file, jsonify, Response
import io
import tempfile
from datetime import datetime, timezone, timedelta
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import cartopy.crs as ccrs
import cartopy.io.img_tiles as cimgt

# Open-Meteo libs
import openmeteo_requests
import requests_cache
from retry_requests import retry

app = Flask(__name__)

# ==================== Config ====================
LAT, LON = 54.32, 10.13  # Kiel
URL = "https://api.open-meteo.com/v1/forecast"
PARAMS = {
    "latitude": LAT,
    "longitude": LON,
    "hourly": ["temperature_2m", "wind_speed_10m", "wind_direction_10m", "wind_gusts_10m"],
    "models": "meteofrance_seamless",
    "forecast_days": 1,
    "utm_source": "wynd-forecast",
}

# Session with cache + retry
cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
openmeteo = openmeteo_requests.Client(session=retry_session)


# ==================== Data helpers ====================
def fetch_wind_data():
    responses = openmeteo.weather_api(URL, params=PARAMS)
    response = responses[0]

    hourly = response.Hourly()
    times = pd.date_range(
        start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
        end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
        freq=pd.Timedelta(seconds=hourly.Interval()),
        inclusive="left",
    )

    wind_speed = hourly.Variables(1).ValuesAsNumpy()  # m/s
    wind_dir = hourly.Variables(2).ValuesAsNumpy()    # degrees

    u = -wind_speed * np.sin(np.deg2rad(wind_dir))
    v = -wind_speed * np.cos(np.deg2rad(wind_dir))

    return times, u, v, wind_speed


# Hourly cache for constant next-hour field
WIND_CACHE = {"valid_until": None, "u": None, "v": None, "speed": None}

def get_constant_wind_next_hour():
    now = datetime.now(timezone.utc)
    valid_until = WIND_CACHE.get("valid_until")
    if valid_until is not None and now < valid_until and all(
        WIND_CACHE[k] is not None for k in ("u", "v", "speed")
    ):
        return WIND_CACHE["u"], WIND_CACHE["v"], WIND_CACHE["speed"], valid_until

    times, u_series, v_series, speed_series = fetch_wind_data()
    idx_candidates = np.where(times >= pd.Timestamp(now))
    idx = int(idx_candidates[0][0]) if len(idx_candidates[0]) > 0 else int(len(times) - 1)

    u0 = float(u_series[idx]); v0 = float(v_series[idx]); s0 = float(speed_series[idx])
    next_hour = (now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1))
    WIND_CACHE.update({"valid_until": next_hour, "u": u0, "v": v0, "speed": s0})
    return u0, v0, s0, next_hour


def fig_to_gif(fig, ani):
    buf = io.BytesIO()
    with tempfile.NamedTemporaryFile(suffix=".gif") as tmp:
        ani.save(tmp.name, writer="pillow", dpi=100)
        tmp.seek(0)
        buf.write(tmp.read())
    plt.close(fig)
    buf.seek(0)
    return buf


# ==================== GIF: Quiver ====================

def plot_wind_animation():
    times, u, v, speed = fetch_wind_data()

    lats = np.linspace(LAT - 0.5, LAT + 0.5, 10)
    lons = np.linspace(LON - 0.5, LON + 0.5, 10)
    LON_GRID, LAT_GRID = np.meshgrid(lons, lats)

    tiler = cimgt.OSM(); proj = tiler.crs
    fig, ax = plt.subplots(subplot_kw={"projection": proj}, figsize=(7, 6))
    ax.set_extent([lons.min(), lons.max(), lats.min(), lats.max()], crs=ccrs.PlateCarree())
    ax.add_image(tiler, 10)

    Q = ax.quiver(
        LON_GRID, LAT_GRID,
        np.zeros_like(LON_GRID), np.zeros_like(LAT_GRID),
        np.zeros_like(LON_GRID),
        cmap="viridis", scale=300, transform=ccrs.PlateCarree()
    )
    ax.set_title("")

    def update(frame):
        u_field = np.full_like(LON_GRID, u[frame])
        v_field = np.full_like(LON_GRID, v[frame])
        c_field = np.full_like(LON_GRID, speed[frame])
        Q.set_UVC(u_field, v_field, c_field)
        return Q,

    ani = animation.FuncAnimation(fig, update, frames=len(times), interval=800, blit=False)
    return fig_to_gif(fig, ani)


# ==================== GIF: Particles (constant hour) ====================

def plot_wind_particles():
    u0, v0, s0, _ = get_constant_wind_next_hour()

    lat_min, lat_max = LAT - 0.5, LAT + 0.5
    lon_min, lon_max = LON - 0.5, LON + 0.5

    num_particles = 600
    rng = np.random.default_rng(42)
    px = rng.uniform(lon_min, lon_max, size=num_particles)
    py = rng.uniform(lat_min, lat_max, size=num_particles)

    tiler = cimgt.OSM(); proj = tiler.crs
    fig = plt.figure(figsize=(7, 6))
    ax = plt.axes(projection=proj)
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
    ax.add_image(tiler, 10)
    particles = ax.scatter(px, py, s=2, c="white", alpha=0.9, transform=ccrs.PlateCarree())

    dt_deg = 0.01

    def step_particles(frame):
        nonlocal px, py
        scale = dt_deg * (0.3 + 0.7 * min(max(1e-6, s0) / 20.0, 1.0))
        px = px + u0 * scale
        py = py + v0 * scale
        out = (px < lon_min) | (px > lon_max) | (py < lat_min) | (py > lat_max)
        if np.any(out):
            count = int(out.sum())
            px[out] = rng.uniform(lon_min, lon_max, size=count)
            py[out] = rng.uniform(lat_min, lat_max, size=count)
        particles.set_offsets(np.c_[px, py])
        return particles,

    ani = animation.FuncAnimation(fig, step_particles, frames=180, interval=40, blit=False)
    return fig_to_gif(fig, ani)


# ==================== Routes ====================

@app.route("/")
def index():
    return (
        """
        <h1>Wetterplots für Kiel</h1>
        <ul>
            <li><a href="/wind_animation">Wind-Animation (GIF)</a></li>
            <li><a href="/wind_particles">Wind-Particles (GIF)</a></li>
            <li><a href="/wind_grid">Wind Grid (JSON)</a></li>
            <li><a href="/wind_canvas">Wind Canvas (interactive)</a></li>
        </ul>
        """
    )


@app.route("/wind_animation")
def wind_animation_route():
    return send_file(plot_wind_animation(), mimetype="image/gif")


@app.route("/wind_particles")
def wind_particles_route():
    return send_file(plot_wind_particles(), mimetype="image/gif")


@app.route("/wind_grid")
def wind_grid_route():
    lat_min, lat_max = LAT - 0.5, LAT + 0.5
    lon_min, lon_max = LON - 0.5, LON + 0.5
    nx, ny = 40, 40

    u0, v0, s0, valid_until = get_constant_wind_next_hour()
    U = np.full((ny, nx), u0, dtype=float)
    V = np.full((ny, nx), v0, dtype=float)

    return jsonify({
        "bbox": [lon_min, lat_min, lon_max, lat_max],
        "nx": int(nx),
        "ny": int(ny),
        "u": U.tolist(),
        "v": V.tolist(),
        "valid_until": valid_until.isoformat(),
        "units": {"u": "m/s", "v": "m/s"}
    })


@app.route("/wind_canvas")
def wind_canvas_route():
    html = f"""
<!doctype html>
<html>
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>Wind Canvas - Windy Style</title>
  <link rel=\"stylesheet\" href=\"https://unpkg.com/leaflet@1.9.4/dist/leaflet.css\" />
  <style>
    html, body, #map {{ height: 100%; margin: 0; }}
    .leaflet-container {{ background: #1a1a2e; }}
    #info {{ position: absolute; top: 10px; right: 10px; z-index: 1000; background: rgba(0,0,0,.7); color: #fff; padding: 10px 12px; border-radius: 6px; font: 13px/1.4 'Segoe UI', sans-serif; box-shadow: 0 2px 8px rgba(0,0,0,0.3); }}
    #info strong {{ color: #50c8ff; }}
  </style>
</head>
<body>
  <div id=\"map\"></div>
  <div id=\"info\"><strong>Wind Animation</strong><br>Data updates hourly</div>
  <script src=\"https://unpkg.com/leaflet@1.9.4/dist/leaflet.js\"></script>
  <script>
    const map = L.map('map', {{ center: [{LAT}, {LON}], zoom: 11, preferCanvas: true }});
    L.tileLayer('https://{{s}}.basemaps.cartocdn.com/dark_all/{{z}}/{{x}}/{{y}}{{r}}.png', {{ maxZoom: 18, attribution: '&copy; OpenStreetMap, CartoDB' }}).addTo(map);

    // Canvas in overlay pane following map transforms
    const overlayPane = map.getPanes().overlayPane;
    const canvas = L.DomUtil.create('canvas', 'wind-canvas');
    overlayPane.appendChild(canvas);
    const ctx = canvas.getContext('2d');
    ctx.lineWidth = 1.5; ctx.lineCap = 'round'; ctx.lineJoin = 'round';

    function positionCanvas() {{
      const size = map.getSize();
      canvas.width = size.x; canvas.height = size.y;
      const topLeft = map.containerPointToLayerPoint([0, 0]);
      L.DomUtil.setPosition(canvas, topLeft);
    }}
    map.on('move zoom resize', positionCanvas);
    positionCanvas();

    let particles = [];
    const numParticles = 2000;
    const maxAge = 120;

    function lonLatToPoint(lon, lat) {{ return map.latLngToContainerPoint([lat, lon]); }}
    function randomLonLat(b) {{
      const lon = b[0] + Math.random() * (b[2]-b[0]);
      const lat = b[1] + Math.random() * (b[3]-b[1]);
      return [lon, lat];
    }}
    function getSpeedColor(speed) {{
      if (speed < 2) return 'rgba(100, 200, 255, ALPHA)';
      if (speed < 5) return 'rgba(80, 220, 150, ALPHA)';
      if (speed < 8) return 'rgba(255, 220, 80, ALPHA)';
      if (speed < 12) return 'rgba(255, 150, 80, ALPHA)';
      return 'rgba(255, 80, 80, ALPHA)';
    }}

    fetch('/wind_grid').then(r => r.json()).then(grid => {{
      const bbox = grid.bbox; const nx = grid.nx, ny = grid.ny; const U = grid.u, V = grid.v;
      const lonMin=bbox[0], latMin=bbox[1], lonMax=bbox[2], latMax=bbox[3];

      particles = Array.from({{length: numParticles}}, () => {{
        const [lon, lat] = randomLonLat(bbox);
        return {{ lon, lat, age: Math.floor(Math.random()*maxAge), prevLon: lon, prevLat: lat }};
      }});

      function sampleVector(lon, lat) {{
        const fx = (lon - lonMin) / (lonMax - lonMin) * (nx - 1);
        const fy = (lat - latMin) / (latMax - latMin) * (ny - 1);
        const ix = Math.max(0, Math.min(nx - 2, Math.floor(fx)));
        const iy = Math.max(0, Math.min(ny - 2, Math.floor(fy)));
        const tx = fx - ix; const ty = fy - iy;
        function bilinear(A) {{
          const a = A[iy][ix], b = A[iy][ix+1], c = A[iy+1][ix], d = A[iy+1][ix+1];
          return a*(1-tx)*(1-ty) + b*tx*(1-ty) + c*(1-tx)*ty + d*tx*ty;
        }}
        const u = bilinear(U); const v = bilinear(V); const s = Math.hypot(u, v);
        return [u, v, s];
      }}

      function evolve() {{
        const w = canvas.width, h = canvas.height;
        // Trail fade
        ctx.fillStyle = 'rgba(26, 26, 46, 0.92)';
        ctx.fillRect(0, 0, w, h);

        for (const p of particles) {{
          if (p.age <= 0 || p.lon < lonMin || p.lon > lonMax || p.lat < latMin || p.lat > latMax) {{
            const r = randomLonLat(bbox); p.lon = r[0]; p.lat = r[1]; p.prevLon=r[0]; p.prevLat=r[1]; p.age = maxAge; continue;
          }}
          const [ux, vy, s] = sampleVector(p.lon, p.lat);
          p.prevLon = p.lon; p.prevLat = p.lat;
          const step = 0.010; // visual tuning
          p.lon += ux * step; p.lat += vy * step;
          const a = lonLatToPoint(p.prevLon, p.prevLat);
          const b = lonLatToPoint(p.lon, p.lat);
          const alpha = Math.min(0.95, 0.5 + 0.5 * (p.age / maxAge));
          ctx.strokeStyle = getSpeedColor(s).replace('ALPHA', alpha);
          ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y); ctx.stroke();
          p.age -= 1;
        }}
        requestAnimationFrame(evolve);
      }}
      evolve();

      // Refresh grid hourly (cheap since constant field)
      setInterval(() => {{ fetch('/wind_grid').then(r=>r.json()).then(g => {{ U.splice(0,U.length,...g.u); V.splice(0,V.length,...g.v); }}); }}, 3600000);
    }});
  </script>
</body>
</html>
    """
    return Response(html, mimetype="text/html")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)