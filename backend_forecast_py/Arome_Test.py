"""
Flask Web-App: Wind-Animation für Kiel (Open-Meteo mit openmeteo-requests)
-------------------------------------------------------------------------
Abruf: http://localhost:8080/wind_animation
Erzeugt eine animierte Karte mit Windrichtungen und farbcodierter Windgeschwindigkeit.
"""

from flask import Flask, send_file  # pyright: ignore[reportMissingImports]
import io
import tempfile
import numpy as np  # pyright: ignore[reportMissingImports]
import pandas as pd  # pyright: ignore[reportMissingImports]
import matplotlib as mpl  # pyright: ignore[reportMissingImports]
mpl.use("Agg")
import matplotlib.pyplot as plt  # pyright: ignore[reportMissingImports]
import matplotlib.animation as animation  # pyright: ignore[reportMissingImports]
import cartopy.crs as ccrs  # pyright: ignore[reportMissingImports]
import cartopy.feature as cfeature  # pyright: ignore[reportMissingImports]

# Open-Meteo libs
import openmeteo_requests  # pyright: ignore[reportMissingImports]
import requests_cache  # pyright: ignore[reportMissingImports]
from retry_requests import retry  # pyright: ignore[reportMissingImports]

app = Flask(__name__)

# ==================== API-Konfiguration ====================
LAT, LON = 54.32, 10.13   # Kiel
URL = "https://api.open-meteo.com/v1/forecast"
PARAMS = {
    "latitude": LAT,
    "longitude": LON,
    "hourly": ["temperature_2m", "wind_speed_10m", "wind_direction_10m", "wind_gusts_10m"],
    "models": "meteofrance_seamless",
    "forecast_days": 1,
    "utm_source": "chatgpt.com",
}

# Session mit Cache & Retry
cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
openmeteo = openmeteo_requests.Client(session=retry_session)


# ==================== Hilfsfunktionen ====================
def fetch_wind_data():
    """Holt Winddaten aus der Open-Meteo API (über openmeteo-requests)."""
    responses = openmeteo.weather_api(URL, params=PARAMS)
    response = responses[0]   # nur 1 Standort

    hourly = response.Hourly()
    times = pd.date_range(
        start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
        end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
        freq=pd.Timedelta(seconds=hourly.Interval()),
        inclusive="left"
    )

    wind_speed = hourly.Variables(1).ValuesAsNumpy()       # m/s
    wind_dir = hourly.Variables(2).ValuesAsNumpy()         # Grad

    # In u/v-Komponenten umrechnen
    u = -wind_speed * np.sin(np.deg2rad(wind_dir))
    v = -wind_speed * np.cos(np.deg2rad(wind_dir))

    return times, u, v, wind_speed


def fig_to_gif(fig, ani):
    """Animation in GIF umwandeln und im Speicher zurückgeben."""
    buf = io.BytesIO()
    # PillowWriter infers GIF from file extension; use a temp file then return bytes
    with tempfile.NamedTemporaryFile(suffix=".gif") as tmp:
        ani.save(tmp.name, writer="pillow", dpi=100)
        tmp.seek(0)
        buf.write(tmp.read())
    plt.close(fig)
    buf.seek(0)
    return buf


# ==================== Plot-Funktion ====================
def plot_wind_animation():
    times, u, v, speed = fetch_wind_data()

    # Dummy-Raster (10x10 Gitter um Kiel)
    lats = np.linspace(LAT - 0.5, LAT + 0.5, 10)
    lons = np.linspace(LON - 0.5, LON + 0.5, 10)
    LON_GRID, LAT_GRID = np.meshgrid(lons, lats)

    fig, ax = plt.subplots(
        subplot_kw={"projection": ccrs.PlateCarree()}, figsize=(7, 6)
    )
    ax.set_extent([lons.min(), lons.max(), lats.min(), lats.max()])
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS)

    Q = ax.quiver(
        LON_GRID, LAT_GRID,
        np.zeros_like(LON_GRID), np.zeros_like(LAT_GRID),
        np.zeros_like(LON_GRID),
        cmap="viridis", scale=300,
        transform=ccrs.PlateCarree()
    )
    cb = plt.colorbar(Q, ax=ax, label="Windgeschwindigkeit (m/s)")
    title = ax.set_title("")

    def update(frame):
        # Gleicher Wert für alle Rasterpunkte (da API nur Punkt liefert)
        u_field = np.full_like(LON_GRID, u[frame])
        v_field = np.full_like(LON_GRID, v[frame])
        c_field = np.full_like(LON_GRID, speed[frame])

        Q.set_UVC(u_field, v_field, c_field)
        title.set_text(f"Kiel – Wind {times[frame]} UTC")

        return Q, title

    ani = animation.FuncAnimation(
        fig, update, frames=len(times), interval=800, blit=False
    )

    return fig_to_gif(fig, ani)


# ==================== Flask-Routen ====================
@app.route("/")
def index():
    return """
    <h1>Wetterplots für Kiel</h1>
    <ul>
        <li><a href="/wind_animation">Wind-Animation</a></li>
    </ul>
    """


@app.route("/wind_animation")
def wind_animation_route():
    return send_file(plot_wind_animation(), mimetype="image/gif")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
