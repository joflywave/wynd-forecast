"""
Flask Web-App: Wetterplots für Kiel (Dummy-Daten)
-------------------------------------------------
Ruft man http://localhost:8080/<plotname> auf, liefert die App ein PNG.
Plots:
 - /temperature
 - /wind
 - /precip
 - /timeseries
"""

from flask import Flask, send_file
import io
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import pandas as pd

app = Flask(__name__)

# ==================== Dummy-Daten erzeugen ====================
times = pd.date_range("2025-10-01T00:00", periods=6, freq="1H")
lats = np.linspace(54.0, 54.6, 20)
lons = np.linspace(9.8, 10.5, 20)

temp_data = 15 + 5 * np.sin(np.linspace(0, 2*np.pi, len(times)))[:, None, None] \
            + np.random.randn(len(times), len(lats), len(lons))
u_data = 5 + np.random.randn(len(times), len(lats), len(lons))
v_data = 2 + np.random.randn(len(times), len(lats), len(lons))
precip_data = np.cumsum(np.abs(np.random.randn(len(times), len(lats), len(lons))), axis=0)

ds = xr.Dataset(
    {
        "temperature": (("time", "lat", "lon"), temp_data),
        "u_wind": (("time", "lat", "lon"), u_data),
        "v_wind": (("time", "lat", "lon"), v_data),
        "precipitation": (("time", "lat", "lon"), precip_data),
    },
    coords={"time": times, "lat": lats, "lon": lons}
)

lon_kiel, lat_kiel = 10.13, 54.32

# ==================== Plot-Funktionen ====================
def fig_to_png(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf

def plot_temperature(idx=0):
    da = ds["temperature"].isel(time=idx)
    fig, ax = plt.subplots(subplot_kw={"projection": ccrs.PlateCarree()}, figsize=(7,6))
    ax.set_extent([lons.min(), lons.max(), lats.min(), lats.max()])
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS)
    im = ax.pcolormesh(ds.lon, ds.lat, da, cmap="coolwarm", transform=ccrs.PlateCarree())
    plt.colorbar(im, ax=ax, label="Temp (°C)")
    ax.set_title(f"Temperatur (2m) {str(da.time.values)} UTC")
    return fig_to_png(fig)

def plot_wind(idx=0):
    u = ds["u_wind"].isel(time=idx)
    v = ds["v_wind"].isel(time=idx)
    fig, ax = plt.subplots(subplot_kw={"projection": ccrs.PlateCarree()}, figsize=(7,6))
    ax.set_extent([lons.min(), lons.max(), lats.min(), lats.max()])
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS)
    ax.quiver(ds.lon, ds.lat, u, v, transform=ccrs.PlateCarree(), scale=400)
    ax.set_title(f"Wind (10m) {str(u.time.values)} UTC")
    return fig_to_png(fig)

def plot_precip(idx=0):
    da = ds["precipitation"].isel(time=idx)
    fig, ax = plt.subplots(subplot_kw={"projection": ccrs.PlateCarree()}, figsize=(7,6))
    ax.set_extent([lons.min(), lons.max(), lats.min(), lats.max()])
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS)
    im = ax.pcolormesh(ds.lon, ds.lat, da, cmap="Blues", transform=ccrs.PlateCarree())
    plt.colorbar(im, ax=ax, label="Niederschlag (mm, akkumuliert)")
    ax.set_title(f"Niederschlag {str(da.time.values)} UTC")
    return fig_to_png(fig)

def plot_timeseries():
    temp = ds["temperature"].interp(lat=lat_kiel, lon=lon_kiel)
    prec = ds["precipitation"].interp(lat=lat_kiel, lon=lon_kiel)
    fig, ax1 = plt.subplots(figsize=(8,4))
    ax1.plot(ds.time, temp, "-o", color="red", label="Temp 2m (°C)")
    ax1.set_ylabel("Temp (°C)", color="red")
    ax2 = ax1.twinx()
    ax2.bar(ds.time, prec, width=0.03, color="blue", alpha=0.5, label="Niederschlag")
    ax2.set_ylabel("Niederschlag (mm)")
    fig.autofmt_xdate()
    fig.suptitle("Zeitreihe Kiel (Dummy-Daten)")
    return fig_to_png(fig)

# ==================== Flask-Routen ====================
@app.route("/")
def index():
    return """
    <h1>Wetterplots (Dummy-Daten, Kiel)</h1>
    <ul>
        <li><a href="/temperature">Temperatur</a></li>
        <li><a href="/wind">Wind</a></li>
        <li><a href="/precip">Niederschlag</a></li>
        <li><a href="/timeseries">Zeitreihe Kiel</a></li>
    </ul>
    """

@app.route("/temperature")
def temp_route():
    return send_file(plot_temperature(), mimetype="image/png")

@app.route("/wind")
def wind_route():
    return send_file(plot_wind(), mimetype="image/png")

@app.route("/precip")
def precip_route():
    return send_file(plot_precip(), mimetype="image/png")

@app.route("/timeseries")
def ts_route():
    return send_file(plot_timeseries(), mimetype="image/png")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
