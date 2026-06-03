
import os
import numpy as np
import pandas as pd
import streamlit as st
from astropy.io import fits
import plotly.graph_objects as go

st.set_page_config(page_title="Quasar Dataset Explorer", layout="wide")

SPECS_DIR = "specs"

def load_fits(source):
    with fits.open(source) as hdul:
        for hdu in hdul:
            if hdu.data is not None:
                data = np.squeeze(np.array(hdu.data, dtype=np.float64))
                return data, hdu.header
    raise ValueError(f"No data found in {source}")

def wavelength_array(header, n):
    return 10 ** (header["CRVAL1"] + np.arange(n) * header["CDELT1"])

def find_pairs(folder):
    pairs = {}
    if not os.path.isdir(folder):
        return {}

    for fname in os.listdir(folder):
        if not fname.lower().endswith(".fits"):
            continue

        full = os.path.join(folder, fname)

        if "_flux" in fname:
            key = fname.replace("_flux.fits", "")
            pairs.setdefault(key, {})["flux"] = full

        elif "_error" in fname:
            key = fname.replace("_error.fits", "")
            pairs.setdefault(key, {})["error"] = full

    return {k:v for k,v in pairs.items() if "flux" in v and "error" in v}

st.title("🌌 Quasar Dataset Explorer")

pairs = find_pairs(SPECS_DIR)

if not pairs:
    st.error(f"No matched FITS pairs found in '{SPECS_DIR}'")
    st.stop()

spectra = []
rows = []

for key, files in pairs.items():
    try:
        flux, header = load_fits(files["flux"])
        error, _ = load_fits(files["error"])

        if len(flux) != len(error):
            continue

        n = len(flux)
        wave = wavelength_array(header, n)

        snr = np.full(n, np.nan)
        good = np.isfinite(flux) & np.isfinite(error) & (error > 0)
        snr[good] = flux[good] / error[good]

        instrument = "HIRES" if "HIRES" in key.upper() else ("UVES" if "UVES" in key.upper() else "Unknown")
        obj = header.get("OBJECT", key)

        z = None
        for k in ["REDSHIFT", "Z", "EMZ", "QSO_Z"]:
            if k in header:
                try:
                    z = float(header[k])
                    break
                except:
                    pass

        masked = 100*np.sum(~np.isfinite(flux))/len(flux)

        rows.append({
            "Object": obj,
            "Instrument": instrument,
            "Pixels": n,
            "Lambda Min (A)": float(np.nanmin(wave)),
            "Lambda Max (A)": float(np.nanmax(wave)),
            "Median S/N": float(np.nanmedian(snr)),
            "Masked %": masked
        })

        spectra.append({
            "object": obj,
            "instrument": instrument,
            "wave": wave,
            "flux": flux,
            "error": error,
            "snr": snr,
            "z": z,
            "header": header
        })

    except Exception as e:
        st.warning(f"Failed: {key} ({e})")

df = pd.DataFrame(rows)

st.header("Dataset Summary")
st.dataframe(df, use_container_width=True)

if not df.empty:
    st.header("S/N Ranking")
    st.dataframe(df.sort_values("Median S/N", ascending=False), use_container_width=True)

st.header("All Spectra Overlay")

normalize = st.checkbox("Normalize Spectra", value=True)

fig = go.Figure()

for spec in spectra:
    mask = np.isfinite(spec["flux"])
    y = spec["flux"][mask]

    if normalize:
        med = np.nanmedian(y)
        if np.isfinite(med) and med != 0:
            y = y / med

    fig.add_trace(go.Scatter(
        x=spec["wave"][mask],
        y=y,
        mode="lines",
        name=spec["object"]
    ))

fig.update_layout(
    xaxis_title="Observed Wavelength (Å)",
    yaxis_title="Normalized Flux" if normalize else "Flux"
)

st.plotly_chart(fig, use_container_width=True)

st.header("Individual Spectrum Viewer")

selected = st.selectbox("Object", [s["object"] for s in spectra])
spec = next(s for s in spectra if s["object"] == selected)

fig2 = go.Figure()
mask = np.isfinite(spec["flux"])

fig2.add_trace(go.Scatter(
    x=spec["wave"][mask],
    y=spec["flux"][mask],
    mode="lines",
    name="Flux"
))

if spec["z"] is not None:
    z = spec["z"]
    fig2.add_vline(x=1215.67*(1+z))
    fig2.add_vline(x=1025.72*(1+z))
    fig2.add_vrect(x0=1040*(1+z), x1=1180*(1+z), opacity=0.15)

fig2.update_layout(
    xaxis_title="Observed Wavelength (Å)",
    yaxis_title="Flux"
)

st.plotly_chart(fig2, use_container_width=True)

fig3 = go.Figure()
snrmask = np.isfinite(spec["snr"])

fig3.add_trace(go.Scatter(
    x=spec["wave"][snrmask],
    y=spec["snr"][snrmask],
    mode="lines",
    name="S/N"
))

fig3.update_layout(
    xaxis_title="Observed Wavelength (Å)",
    yaxis_title="Signal-to-Noise Ratio"
)

st.plotly_chart(fig3, use_container_width=True)

with st.expander("FITS Header"):
    st.json(dict(spec["header"]))
