
import os
import streamlit as st
import numpy as np
import pandas as pd
from astropy.io import fits
import plotly.graph_objects as go

st.set_page_config(page_title="Lyα Forest Explorer", layout="wide")

DEFAULT_FLUX = "J0011+1446_HIRES_flux.fits"
DEFAULT_ERROR = "J0011+1446_HIRES_error.fits"

def load_fits(source):
    with fits.open(source) as hdul:
        data = None
        header = None
        hdu_rows = []

        for i, hdu in enumerate(hdul):
            shape = None if hdu.data is None else str(np.shape(hdu.data))
            hdu_rows.append({"HDU": i, "Name": hdu.name, "Shape": shape})

            if data is None and hdu.data is not None:
                arr = np.array(hdu.data, dtype=np.float64)

                if arr.ndim == 2:
                    if arr.shape[0] == 1:
                        arr = arr[0]
                    elif arr.shape[1] == 1:
                        arr = arr[:, 0]

                data = np.squeeze(arr)
                header = hdu.header

        if data is None:
            raise ValueError("No spectrum found in FITS file")

    return data, header, pd.DataFrame(hdu_rows)

def wavelength_array(header, n):
    return 10 ** (header["CRVAL1"] + np.arange(n) * header["CDELT1"])

st.title("🌌 Lyα Forest Explorer")

flux_upload = st.file_uploader("Upload Flux FITS", type=["fits"], key="flux")
error_upload = st.file_uploader("Upload Error FITS", type=["fits"], key="error")

try:
    if flux_upload and error_upload:
        flux, header, hdu_df = load_fits(flux_upload)
        error, _, _ = load_fits(error_upload)
        st.success("Using uploaded files")
    else:
        flux, header, hdu_df = load_fits(DEFAULT_FLUX)
        error, _, _ = load_fits(DEFAULT_ERROR)
        st.info("Using bundled example files")
except Exception as e:
    st.error(str(e))
    st.stop()

n = min(len(flux), len(error))
flux = flux[:n]
error = error[:n]

wave = wavelength_array(header, n)

snr = np.full(n, np.nan)
good = np.isfinite(flux) & np.isfinite(error) & (error > 0)
snr[good] = flux[good] / error[good]

st.header("Object Summary")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Object", header.get("OBJECT", "Unknown"))
c2.metric("Pixels", n)
c3.metric("Finite Flux", int(np.sum(np.isfinite(flux))))
c4.metric("Masked Flux", int(np.sum(~np.isfinite(flux))))

st.header("Data Quality")

first_valid = np.where(np.isfinite(flux))[0]
if len(first_valid):
    first_valid = int(first_valid[0])
    st.write("First valid pixel:", first_valid)
else:
    first_valid = 0
    st.warning("No valid flux pixels found")

st.header("FITS Structure")
st.dataframe(hdu_df, width="stretch")

preview = pd.DataFrame({
    "Pixel": np.arange(n),
    "Wavelength": wave,
    "Flux": flux,
    "Error": error,
    "SNR": snr
})

st.header("First Valid Data")
valid_preview = preview[np.isfinite(preview["Flux"])].head(1000)
st.dataframe(valid_preview, width="stretch")

st.header("Wavelength Calibration")
st.latex(r"\lambda_i = 10^{CRVAL1 + i\,CDELT1}")

st.header("Spectrum")

mask = np.isfinite(flux)

fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=wave[mask],
        y=flux[mask],
        mode="lines",
        name="Flux"
    )
)

errmask = np.isfinite(error)
fig.add_trace(
    go.Scatter(
        x=wave[errmask],
        y=error[errmask],
        mode="lines",
        name="Error"
    )
)

st.plotly_chart(fig, width="stretch")

st.header("Signal-to-Noise")

snrmask = np.isfinite(snr)

fig2 = go.Figure()
fig2.add_trace(
    go.Scatter(
        x=wave[snrmask],
        y=snr[snrmask],
        mode="lines",
        name="SNR"
    )
)

st.plotly_chart(fig2, width="stretch")

with st.expander("Full FITS Header"):
    st.json(dict(header))
