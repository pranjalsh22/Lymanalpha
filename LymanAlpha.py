
# FINAL FIXED VERSION
import streamlit as st
import numpy as np
import pandas as pd
from astropy.io import fits
import plotly.graph_objects as go

st.set_page_config(page_title="Lyα Forest Explorer", layout="wide")

DEFAULT_FLUX = "J0011+1446_HIRES_flux(1).fits"
DEFAULT_ERROR = "J0011+1446_HIRES_error(1).fits"

def load_fits(source):
    with fits.open(source) as hdul:
        hdu_info = []
        data = None
        header = None

        for i, hdu in enumerate(hdul):
            shape = None if hdu.data is None else str(np.shape(hdu.data))
            hdu_info.append({"HDU": i, "Name": hdu.name, "Shape": shape})

            if data is None and hdu.data is not None:
                arr = np.asarray(hdu.data)

                if arr.dtype.byteorder == ">":
                    arr = arr.byteswap().view(arr.dtype.newbyteorder("="))

                arr = arr.astype(np.float64)

                if arr.ndim > 1:
                    arr = arr[0]

                data = np.squeeze(arr)
                header = hdu.header

        if data is None:
            raise ValueError("No data found in FITS file")

    return data, header, pd.DataFrame(hdu_info)

def wavelength_array(header, n):
    return 10 ** (header["CRVAL1"] + np.arange(n) * header["CDELT1"])

st.title("🌌 Lyα Forest Explorer")

uflux = st.file_uploader(
    "Upload Flux FITS",
    type=["fits"],
    key="flux_upload"
)

uerr = st.file_uploader(
    "Upload Error FITS",
    type=["fits"],
    key="error_upload"
)

if uflux and uerr:
    flux, header, hdu_df = load_fits(uflux)
    error, _, _ = load_fits(uerr)
    st.success("Using uploaded files")
else:
    flux, header, hdu_df = load_fits(DEFAULT_FLUX)
    error, _, _ = load_fits(DEFAULT_ERROR)
    st.info("Using default bundled files")

n = min(len(flux), len(error))
flux = flux[:n]
error = error[:n]

wave = wavelength_array(header, n)

snr = np.full(n, np.nan)
good = np.isfinite(error) & (error > 0)
snr[good] = flux[good] / error[good]

st.header("Object Summary")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Pixels", n)
c2.metric("λ Min", f"{wave.min():.1f}")
c3.metric("λ Max", f"{wave.max():.1f}")
c4.metric("Median SNR", f"{np.nanmedian(snr):.2f}")

st.header("FITS Structure")
st.dataframe(hdu_df, width="stretch")

st.header("Data Preview")

preview = pd.DataFrame({
    "Pixel": np.arange(n),
    "Wavelength": wave.astype(float),
    "Flux": flux.astype(float),
    "Error": error.astype(float),
    "SNR": snr.astype(float)
})

st.dataframe(preview.head(1000), width="stretch")

st.header("Wavelength Calibration")
st.latex(r"\lambda_i = 10^{CRVAL1+i\,CDELT1}")

st.header("Spectrum")

fig = go.Figure()
fig.add_trace(go.Scatter(x=wave, y=flux, name="Flux"))
fig.add_trace(go.Scatter(x=wave, y=error, name="Error"))

st.plotly_chart(fig, width="stretch")

st.header("Signal to Noise")
st.latex(r"S/N=\frac{F}{\sigma}")

fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=wave, y=snr, name="SNR"))
st.plotly_chart(fig2, width="stretch")

st.header("Header Summary")
important = {
    k: header.get(k)
    for k in ["OBJECT","INSTRUME","DATE","CRVAL1","CDELT1"]
    if k in header
}
st.json(important)

with st.expander("Full FITS Header"):
    st.json(dict(header))
