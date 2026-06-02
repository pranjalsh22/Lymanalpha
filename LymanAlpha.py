
import os
import streamlit as st
import numpy as np
import pandas as pd
from astropy.io import fits
import plotly.graph_objects as go

st.set_page_config(page_title="Lyα Forest Explorer v3", layout="wide")

DEFAULT_FLUX = "J0011+1446_HIRES_flux(2).fits"
DEFAULT_ERROR = "J0011+1446_HIRES_error(2).fits"

def load_fits(source):

    with fits.open(source) as hdul:

        hdu_rows = []
        data = None
        header = None

        for i, hdu in enumerate(hdul):

            shape = None if hdu.data is None else str(np.shape(hdu.data))

            hdu_rows.append({
                "HDU": i,
                "Name": hdu.name,
                "Shape": shape
            })

            if data is None and hdu.data is not None:

                # SIMPLIFIED READER
                arr = np.array(
                    hdu.data,
                    dtype=np.float64
                )

                if arr.ndim == 2:

                    if arr.shape[0] == 1:
                        arr = arr[0]

                    elif arr.shape[1] == 1:
                        arr = arr[:, 0]

                    else:
                        arr = arr[0]

                elif arr.ndim > 2:
                    raise ValueError(
                        f"Unsupported FITS shape {arr.shape}"
                    )

                data = np.squeeze(arr)
                header = hdu.header

        if data is None:
            raise ValueError("No spectrum found")

    return data, header, pd.DataFrame(hdu_rows)


def wavelength_array(header, n):

    return 10 ** (
        header["CRVAL1"]
        + np.arange(n) * header["CDELT1"]
    )


st.title("🌌 Lyα Forest Explorer v3")

st.write("Default flux exists:", os.path.exists(DEFAULT_FLUX))
st.write("Default error exists:", os.path.exists(DEFAULT_ERROR))

flux_upload = st.file_uploader(
    "Upload Flux FITS",
    type=["fits"],
    key="flux_v3"
)

error_upload = st.file_uploader(
    "Upload Error FITS",
    type=["fits"],
    key="error_v3"
)

try:

    if flux_upload and error_upload:

        flux, header, hdu_df = load_fits(flux_upload)
        error, _, _ = load_fits(error_upload)

        st.success("Using uploaded files")

    else:

        flux, header, hdu_df = load_fits(DEFAULT_FLUX)
        error, _, _ = load_fits(DEFAULT_ERROR)

        st.info("Using bundled example spectrum")

except Exception as e:

    st.error(str(e))
    st.stop()


st.header("Diagnostic Check")

st.write("Flux shape:", flux.shape)
st.write("Error shape:", error.shape)

st.write("Flux min:", np.nanmin(flux))
st.write("Flux max:", np.nanmax(flux))

st.write("Error min:", np.nanmin(error))
st.write("Error max:", np.nanmax(error))

st.subheader("First 20 Flux Values")
st.dataframe(
    pd.DataFrame({
        "Flux": flux[:20].tolist()
    }),
    width="stretch"
)

st.subheader("First 20 Error Values")
st.dataframe(
    pd.DataFrame({
        "Error": error[:20].tolist()
    }),
    width="stretch"
)

n = min(len(flux), len(error))

flux = flux[:n]
error = error[:n]

wave = wavelength_array(header, n)

snr = np.full(n, np.nan)

good = (
    np.isfinite(error)
    & (error > 0)
)

snr[good] = (
    flux[good]
    / error[good]
)

st.header("Object Summary")

st.write(
    "Object:",
    header.get("OBJECT", "Unknown")
)

st.write(
    "Instrument:",
    header.get("INSTRUME", "Unknown")
)

st.dataframe(
    hdu_df,
    width="stretch"
)

preview = pd.DataFrame({
    "Pixel": np.arange(n, dtype=int),
    "Wavelength": wave.astype(float),
    "Flux": flux.astype(float),
    "Error": error.astype(float),
    "SNR": snr.astype(float)
})

preview = preview.replace(
    [np.inf, -np.inf],
    np.nan
)

st.header("Preview")

st.dataframe(
    preview.head(1000),
    width="stretch"
)

st.header("Spectrum")

fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=wave,
        y=flux,
        name="Flux"
    )
)

fig.add_trace(
    go.Scatter(
        x=wave,
        y=error,
        name="Error"
    )
)

st.plotly_chart(
    fig,
    width="stretch"
)

st.header("SNR")

fig2 = go.Figure()

fig2.add_trace(
    go.Scatter(
        x=wave,
        y=snr,
        name="SNR"
    )
)

st.plotly_chart(
    fig2,
    width="stretch"
)
