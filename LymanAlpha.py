import streamlit as st
import numpy as np
import pandas as pd
from astropy.io import fits
import plotly.graph_objects as go

st.set_page_config(
    page_title="Lyα Forest Explorer",
    layout="wide"
)

# ---------------------------------------------------
# FITS READER
# ---------------------------------------------------

def read_fits(uploaded_file):

    uploaded_file.seek(0)

    with fits.open(uploaded_file) as hdul:

        data = None
        header = None

        for hdu in hdul:

            if hdu.data is not None:

                data = np.asarray(hdu.data).squeeze()
                header = hdu.header
                break

        if data is None:
            raise ValueError("No image data found in FITS file")

    return data, header


def wavelength_array(header, npix):

    if "CRVAL1" not in header:
        raise ValueError("CRVAL1 not found in FITS header")

    if "CDELT1" not in header:
        raise ValueError("CDELT1 not found in FITS header")

    crval1 = header["CRVAL1"]
    cdelt1 = header["CDELT1"]

    pix = np.arange(npix)

    wave = 10 ** (crval1 + pix * cdelt1)

    return wave


def velocity_width(header):

    c = 299792.458

    cdelt1 = header["CDELT1"]

    return c * np.log(10) * cdelt1


# ---------------------------------------------------
# SIDEBAR
# ---------------------------------------------------

st.sidebar.title("Lyα Forest Explorer")

flux_file = st.sidebar.file_uploader(
    "Upload Flux FITS",
    type=["fits"]
)

error_file = st.sidebar.file_uploader(
    "Upload Error FITS",
    type=["fits"]
)

if flux_file is None or error_file is None:

    st.title("Lyα Forest Explorer")

    st.info(
        "Upload a flux FITS file and an error FITS file to begin."
    )

    st.stop()

# ---------------------------------------------------
# LOAD DATA
# ---------------------------------------------------

try:

    flux, flux_header = read_fits(flux_file)

    error, error_header = read_fits(error_file)

except Exception as e:

    st.error(f"FITS read error: {e}")
    st.stop()

# ---------------------------------------------------
# CHECK LENGTHS
# ---------------------------------------------------

n = min(len(flux), len(error))

flux = flux[:n]
error = error[:n]

# ---------------------------------------------------
# WAVELENGTHS
# ---------------------------------------------------

try:

    wave = wavelength_array(
        flux_header,
        len(flux)
    )

except Exception as e:

    st.error(
        f"Could not construct wavelength array: {e}"
    )

    st.stop()

# ---------------------------------------------------
# SNR
# ---------------------------------------------------

snr = np.full_like(
    flux,
    np.nan,
    dtype=float
)

good = error > 0

snr[good] = flux[good] / error[good]

# ---------------------------------------------------
# NAVIGATION
# ---------------------------------------------------

page = st.sidebar.radio(
    "Page",
    [
        "Overview",
        "Spectrum",
        "Signal-to-Noise",
        "Lyα Forest",
        "Header"
    ]
)

# ---------------------------------------------------
# OVERVIEW
# ---------------------------------------------------

if page == "Overview":

    st.title("Data Overview")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Pixels",
        len(flux)
    )

    col2.metric(
        "λ Min (Å)",
        f"{wave.min():.1f}"
    )

    col3.metric(
        "λ Max (Å)",
        f"{wave.max():.1f}"
    )

    try:

        dv = velocity_width(flux_header)

        st.metric(
            "Velocity Width (km/s)",
            f"{dv:.3f}"
        )

    except:
        pass

    st.metric(
        "Median S/N",
        f"{np.nanmedian(snr):.2f}"
    )

# ---------------------------------------------------
# SPECTRUM
# ---------------------------------------------------

elif page == "Spectrum":

    st.title("Spectrum Viewer")

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=wave,
            y=flux,
            mode="lines",
            name="Flux"
        )
    )

    fig.add_trace(
        go.Scatter(
            x=wave,
            y=error,
            mode="lines",
            name="Error"
        )
    )

    fig.update_layout(
        xaxis_title="Wavelength (Å)",
        yaxis_title="Flux",
        height=700
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# ---------------------------------------------------
# SNR
# ---------------------------------------------------

elif page == "Signal-to-Noise":

    st.title("Signal-to-Noise")

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=wave,
            y=snr,
            mode="lines",
            name="S/N"
        )
    )

    fig.update_layout(
        xaxis_title="Wavelength (Å)",
        yaxis_title="S/N",
        height=700
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# ---------------------------------------------------
# LYA FOREST
# ---------------------------------------------------

elif page == "Lyα Forest":

    st.title("Lyα Forest Explorer")

    z_qso = st.number_input(
        "Quasar Redshift",
        value=5.0,
        step=0.01
    )

    lya = 1215.67 * (1 + z_qso)
    lyb = 1025.72 * (1 + z_qso)

    forest_start = 1040 * (1 + z_qso)
    forest_end = 1180 * (1 + z_qso)

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=wave,
            y=flux,
            mode="lines",
            name="Flux"
        )
    )

    fig.add_vline(
        x=lya,
        line_dash="dash"
    )

    fig.add_vline(
        x=lyb,
        line_dash="dash"
    )

    fig.add_vrect(
        x0=forest_start,
        x1=forest_end,
        opacity=0.2
    )

    fig.update_layout(
        xaxis_title="Wavelength (Å)",
        yaxis_title="Flux",
        height=700
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.write(
        f"Lyβ = {lyb:.2f} Å"
    )

    st.write(
        f"Lyα = {lya:.2f} Å"
    )

    st.write(
        f"Forest region = {forest_start:.2f}–{forest_end:.2f} Å"
    )

# ---------------------------------------------------
# HEADER
# ---------------------------------------------------

elif page == "Header":

    st.title("FITS Header")

    header_df = pd.DataFrame(
        {
            "Keyword": list(flux_header.keys()),
            "Value": [
                str(flux_header[k])
                for k in flux_header.keys()
            ]
        }
    )

    st.dataframe(
        header_df,
        use_container_width=True
    )

    st.subheader("Header Dictionary")

    st.json(
        dict(flux_header)
    )
