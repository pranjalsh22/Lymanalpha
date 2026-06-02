import streamlit as st
import numpy as np
import pandas as pd
from astropy.io import fits
import plotly.graph_objects as go

st.set_page_config(
    page_title="Lyα Forest Explorer",
    page_icon="🌌",
    layout="wide"
)

# =========================
# Helper Functions
# =========================

def read_fits(uploaded_file):
    data = fits.getdata(uploaded_file)
    header = fits.getheader(uploaded_file)
    return data, header


def wavelength_array(header, npix):
    crval1 = header["CRVAL1"]
    cdelt1 = header["CDELT1"]

    pix = np.arange(npix)

    wavelength = 10 ** (crval1 + pix * cdelt1)

    return wavelength


def velocity_width(header):
    c = 299792.458  # km/s

    cdelt1 = header["CDELT1"]

    dv = c * np.log(10) * cdelt1

    return dv


def calculate_snr(flux, error):

    good = error > 0

    snr = np.full_like(flux, np.nan, dtype=float)

    snr[good] = flux[good] / error[good]

    return snr


def lya_emission(z):
    return 1215.67 * (1 + z)


def lyb_emission(z):
    return 1025.72 * (1 + z)


# =========================
# Title
# =========================

st.title("🌌 Lyα Forest Explorer")
st.markdown(
    """
Interactive visualization and analysis tool for
HIRES / UVES quasar spectra.

Upload a flux FITS file and corresponding error FITS file.
"""
)

# =========================
# Sidebar
# =========================

st.sidebar.header("Upload Files")

flux_file = st.sidebar.file_uploader(
    "Flux FITS",
    type=["fits"]
)

error_file = st.sidebar.file_uploader(
    "Error FITS",
    type=["fits"]
)

if flux_file is None or error_file is None:

    st.info("Upload both flux and error FITS files to begin.")

    st.stop()

# =========================
# Read Data
# =========================

flux, header = read_fits(flux_file)
error, _ = read_fits(error_file)

wave = wavelength_array(header, len(flux))

snr = calculate_snr(flux, error)

dv = velocity_width(header)

# =========================
# Sidebar Navigation
# =========================

page = st.sidebar.radio(
    "Navigation",
    [
        "Overview",
        "Spectrum",
        "Signal-to-Noise",
        "Lyα Forest",
        "Redshift Mapping",
        "Power Spectrum (Preview)"
    ]
)

# =========================
# OVERVIEW
# =========================

if page == "Overview":

    st.header("Data Overview")

    col1, col2 = st.columns(2)

    with col1:

        st.metric("Pixels", len(flux))

        st.metric(
            "Velocity Bin (km/s)",
            f"{dv:.3f}"
        )

        st.metric(
            "Median S/N",
            f"{np.nanmedian(snr):.2f}"
        )

    with col2:

        st.metric(
            "λ Min (Å)",
            f"{wave.min():.2f}"
        )

        st.metric(
            "λ Max (Å)",
            f"{wave.max():.2f}"
        )

        st.metric(
            "Wavelength Range (Å)",
            f"{wave.max()-wave.min():.2f}"
        )

    st.subheader("Header Information")

    header_df = pd.DataFrame(
        {
            "Keyword": list(header.keys()),
            "Value": [str(header[k]) for k in header.keys()]
        }
    )

    st.dataframe(
        header_df,
        use_container_width=True,
        height=400
    )

    with st.expander("What am I looking at?"):

        st.markdown(
            """
### Flux Spectrum

The flux file contains the measured quasar spectrum.

### Error Spectrum

The error file contains the 1σ uncertainty for every pixel.

### Logarithmic Wavelength Grid

The wavelength is calculated as:

λ = 10^(CRVAL1 + i × CDELT1)

which corresponds to approximately constant velocity spacing.

### Lyα Forest

The dense absorption blueward of the quasar's Lyα emission
line is caused by intervening neutral hydrogen clouds and
forms the Lyα forest.
"""
        )

# =========================
# SPECTRUM
# =========================

elif page == "Spectrum":

    st.header("Spectrum Viewer")

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

# =========================
# SNR
# =========================

elif page == "Signal-to-Noise":

    st.header("Signal-to-Noise")

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

    st.subheader("S/N Distribution")

    hist = go.Figure()

    hist.add_trace(
        go.Histogram(
            x=snr[np.isfinite(snr)],
            nbinsx=100
        )
    )

    hist.update_layout(
        xaxis_title="S/N",
        yaxis_title="Count"
    )

    st.plotly_chart(
        hist,
        use_container_width=True
    )

# =========================
# LYA FOREST
# =========================

elif page == "Lyα Forest":

    st.header("Lyα Forest Explorer")

    z_qso = st.number_input(
        "Quasar Redshift",
        min_value=0.0,
        value=5.0,
        step=0.01
    )

    lya = lya_emission(z_qso)
    lyb = lyb_emission(z_qso)

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
        line_dash="dash",
        annotation_text="Lyα"
    )

    fig.add_vline(
        x=lyb,
        line_dash="dash",
        annotation_text="Lyβ"
    )

    fig.add_vrect(
        x0=forest_start,
        x1=forest_end,
        opacity=0.2,
        annotation_text="Lyα Forest"
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

    st.markdown(f"""
### Important Wavelengths

- Lyβ Emission: **{lyb:.2f} Å**
- Lyα Emission: **{lya:.2f} Å**

### Standard Forest Region

1040 Å < λ_rest < 1180 Å

Observed Forest Range:

**{forest_start:.2f} Å – {forest_end:.2f} Å**
""")

# =========================
# REDSHIFT
# =========================

elif page == "Redshift Mapping":

    st.header("Absorber Redshift Mapping")

    z_abs = wave / 1215.67 - 1

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=wave,
            y=z_abs,
            mode="lines"
        )
    )

    fig.update_layout(
        xaxis_title="Observed Wavelength (Å)",
        yaxis_title="Absorber Redshift",
        height=700
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.markdown(
        """
For every pixel:

z_abs = λ_obs / 1215.67 − 1

This converts wavelength into the redshift of the
absorbing hydrogen cloud.
"""
    )

# =========================
# POWER SPECTRUM
# =========================

elif page == "Power Spectrum (Preview)":

    st.header("Flux Power Spectrum")

    st.warning(
        """
This is only a demonstration FFT.

For scientific analysis you will need:

1. Continuum fitting
2. Mean flux normalization
3. Masking bad pixels
4. Resolution correction
5. Noise subtraction

before reproducing Boera et al. (2019).
"""
    )

    good = np.isfinite(flux)

    flux_good = flux[good]

    flux_good = flux_good - np.mean(flux_good)

    fft = np.fft.rfft(flux_good)

    power = np.abs(fft) ** 2

    k = np.fft.rfftfreq(
        len(flux_good),
        d=dv
    )

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=k[1:],
            y=power[1:],
            mode="lines"
        )
    )

    fig.update_xaxes(type="log")
    fig.update_yaxes(type="log")

    fig.update_layout(
        xaxis_title="k",
        yaxis_title="Power",
        height=700
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.markdown(
        """
This page is a placeholder for the actual
Lyα forest flux power spectrum calculation.
"""
    )
