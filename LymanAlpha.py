
# Lyα Forest Explorer - Complete Research App
# Save as app.py

import streamlit as st
import numpy as np
import pandas as pd
from astropy.io import fits
import plotly.graph_objects as go

st.set_page_config(page_title="Lyα Forest Explorer", layout="wide")

def read_fits(uploaded_file):
    uploaded_file.seek(0)
    with fits.open(uploaded_file) as hdul:
        for hdu in hdul:
            if hdu.data is not None:
                return np.asarray(hdu.data).squeeze(), hdu.header
    raise ValueError("No data found in FITS file")

def wavelength_array(header, npix):
    crval1 = header["CRVAL1"]
    cdelt1 = header["CDELT1"]
    pix = np.arange(npix)
    return 10 ** (crval1 + pix * cdelt1)

st.title("🌌 Lyα Forest Explorer")

flux_file = st.sidebar.file_uploader("Flux FITS", type=["fits"])
error_file = st.sidebar.file_uploader("Error FITS", type=["fits"])

if flux_file is None or error_file is None:
    st.info("Upload both FITS files.")
    st.stop()

flux, header = read_fits(flux_file)
error, _ = read_fits(error_file)

n = min(len(flux), len(error))
flux, error = flux[:n], error[:n]

wave = wavelength_array(header, n)
snr = np.where(error > 0, flux / error, np.nan)

page = st.sidebar.selectbox(
    "Page",
    ["Overview","Data Preview","Spectrum","SNR","LyA Forest","Redshift","Header","Formulas"]
)

if page == "Overview":
    st.header("Overview")
    c1,c2,c3 = st.columns(3)
    c1.metric("Pixels", len(flux))
    c2.metric("λ min", f"{wave.min():.1f} Å")
    c3.metric("λ max", f"{wave.max():.1f} Å")

    st.subheader("Statistics")
    st.write(pd.DataFrame({
        "Statistic":["Mean Flux","Median Flux","Std Flux","Median SNR"],
        "Value":[np.nanmean(flux),np.nanmedian(flux),np.nanstd(flux),np.nanmedian(snr)]
    }))

elif page == "Data Preview":
    df = pd.DataFrame({
        "Pixel": np.arange(len(flux)),
        "Wavelength": wave,
        "Flux": flux,
        "Error": error,
        "SNR": snr
    })
    st.dataframe(df.head(1000), use_container_width=True)

elif page == "Spectrum":
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=wave,y=flux,name="Flux"))
    fig.add_trace(go.Scatter(x=wave,y=error,name="Error"))
    st.plotly_chart(fig, use_container_width=True)

elif page == "SNR":
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=wave,y=snr,name="SNR"))
    st.plotly_chart(fig, use_container_width=True)

elif page == "LyA Forest":
    z = st.number_input("Quasar Redshift", value=5.0)
    lya = 1215.67*(1+z)
    lyb = 1025.72*(1+z)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=wave,y=flux,name="Flux"))
    fig.add_vline(x=lya)
    fig.add_vline(x=lyb)
    st.plotly_chart(fig, use_container_width=True)

elif page == "Redshift":
    zabs = wave/1215.67 - 1
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=wave,y=zabs))
    st.plotly_chart(fig, use_container_width=True)

elif page == "Header":
    st.json(dict(header))

elif page == "Formulas":
    st.latex(r"\lambda_i = 10^{CRVAL1+iCDELT1}")
    st.latex(r"\Delta v = c\ln(10)CDELT1")
    st.latex(r"S/N = F/\sigma")
    st.latex(r"z_{abs}=\lambda_{obs}/1215.67 -1")
    st.latex(r"\lambda_{\alpha}=1215.67(1+z)")
    st.latex(r"\delta_F = F/\langle F\rangle -1")
    st.latex(r"P_F(k)=|\tilde{\delta}_F(k)|^2")
