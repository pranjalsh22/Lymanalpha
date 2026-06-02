
import streamlit as st
import numpy as np
import pandas as pd
from astropy.io import fits
import plotly.graph_objects as go

st.set_page_config(page_title="Lyα Forest Explorer", layout="wide")

def read_fits(uploaded_file):
    uploaded_file.seek(0)
    with fits.open(uploaded_file) as hdul:
        hdu_info = []
        data = None
        header = None
        for i, hdu in enumerate(hdul):
            shape = None if hdu.data is None else str(np.shape(hdu.data))
            hdu_info.append({"HDU": i, "Name": hdu.name, "Shape": shape})
            if data is None and hdu.data is not None:
                data = np.asarray(hdu.data).squeeze()
                header = hdu.header
    if data is None:
        raise ValueError("No data found in FITS file")
    return data, header, pd.DataFrame(hdu_info)

def wavelength_array(header, npix):
    return 10 ** (header["CRVAL1"] + np.arange(npix)*header["CDELT1"])

st.title("🌌 Lyα Forest Explorer")
st.markdown("Upload flux and error FITS files and follow the complete Lyα forest workflow.")

flux_file = st.file_uploader("Flux FITS", type=["fits"])
error_file = st.file_uploader("Error FITS", type=["fits"])

if not (flux_file and error_file):
    st.stop()

flux, header, hdu_df = read_fits(flux_file)
error, _, _ = read_fits(error_file)

n = min(len(flux), len(error))
flux, error = flux[:n], error[:n]

wave = wavelength_array(header, n)
snr = np.where(error > 0, flux/error, np.nan)

st.header("1. FITS Structure")
st.dataframe(hdu_df, use_container_width=True)

st.header("2. Data Preview")
df = pd.DataFrame({
    "Pixel": np.arange(n),
    "Wavelength (Å)": wave,
    "Flux": flux,
    "Error": error,
    "SNR": snr
})
st.dataframe(df.head(500), use_container_width=True)

st.header("3. Wavelength Calibration")
st.latex(r"\lambda_i = 10^{CRVAL1 + i\,CDELT1}")
st.write(f"CRVAL1 = {header.get('CRVAL1')}")
st.write(f"CDELT1 = {header.get('CDELT1')}")

st.header("4. Constant Velocity Binning")
st.latex(r"\Delta v = c\ln(10)\,CDELT1")
dv = 299792.458*np.log(10)*header["CDELT1"]
st.metric("Velocity Bin (km/s)", f"{dv:.3f}")

st.header("5. Spectrum")
fig = go.Figure()
fig.add_trace(go.Scatter(x=wave,y=flux,name="Flux"))
fig.add_trace(go.Scatter(x=wave,y=error,name="Error"))
st.plotly_chart(fig, use_container_width=True)

st.header("6. Signal-to-Noise")
st.latex(r"S/N = \frac{F}{\sigma}")
st.metric("Median SNR", f"{np.nanmedian(snr):.2f}")
fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=wave,y=snr,name="SNR"))
st.plotly_chart(fig2, use_container_width=True)

st.header("7. Lyα Forest Identification")
z = st.number_input("Quasar Redshift", value=5.0, step=0.01)
st.latex(r"\lambda_{\alpha}=1215.67(1+z_{QSO})")
st.latex(r"\lambda_{\beta}=1025.72(1+z_{QSO})")

lya = 1215.67*(1+z)
lyb = 1025.72*(1+z)
forest_start = 1040*(1+z)
forest_end = 1180*(1+z)

fig3 = go.Figure()
fig3.add_trace(go.Scatter(x=wave,y=flux,name="Flux"))
fig3.add_vline(x=lya)
fig3.add_vline(x=lyb)
fig3.add_vrect(x0=forest_start,x1=forest_end,opacity=0.2)
st.plotly_chart(fig3, use_container_width=True)

st.header("8. Forest Region Formula")
st.latex(r"1040\AA < \lambda_{rest} < 1180\AA")
st.latex(r"\lambda_{obs}=\lambda_{rest}(1+z_{QSO})")

st.header("9. Redshift Mapping")
st.latex(r"z_{abs}=\frac{\lambda_{obs}}{1215.67}-1")
zabs = wave/1215.67 - 1
fig4 = go.Figure()
fig4.add_trace(go.Scatter(x=wave,y=zabs))
st.plotly_chart(fig4, use_container_width=True)

st.header("10. Statistics")
stats = pd.DataFrame({
    "Statistic":["Mean Flux","Median Flux","Std Flux","Min Flux","Max Flux"],
    "Value":[np.nanmean(flux),np.nanmedian(flux),np.nanstd(flux),np.nanmin(flux),np.nanmax(flux)]
})
st.dataframe(stats, use_container_width=True)

st.header("11. Power Spectrum Preview")
st.latex(r"\delta_F=\frac{F}{\langle F\rangle}-1")
st.latex(r"P_F(k)=|\tilde{\delta}_F(k)|^2")

good = np.isfinite(flux)
f = flux[good] - np.nanmean(flux[good])
fft = np.fft.rfft(f)
power = np.abs(fft)**2
k = np.fft.rfftfreq(len(f), d=dv)

fig5 = go.Figure()
fig5.add_trace(go.Scatter(x=k[1:], y=power[1:]))
fig5.update_xaxes(type="log")
fig5.update_yaxes(type="log")
st.plotly_chart(fig5, use_container_width=True)

st.header("12. FITS Header")
st.json(dict(header))
