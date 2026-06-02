
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
        hdu_info = []
        for i,hdu in enumerate(hdul):
            shape = None if hdu.data is None else str(np.shape(hdu.data))
            hdu_info.append({"HDU":i,"Name":hdu.name,"Shape":shape})
            if data is None and hdu.data is not None:
                data = np.asarray(hdu.data)
                if data.dtype.byteorder == ">":
                    data = data.byteswap().view(data.dtype.newbyteorder("="))
                data = data.astype(np.float64)
                header = hdu.header
        if data is None:
            raise ValueError("No data found")
    return data.squeeze(), header, pd.DataFrame(hdu_info)

def wavelength_array(header, n):
    return 10**(header["CRVAL1"] + np.arange(n)*header["CDELT1"])

st.title("🌌 Lyα Forest Explorer")
st.caption("Guided workflow for HIRES/UVES Lyα forest spectra")

uflux = st.file_uploader("Upload Flux FITS", type=["fits"])
uerr = st.file_uploader("Upload Error FITS", type=["fits"])

try:
    if uflux and uerr:
        flux, header, hdu_df = load_fits(uflux)
        error, _, _ = load_fits(uerr)
        st.success("Using uploaded files")
    else:
        flux, header, hdu_df = load_fits(DEFAULT_FLUX)
        error, _, _ = load_fits(DEFAULT_ERROR)
        st.info("Using bundled example spectrum (J0011+1446). Upload files to replace it.")
except Exception as e:
    st.error(f"Failed to load files: {e}")
    st.stop()

n=min(len(flux),len(error))
flux,error=flux[:n],error[:n]
wave=wavelength_array(header,n)
snr=np.where(error>0,flux/error,np.nan)

obj = header.get("OBJECT","J0011+1446")
dv = 299792.458*np.log(10)*header["CDELT1"]

st.header("1. Object Summary")
c1,c2,c3,c4=st.columns(4)
c1.metric("Object",obj)
c2.metric("Pixels",n)
c3.metric("λ Min",f"{wave.min():.1f} Å")
c4.metric("λ Max",f"{wave.max():.1f} Å")
st.metric("Velocity Bin",f"{dv:.3f} km/s")

st.header("2. FITS Structure")
st.dataframe(hdu_df,use_container_width=True)

st.header("3. Data Preview")
preview=pd.DataFrame({
    "Pixel":np.arange(n),
    "Wavelength":wave,
    "Flux":flux,
    "Error":error,
    "SNR":snr
})
st.dataframe(preview.head(500),use_container_width=True)

st.header("4. Wavelength Calibration")
st.latex(r"\lambda_i = 10^{CRVAL1 + i\,CDELT1}")
st.write(f"CRVAL1={header['CRVAL1']}   CDELT1={header['CDELT1']}")

st.header("5. Constant Velocity Binning")
st.latex(r"\Delta v = c\ln(10)\,CDELT1")
st.write(f"Δv = {dv:.3f} km/s")

st.header("6. Observed Spectrum")
fig=go.Figure()
fig.add_trace(go.Scatter(x=wave,y=flux,name="Flux"))
fig.add_trace(go.Scatter(x=wave,y=error,name="Error"))
st.plotly_chart(fig,use_container_width=True)

st.header("7. Signal-to-Noise")
st.latex(r"S/N=\frac{F}{\sigma}")
st.metric("Median SNR",f"{np.nanmedian(snr):.2f}")
fig2=go.Figure()
fig2.add_trace(go.Histogram(x=snr[np.isfinite(snr)]))
st.plotly_chart(fig2,use_container_width=True)

st.header("8. Lyα Forest")
z=st.number_input("Quasar Redshift",value=5.0,step=0.01)
st.latex(r"\lambda_{\alpha}=1215.67(1+z)")
st.latex(r"\lambda_{rest}=\frac{\lambda_{obs}}{1+z}")

rest_wave=wave/(1+z)
forest_mask=(rest_wave>1040)&(rest_wave<1180)

lya=1215.67*(1+z)
lyb=1025.72*(1+z)

fig3=go.Figure()
fig3.add_trace(go.Scatter(x=wave,y=flux,name="Flux"))
fig3.add_vline(x=lya)
fig3.add_vline(x=lyb)
st.plotly_chart(fig3,use_container_width=True)

st.subheader("Rest Frame Spectrum")
fig4=go.Figure()
fig4.add_trace(go.Scatter(x=rest_wave,y=flux,name="Flux"))
fig4.add_vrect(x0=1040,x1=1180,opacity=0.2)
st.plotly_chart(fig4,use_container_width=True)

forest_df=pd.DataFrame({
    "Rest_Wavelength":rest_wave[forest_mask],
    "Observed_Wavelength":wave[forest_mask],
    "Flux":flux[forest_mask],
    "Error":error[forest_mask]
})

st.write(f"Forest pixels: {len(forest_df)}")
st.download_button("Download Forest CSV", forest_df.to_csv(index=False), "forest_region.csv")

st.header("9. Absorber Redshift")
st.latex(r"z_{abs}=\frac{\lambda_{obs}}{1215.67}-1")
zabs=wave/1215.67-1
fig5=go.Figure()
fig5.add_trace(go.Scatter(x=wave,y=zabs))
st.plotly_chart(fig5,use_container_width=True)

st.header("10. Statistics")
stats=pd.DataFrame({
"Statistic":["Mean Flux","Median Flux","Std Flux","Median Error","Median SNR"],
"Value":[np.nanmean(flux),np.nanmedian(flux),np.nanstd(flux),np.nanmedian(error),np.nanmedian(snr)]
})
st.dataframe(stats,use_container_width=True)

st.header("11. Power Spectrum Preview")
st.warning("Educational FFT only. Not a physical Lyα flux power spectrum.")
st.latex(r"\delta_F=\frac{F}{\langle F \rangle}-1")
st.latex(r"P_F(k)=|\tilde{\delta}_F(k)|^2")

with st.expander("12. Full FITS Header"):
    st.json(dict(header))

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
        hdu_info = []
        for i,hdu in enumerate(hdul):
            shape = None if hdu.data is None else str(np.shape(hdu.data))
            hdu_info.append({"HDU":i,"Name":hdu.name,"Shape":shape})
            if data is None and hdu.data is not None:
                data = np.asarray(hdu.data)
                if data.dtype.byteorder == ">":
                    data = data.byteswap().view(data.dtype.newbyteorder("="))
                data = data.astype(np.float64)
                header = hdu.header
        if data is None:
            raise ValueError("No data found")
    return data.squeeze(), header, pd.DataFrame(hdu_info)

def wavelength_array(header, n):
    return 10**(header["CRVAL1"] + np.arange(n)*header["CDELT1"])

st.title("🌌 Lyα Forest Explorer")
st.caption("Guided workflow for HIRES/UVES Lyα forest spectra")

uflux = st.file_uploader("Upload Flux FITS", type=["fits"])
uerr = st.file_uploader("Upload Error FITS", type=["fits"])

try:
    if uflux and uerr:
        flux, header, hdu_df = load_fits(uflux)
        error, _, _ = load_fits(uerr)
        st.success("Using uploaded files")
    else:
        flux, header, hdu_df = load_fits(DEFAULT_FLUX)
        error, _, _ = load_fits(DEFAULT_ERROR)
        st.info("Using bundled example spectrum (J0011+1446). Upload files to replace it.")
except Exception as e:
    st.error(f"Failed to load files: {e}")
    st.stop()

n=min(len(flux),len(error))
flux,error=flux[:n],error[:n]
wave=wavelength_array(header,n)
snr=np.where(error>0,flux/error,np.nan)

obj = header.get("OBJECT","J0011+1446")
dv = 299792.458*np.log(10)*header["CDELT1"]

st.header("1. Object Summary")
c1,c2,c3,c4=st.columns(4)
c1.metric("Object",obj)
c2.metric("Pixels",n)
c3.metric("λ Min",f"{wave.min():.1f} Å")
c4.metric("λ Max",f"{wave.max():.1f} Å")
st.metric("Velocity Bin",f"{dv:.3f} km/s")

st.header("2. FITS Structure")
st.dataframe(hdu_df,use_container_width=True)

st.header("3. Data Preview")
preview=pd.DataFrame({
    "Pixel":np.arange(n),
    "Wavelength":wave,
    "Flux":flux,
    "Error":error,
    "SNR":snr
})
st.dataframe(preview.head(500),use_container_width=True)

st.header("4. Wavelength Calibration")
st.latex(r"\lambda_i = 10^{CRVAL1 + i\,CDELT1}")
st.write(f"CRVAL1={header['CRVAL1']}   CDELT1={header['CDELT1']}")

st.header("5. Constant Velocity Binning")
st.latex(r"\Delta v = c\ln(10)\,CDELT1")
st.write(f"Δv = {dv:.3f} km/s")

st.header("6. Observed Spectrum")
fig=go.Figure()
fig.add_trace(go.Scatter(x=wave,y=flux,name="Flux"))
fig.add_trace(go.Scatter(x=wave,y=error,name="Error"))
st.plotly_chart(fig,use_container_width=True)

st.header("7. Signal-to-Noise")
st.latex(r"S/N=\frac{F}{\sigma}")
st.metric("Median SNR",f"{np.nanmedian(snr):.2f}")
fig2=go.Figure()
fig2.add_trace(go.Histogram(x=snr[np.isfinite(snr)]))
st.plotly_chart(fig2,use_container_width=True)

st.header("8. Lyα Forest")
z=st.number_input("Quasar Redshift",value=5.0,step=0.01)
st.latex(r"\lambda_{\alpha}=1215.67(1+z)")
st.latex(r"\lambda_{rest}=\frac{\lambda_{obs}}{1+z}")

rest_wave=wave/(1+z)
forest_mask=(rest_wave>1040)&(rest_wave<1180)

lya=1215.67*(1+z)
lyb=1025.72*(1+z)

fig3=go.Figure()
fig3.add_trace(go.Scatter(x=wave,y=flux,name="Flux"))
fig3.add_vline(x=lya)
fig3.add_vline(x=lyb)
st.plotly_chart(fig3,use_container_width=True)

st.subheader("Rest Frame Spectrum")
fig4=go.Figure()
fig4.add_trace(go.Scatter(x=rest_wave,y=flux,name="Flux"))
fig4.add_vrect(x0=1040,x1=1180,opacity=0.2)
st.plotly_chart(fig4,use_container_width=True)

forest_df=pd.DataFrame({
    "Rest_Wavelength":rest_wave[forest_mask],
    "Observed_Wavelength":wave[forest_mask],
    "Flux":flux[forest_mask],
    "Error":error[forest_mask]
})

st.write(f"Forest pixels: {len(forest_df)}")
st.download_button("Download Forest CSV", forest_df.to_csv(index=False), "forest_region.csv")

st.header("9. Absorber Redshift")
st.latex(r"z_{abs}=\frac{\lambda_{obs}}{1215.67}-1")
zabs=wave/1215.67-1
fig5=go.Figure()
fig5.add_trace(go.Scatter(x=wave,y=zabs))
st.plotly_chart(fig5,use_container_width=True)

st.header("10. Statistics")
stats=pd.DataFrame({
"Statistic":["Mean Flux","Median Flux","Std Flux","Median Error","Median SNR"],
"Value":[np.nanmean(flux),np.nanmedian(flux),np.nanstd(flux),np.nanmedian(error),np.nanmedian(snr)]
})
st.dataframe(stats,use_container_width=True)

st.header("11. Power Spectrum Preview")
st.warning("Educational FFT only. Not a physical Lyα flux power spectrum.")
st.latex(r"\delta_F=\frac{F}{\langle F \rangle}-1")
st.latex(r"P_F(k)=|\tilde{\delta}_F(k)|^2")

with st.expander("12. Full FITS Header"):
    st.json(dict(header))
