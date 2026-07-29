#version 9

#----section 1: imports config and dataset folder-------------
import os
import numpy as np
import pandas as pd
import streamlit as st
from astropy.io import fits
import plotly.graph_objects as go
from astropy.timeseries import LombScargle
from astropy.cosmology import FlatLambdaCDM
from scipy.interpolate import interp1d

st.set_page_config(page_title=".fit file plots",layout="wide")

SPEC_DIR = "spec" #name of the folder

redshifts = {
    "J0306+1853_HIRES": 5.363,
    "J0957+0610_UVES": 4.28,
    "J0131-0321_HIRES": 5.18,
    "J1659+2709_HIRES": 6.15,
    "J1204-0021_HIRES": 5.09,
    "J0231-0728_HIRES": 5.41,
    "J2111-0156_HIRES": 4.99,
    "J0011+1446_HIRES": 4.97,
    "J1101+0531_UVES": 4.93,
    "J0741+2520_HIRES": 5.19,
    "J1425+0827_UVES": 5.15,
    "J1008-0212_UVES": 5.07,
    "J0025-0145_HIRES": 5.07,
    "J0747+1153_HIRES": 5.26,
    "J0915+4924_HIRES": 5.20,}

# Sherwood / Sherwood-Relics cosmology
cosmo = FlatLambdaCDM(H0=67.8,Om0=0.308,Ob0=0.0482)

#----section 2:user defined functions
#----section 2.1: create arr = array of data from source using load_fits(source)----------
def load_fits(source):
    with fits.open(source) as hdul: #HDU is Header data unit list. in this case there's only one HDU 
        for hdu in hdul:
            if hdu.data is not None:
                arr = np.array(hdu.data,dtype=np.float64) 
                return (np.squeeze(arr),hdu.header)
    raise ValueError(f"No spectrum found in {source}")

#----section 2.2:create wavelength array---------
def wavelength_array(header, n):
    return 10 ** (header["CRVAL1"]+ np.arange(n) * header["CDELT1"])

#----section 2.3:calculating velocity spacing, snr quality
def velocity_spacing(header):
    return (299792.458* np.log(10)* header["CDELT1"])

#----section 2.4: quality labels-----------------
def quality_label(snr):
    if snr > 20:
        return "Excellent(>20)"
    elif snr > 10:
        return "Good(>10)"
    elif snr > 5:
        return "Moderate(>5)"
    else:
        return "Poor(<5)"

#----section 2.5: creating pair of error and flux files for same quasar
def find_pairs(folder):
    pairs = {}
    if not os.path.isdir(folder):
        return {}
    for fname in os.listdir(folder):
        if not fname.endswith(".fits"):
            continue
        full = os.path.join(folder,fname)
        if "_flux" in fname:
            key = fname.replace("_flux.fits","")
            pairs.setdefault(key,{})
            pairs[key]["flux"] = full
        elif "_error" in fname:
            key = fname.replace("_error.fits","")
            pairs.setdefault(key,{})
            pairs[key]["error"] = full
    return {k: v for k, v in pairs.items() if ("flux" in v and "error" in v)}

#---section 2.6: SNR calculation--------------------------------
def snr_calculations(n,header,flux,error):
    snr = np.full(n,np.nan) 
    good = (np.isfinite(flux) & np.isfinite(error) & (error > 0))
    snr[good] = (flux[good]/error[good])
    median_snr = float(np.nanmedian(snr))
    masked_fraction = (np.sum(~np.isfinite(flux))/len(flux) * 100)
    return snr,median_snr,masked_fraction

#---section 2.7: Lya power spectrum--------------------------------
def extract_lya_forest(wave_obs,flux,error,z,rest_min=1040,rest_max=1180):
    wave_rest = (wave_obs/ (1 + z))
    good_flux = np.isfinite(flux)
    good_error = (np.isfinite(error) & (error > 0))
    mask = ((wave_rest >= rest_min) & (wave_rest <= rest_max) & good_flux & good_error)
    return (wave_rest[mask],flux[mask])

#---section 2.8: flux_contrast--------------------------------
def flux_contrast(flux_forest):
    Fmean = np.nanmean(flux_forest)
    if not np.isfinite(Fmean):
        return None, None
    if Fmean == 0:
        return None, None
    deltaF = (flux_forest- Fmean) / Fmean
    return (Fmean,deltaF)

#---section 2.9:velocity grid--------------------------------
def velocity_grid(wave_rest):
    c = 299792.458
    velocity = (c* np.log(wave_rest))
    dv = np.median(np.diff(velocity))
    return (velocity,dv)

#---- section 2.10:FFT------------
def compute_fft(deltaF):
    N = len(deltaF)
    fft_vals = np.fft.rfft(deltaF)
    return (fft_vals,N)

#---- section 2.11: k array
def compute_k(N,dv):
    k = (2* np.pi* np.fft.rfftfreq(N,d=dv))
    return k

#---- section 2.12: power spectrum
def compute_power_spectrum(fft_vals,N,dv):
    Pk = (dv / N) * np.abs(fft_vals)**2
    return Pk

#---- section 2.14: master power spectrum function
def lya_power_spectrum_fft(wave_obs,flux,error,z):
    wave_rest, flux_forest = (extract_lya_forest(wave_obs,flux,error,z))
    if len(flux_forest) < 10:
        return None
    Fmean, deltaF = (flux_contrast(flux_forest))
    if deltaF is None:
        return None
    velocity, dv_forest = (velocity_grid(wave_rest))
    ps = power_spectrum_fft(deltaF, dv_forest)
    return {
        "wave_rest": wave_rest,
        "flux_forest": flux_forest,
        "Fmean": Fmean,
        "deltaF": deltaF,
        "velocity": velocity,
        "dv_forest": dv_forest,
        **ps}

#------section 2.15: bining the power spectrum-------------
def bin_power_spectrum(k, pk):
    valid = (np.isfinite(k) & np.isfinite(pk) & (k > 0) & (pk > 0))
    k = k[valid]
    pk = pk[valid]
    logk = np.log10(k)
    k_bin = []
    pk_bin = []
    pk_err = []
    n_modes = []

    for i in range(len(K_BIN_EDGES)-1):
        m = ((logk >= K_BIN_EDGES[i]) & (logk <  K_BIN_EDGES[i+1]))
        if np.sum(m) == 0:
            k_bin.append(np.nan)
            pk_bin.append(np.nan)
            pk_err.append(np.nan)
            n_modes.append(0)
            continue
        
        k_bin.append(np.mean(k[m]))
        pk_bin.append(np.mean(pk[m]))
        if np.sum(m) > 1:
            pk_err.append(np.std(pk[m], ddof=1))
        else:
            pk_err.append(0.0)

        n_modes.append(np.sum(m))

        
    return (
        np.asarray(k_bin),
        np.asarray(pk_bin),
        np.asarray(pk_err),
        np.asarray(n_modes))

#------section 2.16: Generic FFT Power Spectrum-------------
def power_spectrum_fft(deltaF, dv):

    fft_vals, N = compute_fft(deltaF)
    k = compute_k(N, dv)
    pk = compute_power_spectrum(fft_vals, N, dv)

    k_bin, pk_bin, pk_err,n_modes = bin_power_spectrum(
        k,
        pk
        )
    return {
        "fft": fft_vals,
        "k": k,
        "pk": pk,
        "k_bin": k_bin,
        "pk_bin": pk_bin,
        "pk_err": pk_err,
        "n_modes": n_modes}

#------section 2.16:Finding rolling mean-------------
def rolling_mean_flux(flux, chi, window_cMpc):
    half_window = window_cMpc / 2.0
    smooth = np.full_like(flux, np.nan, dtype=float)
    for i in range(len(flux)):
        left = np.searchsorted(chi, chi[i] - half_window)
        right = np.searchsorted(chi,chi[i] + half_window,side="right")
        values = flux[left:right]
        good = np.isfinite(values)
        if np.any(good):
            smooth[i] = np.mean(values[good])
    return smooth

#------section 2.17: Generic Lomb-Scargle Power Spectrum-------------
def power_spectrum_lomb(velocity, deltaF, dv):
    N = len(deltaF)
    k = compute_k(N, dv)
    k = k[1:]
    frequency = k / (2 * np.pi)
    ls = LombScargle(velocity,deltaF,normalization="psd")
    pk = ls.power(frequency)
    pk *= dv    # Convert Astropy PSD normalization to the Lyα FFT normalization.
    k_bin, pk_bin, pk_err,n_modes = bin_power_spectrum(k, pk)

    return {
        "k": k,
        "pk": pk,
        "k_bin": k_bin,
        "pk_bin": pk_bin,
        "pk_err": pk_err,
        "n_modes": n_modes}

#------section 2.17:Finding delta F-------------
def flux_contrast_rolling(flux_forest,chi_forest,window_cMpc):
    smooth = rolling_mean_flux(flux_forest,chi_forest,window_cMpc)
    deltaF = (flux_forest / smooth) - 1
    return smooth, deltaF

#------section 2.18: 
def extract_lya_forest_lomb(wave_obs,flux,error,z,rest_min=1040,rest_max=1180):
    wave_rest = (wave_obs /(1+z))
    forest = ((wave_rest >= rest_min) & (wave_rest <= rest_max))
    good = (np.isfinite(flux) & np.isfinite(error) & (error > 0))
    return (wave_rest[forest],flux[forest],good[forest])

#------section 2.19: lomb periodogram method for power spectrum
def lya_power_spectrum_lomb(wave_obs,flux,error,z,window_cMpc,segment_length):
    #1)Forest extraction
    wave_rest = wave_obs / (1 + z)
    forest = ((wave_rest >= 1040) & (wave_rest <= 1180))
    if np.sum(forest) < 10:
        return None
    wave_rest = wave_rest[forest]
    chi_forest = comoving_coordinate(wave_obs[forest])
    flux_forest = flux[forest]
    error_forest = error[forest]

    #2)Velocity coordinate
    velocity, dv_forest = velocity_grid(wave_rest)

    #3)Rolling mean normalization
    smooth, deltaF = flux_contrast_rolling(flux_forest,chi_forest,window_cMpc)
    good = (np.isfinite(flux_forest)&np.isfinite(error_forest)&(error_forest > 0)&np.isfinite(smooth)&(smooth != 0))
    if np.sum(good) < 10:
        return None

    # 4) Split into fixed comoving segments
    segments = split_into_segments(chi_forest,segment_length=segment_length)
    segment_ps = []

    for indices in segments:
        segment_good = good[indices]
        if np.sum(segment_good) < 10:
            continue
        velocity_seg = velocity[indices][segment_good]
        deltaF_seg = deltaF[indices][segment_good]
        ps = power_spectrum_lomb(velocity_seg,deltaF_seg,dv_forest)
        segment_ps.append(ps)

    average_ps = average_segment_power_spectra(segment_ps)

    # Raw Lomb spectrum of the full forest (used for Section A)
    ps_raw = power_spectrum_lomb(velocity[good],deltaF[good],dv_forest)

    return {
        "wave_rest": wave_rest,
        "flux_forest": flux_forest,
        "smooth": smooth,
        "deltaF": deltaF,
        "velocity": velocity,
        "dv_forest": dv_forest,
        "n_good": np.sum(good),
        "window_cMpc": window_cMpc,
        "segment_ps": segment_ps,

        # Raw spectrum
        "k": ps_raw["k"],
        "pk": ps_raw["pk"],
        "raw_k_bin": ps_raw["k_bin"],
        "raw_n_modes": ps_raw["n_modes"],
        
        # Averaged segment spectrum
        "k_bin": average_ps["k_bin"],
        "pk_bin": average_ps["pk_bin"],
        "pk_err": average_ps["pk_err"]}

#------section 2.20: setup to download files with correct name aka Quasar_property.png
def plotly_download_config(quasar_name,graph_name):
    return {"toImageButtonOptions": {"format": "png","filename":f"{quasar_name}_{graph_name}","height": 800,"width": 1200,"scale": 2}}

#------section 2.21: Rebin spectrum--------------------------------
def rebin_spectrum(wave, flux, error, factor=2):
    n = (len(flux) // factor) * factor
    wave = wave[:n]
    flux = flux[:n]
    error = error[:n]
    wave_rebin = wave.reshape(-1, factor).mean(axis=1)
    flux_rebin = flux.reshape(-1, factor).mean(axis=1)
    error_rebin = (np.sqrt(np.sum(error.reshape(-1, factor)**2,axis=1)) / factor)
    return (wave_rebin,flux_rebin,error_rebin)

#------section 2.23: rolling window size--------------------------------
def rolling_window_pixels(window_cMpc, chi):    #Convert a physical window (h^-1 cMpc) into an equivalent number of pixels.
    spacing = mean_pixel_spacing(chi)
    window_pixels = int(np.round(window_cMpc / spacing))
    if window_pixels % 2 == 0:
        window_pixels += 1
    return max(3, window_pixels)

#------section 2.24: comoving coordinate--------------------------------
def comoving_coordinate(wave):    #Comoving coordinate of each pixel in h^-1 cMpc.
    z = wave / 1215.67 - 1.0
    chi = cosmo.comoving_distance(z).value
    return chi * cosmo.h 

def mean_pixel_spacing(chi): #Mean pixel spacing in h^-1 cMpc.
    return np.mean(np.diff(chi))

#------section 2.25: Split forest into fixed comoving segments------------
def split_into_segments(chi, segment_length=10.0):
    
    #Parameters
    #chi : ndarray  Comoving coordinate (h^-1 cMpc).
    #segment_length : float Segment size in h^-1 cMpc.
    #Returns segments : List of index arrays, one array per complete segment.
    
    if len(chi) == 0:
        return []

    chi0 = chi[0]
    chi_end = chi[-1]
    n_segments = int((chi_end - chi0) // segment_length)
    segments = []

    for i in range(n_segments):
        start = chi0 + i * segment_length
        stop = start + segment_length
        indices = np.where((chi >= start) & (chi < stop))[0]
        if len(indices) > 1:
            segments.append(indices)
    return segments

#------section 2.26: Average segment power spectra------------
def average_segment_power_spectra(segment_ps):

    if len(segment_ps) == 0:
        return None
    k_bin = segment_ps[0]["k_bin"]
    pk = np.array([ps["pk_bin"] for ps in segment_ps])
    mean_pk = np.mean(pk, axis=0)
    std_pk = np.std(pk, axis=0, ddof=1)

    for i, ps in enumerate(segment_ps):
        print(i, np.allclose(ps["k_bin"], segment_ps[0]["k_bin"]))

    return {"k_bin": k_bin,
        "pk_bin": mean_pk,
        "pk_err": std_pk}


#------ Section 3:Setting up the data------------------------------------ 
window_cMpc = st.sidebar.number_input("Rolling Mean Window (h⁻¹ cMpc)",min_value=10.0,max_value=100.0,value=40.0,step=5.0)

st.sidebar.caption("Notation: cMpc = comoving Mpc (standard Lyα forest convention)")

segment_length = st.sidebar.number_input("Segment Length (h⁻¹ cMpc)",
    min_value=5.0,max_value=50.0,value=10.0,step=1.0)

logk_min = st.sidebar.number_input("Minimum log10(k)",value=-2.2,step=0.1)

logk_max = st.sidebar.number_input("Maximum log10(k)",value=-0.7,step=0.1)

delta_logk = st.sidebar.number_input("Δlog10(k)",value=0.1,step=0.01)

st.sidebar.markdown("---")
st.sidebar.markdown("### Cosmology Conversion")
st.sidebar.latex(r"\chi(z)=\int_0^z\frac{c\,dz'}{H(z')}")
st.sidebar.latex(r"\chi_{h^{-1}}=\chi\,h")
st.sidebar.latex(r"\Delta\chi=\chi_{i+1}-\chi_i")
st.sidebar.latex(r"N_{\rm pix}=\frac{L_{\rm cMpc}}{\langle\Delta\chi\rangle}")



K_BIN_EDGES = np.arange(logk_min,logk_max + delta_logk,delta_logk)

pairs = find_pairs(SPEC_DIR)
summary_rows = []
spectra = []

if len(pairs) == 0:
    st.error(f"No matched spectra found in {SPEC_DIR}")
    st.stop()

for key, files in pairs.items():
    try:
        flux, header = load_fits(files["flux"])
        error, _ = load_fits(files["error"])
        n = len(flux)
        wave = wavelength_array(header,n)
        snr, median_snr, masked_fraction = (snr_calculations(n,header,flux,error))
        object_name = header.get("OBJECT",key)
        instrument = header.get("INSTRUME","Unknown")
        z = redshifts[key]
        dv = velocity_spacing(header)
        chi = comoving_coordinate(wave)

        window_pixels = rolling_window_pixels(window_cMpc,chi)

        pixel_spacing = mean_pixel_spacing(chi)
        
        ps_fft = (lya_power_spectrum_fft(wave,flux,error,z))
        ps_lomb = lya_power_spectrum_lomb(wave,flux,error,z,window_cMpc,segment_length)        
        spectra.append({
            "object": object_name,
            "z": z,
            "instrument": instrument,
            "wave": wave,
            "flux": flux,
            "error": error,
            "snr": snr,
            "header": header,
            "median_snr": median_snr,
            "masked_fraction": masked_fraction,
            "dv": dv,
            "ps_fft": ps_fft,
            "ps_lomb": ps_lomb,
            "window_pixels": window_pixels,
            "chi": chi,
            "pixel_spacing": pixel_spacing,
            "segment_length": segment_length,
            "window_cMpc": window_cMpc})

        summary_rows.append({
            "Object": object_name,
            "z": z,
            "Instrument": instrument,
            "Pixels": len(flux),
            "Lambda Min": wave.min(),
            "Lambda Max": wave.max(),
            "Median S/N": median_snr,
            "Masked %": masked_fraction,
            "dv (km/s)": dv,
            "Pixel Spacing (h⁻¹ cMpc)": pixel_spacing,
            "Rolling Window (h⁻¹ cMpc)": window_cMpc,
            "Window Pixels": window_pixels})

    except Exception as e:
        st.info(f"{key}: {e}")

#-----section 4: Display---------------------------
#-----section 4.0: Setup-------------------------
st.title("Power spectrum")
summary_df = pd.DataFrame(summary_rows)
st.header("Dataset Summary")
st.dataframe(summary_df,use_container_width=True)
st.header("Individual Spectra")


#-----section 4.2: Loop Through Spectra
for spec in spectra:
    with st.expander(f"{spec['object']} ({spec['instrument']})",expanded=False):
    
        #-----section 4.2.1: Extract Stored Data
        flux = spec["flux"]
        error = spec["error"]
        wave = spec["wave"]
        snr = spec["snr"]
        ps_fft = spec["ps_fft"]
        ps_lomb = spec["ps_lomb"]
        if ps_lomb is not None:
            window_pixels = spec["window_pixels"]
            window_cMpc = spec["window_cMpc"]
            
        #-----section 4.2.2: Basic Statistics:
        finite_pixels = int(np.sum(np.isfinite(flux)))
        masked_pixels = int(np.sum(~np.isfinite(flux)))
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Median S/N",f"{spec['median_snr']:.3e}")
        c2.metric("Quality",quality_label(spec["median_snr"]))
        c3.metric("Masked %",f"{spec['masked_fraction']:.3e}")
        c4.metric("Velocity Spacing",f"{spec['dv']:.3e}")
        c5.metric("Redshift",f"{spec['z']:.3e}")
        
        st.write(f"Pixels: {len(flux)}")
        st.write(f"Finite Pixels: {finite_pixels}")
        st.write(f"Masked Pixels: {masked_pixels}")
        st.write(f"Wavelength Range: {wave.min():.1f} – {wave.max():.1f} Å")

        #-----section 4.2.3: Lyα Forest Diagnostics
        if (ps_fft is not None and ps_lomb is not None):
            col1, col2 = st.columns(2)
            
            with col1:
                F1,F2=st.columns(2)
                st.markdown("### Using FFT")
                st.metric("Forest Pixels",len(ps_fft["deltaF"]))
                st.metric("Mean Flux",f"{ps_fft['Fmean']:.3e}")
                st.metric("Minimum Modes / Bin",np.min(ps_lomb["raw_n_modes"]))
                st.metric("Mean Modes / Bin",f"{np.mean(ps_lomb['raw_n_modes']):.1f}")

            with col2:
                st.write("")
                st.markdown("### Using Lomb-Scargle Periodogram")
                st.metric("Good Pixels",ps_lomb["n_good"])
                st.metric("Forest dv",f"{ps_lomb['dv_forest']:.2f}")
                st.metric("Segment Length", f"{spec['segment_length']:.1f} h⁻¹ cMpc")
                st.metric("Segments", len(ps_lomb["segment_ps"]))
                st.metric("Minimum Modes / Bin", np.min(ps_lomb["raw_n_modes"]))
                st.write(ps_lomb.keys())
                st.metric("Mean Modes / Bin",f"{np.mean(ps_lomb['raw_n_modes']):.1f}")


            st.dataframe(pd.DataFrame({
                "log10(k)": np.log10(ps_lomb["raw_k_bin"]),
                "Modes": ps_lomb["raw_n_modes"]}))
            
        #-----section 4.2.4: Lyα Power Spectrum Plot
        if (ps_fft is not None and ps_lomb is not None ):

            #A) Raw Power Spectra
            with st.expander("A) Raw Power Spectra:"):
                fig_fft = go.Figure()
                fig_fft.add_trace(
                    go.Scatter(x=np.log10(ps_fft["k"]),y=np.log10(ps_fft["k"] * ps_fft["pk"] / np.pi),mode="lines",name="FFT"))
                fig_fft.update_layout(
                    title="FFT Power Spectrum",
                    xaxis_title="log₁₀(k / km⁻¹ s)",
                    yaxis_title="log₁₀(kP(k)/π)")

                st.plotly_chart(fig_fft,use_container_width=True,
                    config=plotly_download_config(spec["object"],"FFTPowerSpectrum"))
            
                fig_lomb = go.Figure()
                fig_lomb.add_trace(go.Scatter(
                    x=np.log10(ps_lomb["k"]),
                    y=np.log10(ps_lomb["k"] * ps_lomb["pk"] / np.pi),
                    mode="lines",
                    name="Lomb-Scargle"))
                fig_lomb.update_layout(title=(f"Lomb-Scargle Power Spectrum (Window={window_pixels} px, Bins=20)"),
                    xaxis_title="log₁₀(k / km⁻¹ s)",
                    yaxis_title="log₁₀(kP(k)/π)")

                st.plotly_chart(fig_lomb,use_container_width=True,
                    config=plotly_download_config(spec["object"],"LombScarglePowerSpectrum"))
                
            #B) Binned Power Spectra
            with st.expander("B) Binned Power Spectra", expanded=True):

                # FFT
                fig_fft_bin = go.Figure()

                fig_fft_bin.add_trace(go.Scatter(
                    x=np.log10(ps_fft["k_bin"]),
                    y=np.log10(ps_fft["k_bin"] * ps_fft["pk_bin"] / np.pi),
                    mode="markers+lines",
                    error_y=dict(type="data",array=ps_fft["pk_err"] / (ps_fft["pk_bin"] * np.log(10)),
                        visible=True),name="FFT Binned"))

                fig_fft_bin.update_layout(title="FFT Binned Power Spectrum",
                    xaxis_title="log₁₀(k / km⁻¹ s)",
                    yaxis_title="log₁₀(kP(k)/π)")

                st.plotly_chart(fig_fft_bin,use_container_width=True,
                    config=plotly_download_config(spec["object"],"FFT_Binned_PowerSpectrum"))

                # Lomb
                fig_lomb_bin = go.Figure()

                fig_lomb_bin.add_trace(go.Scatter(
                    x=np.log10(ps_lomb["k_bin"]),
                    y=np.log10(ps_lomb["k_bin"] * ps_lomb["pk_bin"] / np.pi),
                    mode="markers+lines",
                    error_y=dict(type="data",
                        array=ps_lomb["pk_err"] / (ps_lomb["pk_bin"] * np.log(10)),
                        visible=True),name="Lomb Binned"))

                fig_lomb_bin.update_layout(title=f"Lomb-Scargle Binned Power Spectrum (Window={window_pixels} px, Bins=20)",
                    xaxis_title="log₁₀(k / km⁻¹ s)",
                    yaxis_title="log₁₀(kP(k)/π)")

                st.plotly_chart(fig_lomb_bin,use_container_width=True,
                    config=plotly_download_config(spec["object"],"LombScargle_binned_PowerSpectrum"))

                
            # C) FFT vs Lomb Comparison
            with st.expander("C) FFT vs Lomb-Scargle Comparison", expanded=True):

                fig_compare = go.Figure()

                fig_compare.add_trace(go.Scatter(
                    x=np.log10(ps_fft["k_bin"]),
                    y=np.log10(ps_fft["k_bin"] * ps_fft["pk_bin"] / np.pi),
                    mode="lines",name="FFT"))

                fig_compare.add_trace(go.Scatter(
                    x=np.log10(ps_lomb["k_bin"]),
                    y=np.log10(ps_lomb["k_bin"] * ps_lomb["pk_bin"] / np.pi),
                    mode="lines",name="Lomb-Scargle"))

                fig_compare.update_layout(
                    title=f"FFT (Global Mean) vs Lomb-Scargle (Rolling Mean: Window={window_pixels} px, Bins=20)",
                    xaxis_title="log₁₀(k / km⁻¹ s)",
                    yaxis_title="log₁₀(kP(k)/π)")

                st.plotly_chart(fig_compare,use_container_width=True,
                    config=plotly_download_config(spec["object"],"FFT_vs_LombScarglePowerSpectrum"))

            # D) Large-scale Power Stability Test
            with st.expander("D) Large-scale Power Stability Test"):

                wave2, flux2, error2 = rebin_spectrum(wave,flux,error,factor=2)

                ps_fft2 = lya_power_spectrum_fft(wave2,flux2,error2,spec["z"])

                ps_lomb2 = lya_power_spectrum_lomb(wave2,flux2,error2,
                    spec["z"],spec["window_cMpc"],segment_length)
                fig = go.Figure()

                # FFT Rebinned
                fig.add_trace(go.Scatter(
                    x=np.log10(ps_fft2["k_bin"]),
                    y=np.log10(ps_fft2["k_bin"] * ps_fft2["pk_bin"] / np.pi),
                    mode="lines",
                    name="FFT (Rebinned)"))

                # FFT Original
                fig.add_trace(go.Scatter(
                    x=np.log10(ps_fft["k_bin"]),
                    y=np.log10(ps_fft["k_bin"] * ps_fft["pk_bin"] / np.pi),
                    mode="lines",
                    name="FFT (Original)",
                    line=dict(dash="dot")))

                # Lomb Rebinned
                fig.add_trace(go.Scatter(
                    x=np.log10(ps_lomb2["k_bin"]),
                    y=np.log10(ps_lomb2["k_bin"] * ps_lomb2["pk_bin"] / np.pi),
                    mode="lines",name="Lomb (Rebinned)"))

                # Lomb Original
                fig.add_trace(go.Scatter(
                    x=np.log10(ps_lomb["k_bin"]),
                    y=np.log10(ps_lomb["k_bin"] * ps_lomb["pk_bin"] / np.pi),
                    mode="lines",name="Lomb (Original)",
                    line=dict(dash="dot")))

                fig.update_layout(title="Large-scale Power Stability after Pixel Rebinning",
                    xaxis_title="log₁₀(k / km⁻¹ s)",
                    yaxis_title="log₁₀(kP(k)/π)")

                st.plotly_chart(fig,use_container_width=True,config=plotly_download_config(spec["object"],"LargeScalePowerStability"))

                st.write(ps_fft["k"].min(), ps_fft["k"].max())
                st.write(ps_fft2["k"].min(), ps_fft2["k"].max())
                st.write(ps_lomb["k"].min(), ps_lomb["k"].max())
                st.write(ps_lomb2["k"].min(), ps_lomb2["k"].max())

            #------------------------------------------------------    
            # Data Tables
            with st.expander("Data Table"):

                # Raw Data Tables
                st.subheader("Raw Power Spectrum Tables")
                fft_raw_df = pd.DataFrame({
                    "log10(k)": np.log10(ps_fft["k"]),
                    "log10(kP(k)/π)": np.log10(ps_fft["k"] * ps_fft["pk"] / np.pi)})
                lomb_raw_df = pd.DataFrame({
                    "log10(k)": np.log10(ps_lomb["k"]),
                    "log10(kP(k)/π)": np.log10(ps_lomb["k"] * ps_lomb["pk"] / np.pi)})
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("### FFT Raw")
                    st.dataframe(fft_raw_df,height=300,use_container_width=True)
                with col2:
                    st.markdown("### Lomb-Scargle Raw")
                    st.dataframe(lomb_raw_df,height=300,use_container_width=True)

                # Binned Data Tables
                st.subheader("Binned Power Spectrum Tables")
                fft_bin_df = pd.DataFrame({
                    "log10(k)": np.log10(ps_fft["k_bin"]),
                    "log10(kP(k)/π)": np.log10(ps_fft["k_bin"] * ps_fft["pk_bin"] / np.pi),
                    "σ(P)": ps_fft["pk_err"]})

                lomb_bin_df = pd.DataFrame({
                    "log10(k)": np.log10(ps_lomb["k_bin"]),
                    "log10(kP(k)/π)": np.log10(ps_lomb["k_bin"] * ps_lomb["pk_bin"] / np.pi),
                    "σ(P)": ps_lomb["pk_err"]})
                
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("### FFT Binned")
                    st.dataframe(fft_bin_df,use_container_width=True)

                with col2:
                    st.markdown("### Lomb-Scargle Binned")
                    st.dataframe(lomb_bin_df,use_container_width=True)

        #-----section 4.2.5: Rolling Mean Diagnostics
        if ps_lomb is not None:
             with st.expander("Rolling Mean Normalization Diagnostics"):

                # Plot A : Flux and Rolling Mean
                st.markdown("#### Forest Flux and Rolling Mean")
                fig_roll = go.Figure()
                fig_roll.add_trace(go.Scatter(
                        x=ps_lomb["wave_rest"],
                        y=ps_lomb["flux_forest"],
                        mode="lines",name="Flux"))

                fig_roll.add_trace(go.Scatter(
                        x=ps_lomb["wave_rest"],
                        y=ps_lomb["smooth"],
                        mode="lines",
                        name=f"Rolling Mean ({window_pixels} px)"))

                fig_roll.update_layout(title=f"Flux and Rolling Mean (Window = {window_pixels} pixels)",xaxis_title="Rest Wavelength (Å)",yaxis_title="Flux")
                    
                st.plotly_chart(fig_roll,use_container_width=True)

                # Plot B : Flux Contrast
                st.markdown("#### Rolling-Mean Normalized Flux Contrast")
                fig_delta = go.Figure()
                fig_delta.add_trace(go.Scatter(
                        x=ps_lomb["wave_rest"],
                        y=ps_lomb["deltaF"],
                        mode="lines",name="δF"))

                fig_delta.update_layout(title=f"δF = Flux / Rolling Mean - 1 (Window = {window_pixels} pixels)",xaxis_title="Rest Wavelength (Å)",yaxis_title="δF")
                st.plotly_chart(fig_delta,use_container_width=True)

        #-----section 4.2.6:Flux Spectrum
        with st.expander("Flux Spectrum"):
            mask = np.isfinite(flux)
            fig1 = go.Figure()
            fig1.add_trace(go.Scatter(
                    x=wave[mask],
                    y=flux[mask],
                    mode="lines",name="Flux"))

            fig1.update_layout(
                title="Flux Spectrum",
                xaxis_title="Observed Wavelength (Å)",
                yaxis_title="Flux")

            fig1.update_yaxes(tickformat=".2e")
            st.plotly_chart(fig1,use_container_width=True)
            
        #-----section 4.2.7:Signal-to-Noise Plot
        with st.expander("Signal to Noise Ratio",expanded=False):
            smask = np.isfinite(snr)
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=wave[smask],y=snr[smask],mode="lines",name="SNR"))

            fig2.update_layout(title="Signal-to-Noise Ratio",
                xaxis_title="Observed Wavelength (Å)",
                yaxis_title="S/N")
            st.plotly_chart(fig2,use_container_width=True)
        
        #-----section 4.2.8:Text Analysis
        st.write(f"This {spec['instrument']} spectrum contains {len(flux):,} pixels. The median S/N is {spec['median_snr']:.2f}, which corresponds to {quality_label(spec['median_snr'])} data quality. The masked fraction is {spec['masked_fraction']:.2f}% and the velocity spacing is {spec['dv']:.2f} km/s.")
            
        #-----section 4.2.9: FITS Header
        with st.expander("FITS Header"):
            st.json(dict(spec["header"]))



#-----------------------------------------------------------------------------------------------------------------------
#-----section 5: Testing the Pipeline------------------------------------
with st.expander("White Noise Validation Test", expanded=False):

    with st.expander("Normalisation used(theory):"):

        st.markdown("""For Gaussian white noise, the power spectrum is expected to be:""")
        st.latex(r"\left<P(k)\right>=\sigma^2\Delta v")
        st.markdown("The FFT power spectrum is computed using the normalization")
        st.latex(r"P_{\rm FFT}(k)=\frac{\Delta v}{N} \left|\mathrm{FFT}(\delta_F)\right|^2")

        st.markdown("""This normalization gives the power spectrum units of km s⁻¹ and
        reproduces the theoretical white-noise expectation
        ⟨P(k)⟩ = σ²Δv.""")

        st.markdown("""The Lomb–Scargle periodogram is computed using Astropy'sPSD normalization.""")
        st.latex(r"P_{\rm Lomb}(k)=\Delta v\,P_{\rm PSD}(k)")
        st.markdown("""Multiplication by Δv converts the Astropy PSD normalization to the
        same normalization convention as the FFT estimator.""")
        
    #---------------------------------------------------------
    # Generate White Noise
    #---------------------------------------------------------
    N = st.number_input("Number of points",value=5000)
    dv = st.number_input("dv=",value=2.5)

    sigma=st.number_input(r"$\sigma$ =",value=1)

    velocity = np.arange(N) * dv
    deltaF = np.random.normal(0, sigma, N)
    expected_power = sigma**2 * dv
    
    fft_test = power_spectrum_fft(deltaF, dv)
    lomb_test = power_spectrum_lomb(velocity, deltaF, dv)  

    st.subheader("Validation Statistics")
    st.write(f"σ = {sigma:.4f}")
    st.write(f"Expected <P(k)> = σ² dv = {expected_power:.4f}")
    st.write(f"FFT Mean = {np.mean(fft_test['pk']):.4f}")
    st.write(f"Lomb Mean = {np.mean(lomb_test['pk']):.4f}")


    #---------------------------------------------------------
    # A) White Noise Signal
    #---------------------------------------------------------
    with st.expander("A) White Noise Signal"):

        fig = go.Figure()

        fig.add_trace(go.Scatter(x=velocity,y=deltaF,
                mode="lines",name="White Noise"))

        fig.update_layout(title="Generated White Noise",
            xaxis_title="Velocity (km/s)",
            yaxis_title="δF")

        st.plotly_chart(fig, use_container_width=True)
        
    #---------------------------------------------------------
    # B) Binned P(k)
    #---------------------------------------------------------
    with st.expander("B) Binned P(k)", expanded=True):
        
        valid_fft = (np.isfinite(fft_test["k"])
            & np.isfinite(fft_test["pk"])
            & (fft_test["k"] > 0)
            & (fft_test["pk"] > 0))

        valid_lomb = (np.isfinite(lomb_test["k"])
            & np.isfinite(lomb_test["pk"])
            & (lomb_test["k"] > 0)
            & (lomb_test["pk"] > 0))
        #creating 21 bins 
        fft_bins = np.linspace(
            np.log10(fft_test["k"][valid_fft]).min(),
            np.log10(fft_test["k"][valid_fft]).max(),21)
        lomb_bins = np.linspace(
            np.log10(lomb_test["k"][valid_lomb]).min(),
            np.log10(lomb_test["k"][valid_lomb]).max(),21)

        fft_x = []
        fft_y = []

        for i in range(20):
            m = ((np.log10(fft_test["k"][valid_fft]) >= fft_bins[i])&
                (np.log10(fft_test["k"][valid_fft]) < fft_bins[i+1]))
            
            if np.sum(m):
                fft_x.append(np.mean(np.log10(fft_test["k"][valid_fft][m])))
                pk_values = fft_test["pk"][valid_fft][m]
                fft_y.append(np.log10(np.mean(pk_values)))

        lomb_x = []
        lomb_y = []

        for i in range(20):
            m = ((np.log10(lomb_test["k"][valid_lomb]) >= lomb_bins[i])&
                (np.log10(lomb_test["k"][valid_lomb]) < lomb_bins[i+1]))
            
            if np.sum(m):
                lomb_x.append(np.mean(np.log10(lomb_test["k"][valid_lomb][m])))
                pk_values = lomb_test["pk"][valid_lomb][m]
                lomb_y.append(np.log10(np.mean(pk_values)))
                
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(x=fft_x,y=fft_y,
                mode="markers+lines",name="FFT"))
        
        fig.add_hline(y=np.log10(expected_power),
            line_dash="dash",line_color="black",
            annotation_text="Expected <P(k)> = σ² dv")
        
        fig.add_trace(go.Scatter(x=lomb_x,y=lomb_y,
                mode="markers+lines",name="Lomb-Scargle"))

        fig.update_layout(title="White Noise Binned Power Spectrum",
            xaxis_title="log₁₀(k / km⁻¹ s)",
            yaxis_title="log₁₀(P(k))")

        st.plotly_chart(fig, use_container_width=True)

    #---------------------------------------------------------
    # C) White Noise Rebinning Test
    #---------------------------------------------------------
    with st.expander("C) White Noise Rebinning Test", expanded=False):

        st.markdown("""This test checks whether averaging neighbouring white-noise pixels
        preserves the power-spectrum amplitude.
        If the original white noise has variance σ² and pixel spacing Δv,
        then averaging every b neighbouring pixels should decrease the
        variance by approximately b while increasing the pixel spacing by b.""")

        st.latex(r"\sigma_{\rm rebin}^2 \approx \frac{\sigma^2}{b}")
        st.latex(r"\Delta v_{\rm rebin}=b\Delta v")
        st.latex(r"\left<P_{\rm rebin}(k)\right>=\frac{\sigma^2}{b}(b\Delta v)=\sigma^2\Delta v")
        
        with st.form("white_noise_rebinning_form"):

            rebin_factor_input = st.number_input("White-noise rebinning factor",
                min_value=2,max_value=20,value=2,step=1)
            run_rebinning = st.form_submit_button("Run Rebinning Test")

        if run_rebinning:

            b = int(rebin_factor_input)

            # Make sure array length is divisible by b
            n_use = (len(deltaF) // b) * b

            # Average every b neighbouring pixels
            deltaF_rebin = (deltaF[:n_use].reshape(-1, b).mean(axis=1))

            # New pixel spacing
            dv_rebin = b * dv

            # New velocity positions
            velocity_rebin = (np.arange(len(deltaF_rebin))* dv_rebin)

            # Recompute both power spectra
            fft_rebin = power_spectrum_fft(deltaF_rebin,dv_rebin)
            lomb_rebin = power_spectrum_lomb(velocity_rebin,deltaF_rebin,dv_rebin)

            # Measured variances
            variance_original = np.var(deltaF)
            variance_rebin = np.var(deltaF_rebin)

            # Mean measured power-spectrum amplitudes
            fft_original_mean = np.mean(fft_test["pk"])
            fft_rebin_mean = np.mean(fft_rebin["pk"])

            lomb_original_mean = np.mean(lomb_test["pk"])
            lomb_rebin_mean = np.mean(lomb_rebin["pk"])

            # Expected rebinned values
            fft_expected_rebin = fft_original_mean
            lomb_expected_rebin = lomb_original_mean


            fft_ratio = (fft_rebin_mean/ fft_expected_rebin)
            lomb_ratio = (lomb_rebin_mean / lomb_expected_rebin)


            fft_percent_change = (100* (fft_rebin_mean - fft_original_mean)/ fft_original_mean)

            lomb_percent_change = (100* (lomb_rebin_mean - lomb_original_mean)/ lomb_original_mean)


            valid_fft_rebin = (np.isfinite(fft_rebin["k"])& np.isfinite(fft_rebin["pk"])
                & (fft_rebin["k"] > 0) & (fft_rebin["pk"] > 0))

            fft_rebin_bins = np.linspace(np.log10(fft_rebin["k"][valid_fft_rebin]).min(),
                np.log10(fft_rebin["k"][valid_fft_rebin]).max(),21)

            fft_rebin_x = []
            fft_rebin_y = []

            for i in range(20):
                logk_values = np.log10(fft_rebin["k"][valid_fft_rebin])
                m = ((logk_values >= fft_rebin_bins[i]) & (logk_values < fft_rebin_bins[i + 1]))

                if np.sum(m):
                    fft_rebin_x.append(np.mean(logk_values[m]))
                    pk_values = (fft_rebin["pk"][valid_fft_rebin][m])
                    fft_rebin_y.append(np.log10(np.mean(pk_values)))


            valid_lomb_rebin = (np.isfinite(lomb_rebin["k"]) & np.isfinite(lomb_rebin["pk"])
                & (lomb_rebin["k"] > 0) & (lomb_rebin["pk"] > 0))

            lomb_rebin_bins = np.linspace(
                np.log10(lomb_rebin["k"][valid_lomb_rebin]).min(),
                np.log10(lomb_rebin["k"][valid_lomb_rebin]).max(),21)

            lomb_rebin_x = []
            lomb_rebin_y = []

            for i in range(20):
                logk_values = np.log10(lomb_rebin["k"][valid_lomb_rebin])
                m = ((logk_values >= lomb_rebin_bins[i]) & (logk_values < lomb_rebin_bins[i + 1]))

                if np.sum(m):
                    lomb_rebin_x.append(np.mean(logk_values[m]))
                    pk_values = (lomb_rebin["pk"][valid_lomb_rebin][m])
                    lomb_rebin_y.append(np.log10(np.mean(pk_values)))


            st.session_state["white_noise_rebin_result"] = {
                "b": b,
                
                "variance_original": variance_original,
                "variance_rebin": variance_rebin,

                "fft_original_mean": fft_original_mean,
                "fft_expected_rebin": fft_expected_rebin,
                "fft_rebin_mean": fft_rebin_mean,
                "fft_ratio": fft_ratio,
                "fft_percent_change": fft_percent_change,

                "lomb_original_mean": lomb_original_mean,
                "lomb_expected_rebin": lomb_expected_rebin,
                "lomb_rebin_mean": lomb_rebin_mean,
                "lomb_ratio": lomb_ratio,
                "lomb_percent_change": lomb_percent_change,

                "fft_rebin_x": fft_rebin_x,
                "fft_rebin_y": fft_rebin_y,

                "lomb_rebin_x": lomb_rebin_x,
                "lomb_rebin_y": lomb_rebin_y}


        if "white_noise_rebin_result" in st.session_state:

            result = st.session_state["white_noise_rebin_result"]
            b = result["b"]

            st.subheader("Rebinning Statistics")

            rebin_stats = pd.DataFrame({
                "Method": ["FFT","Lomb-Scargle"],
                "Original Mean P(k)": [result["fft_original_mean"],result["lomb_original_mean"]],
                "Expected Rebinned P(k)": [result["fft_expected_rebin"],result["lomb_expected_rebin"]],
                "Measured Rebinned P(k)": [result["fft_rebin_mean"],result["lomb_rebin_mean"]],
                "Rebinned / Original": [result["fft_ratio"],result["lomb_ratio"]],
                "Change (%)": [result["fft_percent_change"],result["lomb_percent_change"]]})

            st.dataframe(rebin_stats,hide_index=True,use_container_width=True)

            variance_ratio = (result["variance_rebin"]/ result["variance_original"])

            st.write(f"Expected variance ratio (1/b) = {1 / b:.4f}")
            st.write(f"Measured variance ratio = {variance_ratio:.4f}")

            fig = go.Figure()

            # Original FFT
            fig.add_trace(go.Scatter(x=fft_x,y=fft_y,
                    mode="markers+lines",
                    name="FFT Original",
                    line=dict(dash="dot")))

            # Rebinned FFT
            fig.add_trace(go.Scatter(
                    x=result["fft_rebin_x"],y=result["fft_rebin_y"],
                    mode="markers+lines",
                    name="FFT Rebinned"))

            # Original Lomb
            fig.add_trace(go.Scatter(
                    x=lomb_x,y=lomb_y,
                    mode="markers+lines",
                    name="Lomb Original",
                    line=dict(dash="dot")))

            # Rebinned Lomb
            fig.add_trace(go.Scatter(
                    x=result["lomb_rebin_x"],y=result["lomb_rebin_y"],
                    mode="markers+lines",
                    name="Lomb Rebinned"))

            fig.add_hline(y=np.log10(expected_power),
                line_dash="dash",line_color="black",annotation_text="σ²Δv")

            fig.update_layout(
                title=("White Noise Power Spectrum Before and After Rebinning"),
                xaxis_title="log₁₀(k / km⁻¹ s)",
                yaxis_title="log₁₀(P(k))")

            st.plotly_chart(fig,use_container_width=True)
    #---------------------------------------------------------
    # D) White Noise Masking Test
    #---------------------------------------------------------
    with st.expander("D) White Noise Masking Test", expanded=False):

        st.markdown("""This test checks whether masking contiguous sections of white noise
        changes the power-spectrum amplitude recovered by the Lomb–Scargle pipeline.
        Three power spectra are compared:
        1. The original unmasked white noise through the existing pipeline.
        2. The masked white noise through the existing pipeline.
        3. The same masked white noise evaluated on the original k-grid.
        The third calculation is a diagnostic test. It helps determine whether
        differences after masking are caused by the frequency-grid construction
        or by masking itself.""")


        st.latex(r"P_{\rm original}(k)"
            r"\quad \mathrm{vs.} \quad P_{\rm masked,pipeline}(k) \quad \mathrm{vs.} \quad"
            r"P_{\rm masked,same\text{-}grid}(k)")


        with st.form("white_noise_masking_form"):
            mask_fraction_input = st.slider("Masked fraction",
                min_value=0.05,
                max_value=0.50,
                value=0.20,
                step=0.05)

            n_mask_regions_input = st.number_input("Number of contiguous masked regions",
                min_value=1,
                max_value=20,
                value=5,step=1)

            run_masking = st.form_submit_button("Run Masking Test")

        if run_masking:
            mask_fraction = float(mask_fraction_input)
            n_mask_regions = int(n_mask_regions_input)
            N_mask = len(deltaF)

            # Exact total number of pixels to mask
            n_mask_total = int(round(mask_fraction * N_mask))
            n_mask_regions = min(n_mask_regions,n_mask_total)

            # Divide the total masked pixels among regions
            # This guarantees:
            # sum(region_sizes) == n_mask_total
            base_gap_size = (n_mask_total// n_mask_regions)
            remainder = (n_mask_total % n_mask_regions)
            region_sizes = np.full(n_mask_regions,base_gap_size,dtype=int)
            region_sizes[:remainder] += 1


            # Create non-overlapping contiguous masks
            # The complete array is divided into separate sections. One gap is randomly placed in eachsection.
            mask_rng = np.random.default_rng(42)
            mask = np.zeros(N_mask,dtype=bool)
            section_edges = np.linspace(0,N_mask,n_mask_regions + 1,dtype=int)


            for i in range(n_mask_regions):
                section_start = section_edges[i]
                section_stop = section_edges[i + 1]
                gap_size = region_sizes[i]
                max_start = (section_stop- gap_size)
                start = mask_rng.integers(section_start,max_start + 1)
                stop = (start+ gap_size)
                mask[start:stop] = True
            
            velocity_masked = velocity[~mask]
            deltaF_masked = deltaF[~mask]
            actual_mask_fraction = np.mean(mask)

            # CALCULATION 1
            # ORIGINAL WHITE NOISE Already calculated earlier

            lomb_original = lomb_test

            # CALCULATION 2
            # MASKED WHITE NOISE THROUGH ACTUAL PIPELINE

            lomb_masked_pipeline = power_spectrum_lomb(velocity_masked,deltaF_masked,dv)


            # CALCULATION 3
            # MASKED WHITE NOISE ON ORIGINAL k-GRID
            # Diagnostic calculation only.
            
            k_same_grid = lomb_original["k"]
            frequency_same_grid = (k_same_grid/ (2 * np.pi))
            ls_same_grid = LombScargle(velocity_masked,deltaF_masked,normalization="psd")

            pk_masked_same_grid = ls_same_grid.power(frequency_same_grid)


            # Same normalization correction currently used
            # by your power_spectrum_lomb() function.
            pk_masked_same_grid *= dv

            # Mean P(k) amplitudes
            original_mean = np.mean(lomb_original["pk"])
            pipeline_mean = np.mean(lomb_masked_pipeline["pk"])
            same_grid_mean = np.mean(pk_masked_same_grid)
            
            # Ratios  Desired values ≈ 1
            pipeline_ratio = (pipeline_mean / original_mean)
            same_grid_ratio = (same_grid_mean/ original_mean)

            # Percentage changes
            pipeline_percent_change = 100 * (pipeline_mean - original_mean) / original_mean
            same_grid_percent_change = 100 * (same_grid_mean - original_mean) / original_mean
            
            #=====================================================
            # BIN ALL THREE P(k) SPECTRA
            # Average P(k) in linear space first, then take log10 for plotting.
            #=====================================================


            #-----------------------------------------------------
            # Helper used ONLY for display binning.
            #
            # This does not change the power-spectrum physics.
            #-----------------------------------------------------

            def bin_pk_for_display(k, pk, n_bins=20):
                valid = (np.isfinite(k) & np.isfinite(pk) & (k > 0)& (pk > 0))
                k_valid = k[valid]
                pk_valid = pk[valid]
                logk = np.log10(k_valid)

                bins = np.linspace(logk.min(),logk.max(),n_bins + 1)
                x_bin = []
                y_bin = []

                for i in range(n_bins):
                    m = ((logk >= bins[i]) & (logk < bins[i + 1]))
                    if np.sum(m):
                        x_bin.append(np.mean(logk[m]))
                        y_bin.append(np.log10(np.mean(pk_valid[m])))
                return (np.asarray(x_bin),np.asarray(y_bin))

            # Original pipeline output
            original_x, original_y = (bin_pk_for_display(lomb_original["k"],lomb_original["pk"]))

            # Masked actual-pipeline output
            pipeline_x, pipeline_y = (bin_pk_for_display(lomb_masked_pipeline["k"],lomb_masked_pipeline["pk"]))

            # Masked same-grid output
            same_grid_x, same_grid_y = (bin_pk_for_display(k_same_grid,pk_masked_same_grid))

            #=====================================================
            # STORE RESULT
            #
            # Prevent expensive calculations from rerunning
            # whenever Streamlit reruns the script.
            #=====================================================

            st.session_state["white_noise_mask_three_curve_result"] = {
                "requested_mask_fraction":mask_fraction,
                "actual_mask_fraction":actual_mask_fraction,
                "n_mask_regions":n_mask_regions,
                "n_original":len(deltaF),
                "n_retained":len(deltaF_masked),
                "original_mean":original_mean,
                "pipeline_mean":pipeline_mean,
                "same_grid_mean":same_grid_mean,
                "pipeline_ratio":pipeline_ratio,
                "same_grid_ratio":same_grid_ratio,
                "pipeline_percent_change":pipeline_percent_change,
                "same_grid_percent_change":same_grid_percent_change,
                "velocity":velocity,
                "deltaF":deltaF,
                "velocity_masked":velocity_masked,
                "deltaF_masked":deltaF_masked,
                "original_x":original_x,
                "original_y":original_y,
                "pipeline_x":pipeline_x,
                "pipeline_y":pipeline_y,
                "same_grid_x":same_grid_x,
                "same_grid_y":same_grid_y}

        if ("white_noise_mask_three_curve_result" in st.session_state):
            result = st.session_state["white_noise_mask_three_curve_result"]

            # Statistics Table
            st.subheader("Masking Statistics")
            mask_stats = pd.DataFrame({
                "Calculation": ["Original Lomb", "Masked Lomb (Pipeline Grid)", "Masked Lomb (Original k-grid)"],
                "Mean P(k)": [result["original_mean"],result["pipeline_mean"],result["same_grid_mean"]],
                "Relative to Original": [1.0,result["pipeline_ratio"],result["same_grid_ratio"]],
                "Change (%)": [0.0,result["pipeline_percent_change"],result["same_grid_percent_change"]]})


            st.dataframe(mask_stats,hide_index=True,use_container_width=True)

            st.write(f"Requested masked fraction ={result['requested_mask_fraction']:.4f}")
            st.write(f"Actual masked fraction = {result['actual_mask_fraction']:.4f}")
            st.write(f"Original pixels = {result['n_original']}")
            st.write(f"Retained pixels = {result['n_retained']}")


            st.markdown("""**Interpretation**
            - If both masked calculations remain close to the original power, the existing Lomb–Scargle pipeline is robust to masking.
            - If the original-k-grid calculation is closer to the original power than the pipeline-grid calculation, the frequency-grid
              construction may contribute to the masking bias.
            - If both masked calculations change similarly, the difference is
              more likely associated with the sampling window or the Lomb–Scargle estimator itself.""")

            # Mask Visualization

            fig_mask = go.Figure()
            fig_mask.add_trace(go.Scattergl(
                    x=result["velocity"],y=result["deltaF"],
                    mode="lines",name="Original Noise"))
            
            fig_mask.add_trace(go.Scattergl(
                    x=result["velocity_masked"],
                    y=result["deltaF_masked"],
                    mode="markers",
                    name="Retained Samples",
                    marker=dict(size=2)))
            fig_mask.update_layout(
                title=("White Noise after Masking Contiguous Sections"),
                xaxis_title="Velocity (km/s)",yaxis_title="δF")
            st.plotly_chart(fig_mask,use_container_width=True)

            #-----------------------------------------------------
            # THREE-CURVE P(k) COMPARISON
            #-----------------------------------------------------
            fig_compare_mask = go.Figure()
            # Original
            fig_compare_mask.add_trace(go.Scatter(
                    x=result["original_x"],
                    y=result["original_y"],
                    mode="markers+lines",
                    name="Original Lomb",
                    line=dict(dash="dot")))

            # Masked through actual pipeline
            fig_compare_mask.add_trace(go.Scatter(
                    x=result["pipeline_x"],
                    y=result["pipeline_y"],
                    mode="markers+lines",
                    name="Masked Lomb (Pipeline Grid)"))

            # Masked evaluated on original grid
            fig_compare_mask.add_trace(go.Scatter(
                    x=result["same_grid_x"],
                    y=result["same_grid_y"],
                    mode="markers+lines",
                    name="Masked Lomb (Original k-grid)",
                    line=dict(dash="dash")))
            #-----------------------------------------------------
            # Original measured mean P(k) reference
            #-----------------------------------------------------
            fig_compare_mask.add_hline(
                y=np.log10(result["original_mean"]),
                line_dash="dash",
                line_color="black",
                annotation_text=("Original Mean P(k)"))
            
            fig_compare_mask.update_layout(
                title=("Masking Diagnostic: Pipeline Grid vs Original k-grid"),
                xaxis_title=("log₁₀(k / km⁻¹ s)"),yaxis_title=("log₁₀(P(k))"))

            st.plotly_chart(fig_compare_mask,use_container_width=True)


    #---------------------------------------------------------
    # E) Monte Carlo Masking Validation
    #---------------------------------------------------------
    with st.expander("E) Monte Carlo Masking Validation", expanded=False):

        st.latex(r"R_{\rm pipeline}= \frac{\langle P_{\rm masked,pipeline}(k)\rangle} {\langle P_{\rm original}(k)\rangle}")
        st.latex(r"R_{\rm same-grid}= \frac{\langle P_{\rm masked,same-grid}(k)\rangle} {\langle P_{\rm original}(k)\rangle}")
        st.markdown("""A successful masking test should give a mean ratio close to 1. Smaller scatter indicates greater realization-to-realization stability.""")

        with st.form("monte_carlo_masking_form"):
            n_realizations_input = st.number_input("Number of white-noise realizations",min_value=10, max_value=1000, value=100, step=10)
            mc_mask_fraction_input = st.slider("Monte Carlo masked fraction",min_value=0.05, max_value=0.50, value=0.20, step=0.05)
            mc_n_regions_input = st.number_input("Number of contiguous masked regions",min_value=1, max_value=20, value=3, step=1)
            run_monte_carlo = st.form_submit_button("Run Monte Carlo Test")


        if run_monte_carlo:

            n_realizations = int(n_realizations_input)
            mc_mask_fraction = float(mc_mask_fraction_input)
            mc_n_regions = int(mc_n_regions_input)

            pipeline_ratios = []
            same_grid_ratios = []

            # Fixed seed makes the complete experiment reproducible.
            mc_rng = np.random.default_rng(12345)
            progress_bar = st.progress(0)
            status_text = st.empty()

            for realization in range(n_realizations):

                deltaF_mc = mc_rng.normal(0, sigma, int(N))
                velocity_mc = np.arange(int(N)) * dv

                lomb_original_mc = power_spectrum_lomb(velocity_mc, deltaF_mc, dv)

                # Create exact, non-overlapping contiguous masked regions.
                N_mc = len(deltaF_mc)
                n_mask_total = int(round(mc_mask_fraction * N_mc))
                n_regions = min(mc_n_regions, n_mask_total)

                base_size = n_mask_total // n_regions
                remainder = n_mask_total % n_regions

                region_sizes = np.full(n_regions, base_size, dtype=int)
                region_sizes[:remainder] += 1

                section_edges = np.linspace(0, N_mc, n_regions + 1, dtype=int)
                mask_mc = np.zeros(N_mc, dtype=bool)

                for i in range(n_regions):
                    section_start = section_edges[i]
                    section_stop = section_edges[i + 1]
                    gap_size = region_sizes[i]

                    max_start = section_stop - gap_size
                    start = mc_rng.integers(section_start, max_start + 1)

                    mask_mc[start:start + gap_size] = True


                velocity_masked_mc = velocity_mc[~mask_mc]
                deltaF_masked_mc = deltaF_mc[~mask_mc]


                # Masked data through the unchanged pipeline.
                lomb_pipeline_mc = power_spectrum_lomb(velocity_masked_mc, deltaF_masked_mc, dv)

                # Diagnostic calculation using original unmasked k-grid.
                k_same_grid_mc = lomb_original_mc["k"]
                frequency_same_grid_mc = k_same_grid_mc / (2 * np.pi)

                ls_same_grid_mc = LombScargle(
                    velocity_masked_mc,
                    deltaF_masked_mc,
                    normalization="psd")

                pk_same_grid_mc = (ls_same_grid_mc.power(frequency_same_grid_mc) * dv)

                # Compare mean power-spectrum amplitudes.
                original_mean_mc = np.mean(lomb_original_mc["pk"])
                pipeline_mean_mc = np.mean(lomb_pipeline_mc["pk"])
                same_grid_mean_mc = np.mean(pk_same_grid_mc)

                pipeline_ratios.append(pipeline_mean_mc / original_mean_mc)
                same_grid_ratios.append(same_grid_mean_mc / original_mean_mc)

                progress_bar.progress((realization + 1) / n_realizations)
                status_text.write(f"Completed {realization + 1} / {n_realizations} realizations")


            pipeline_ratios = np.asarray(pipeline_ratios)
            same_grid_ratios = np.asarray(same_grid_ratios)

            pipeline_mean_ratio = np.mean(pipeline_ratios)
            pipeline_std_ratio = np.std(pipeline_ratios, ddof=1)

            same_grid_mean_ratio = np.mean(same_grid_ratios)
            same_grid_std_ratio = np.std(same_grid_ratios, ddof=1)

            pipeline_mae = np.mean(np.abs(pipeline_ratios - 1))
            same_grid_mae = np.mean(np.abs(same_grid_ratios - 1))

            pipeline_rmse = np.sqrt(np.mean((pipeline_ratios - 1)**2))
            same_grid_rmse = np.sqrt(np.mean((same_grid_ratios - 1)**2))

            pipeline_better_fraction = np.mean(
                np.abs(pipeline_ratios - 1) < np.abs(same_grid_ratios - 1))

            same_grid_better_fraction = np.mean(
                np.abs(same_grid_ratios - 1) < np.abs(pipeline_ratios - 1))

            paired_difference = same_grid_ratios - pipeline_ratios
            mean_paired_difference = np.mean(paired_difference)
            std_paired_difference = np.std(paired_difference, ddof=1)
            
            # Improvement in absolute amplitude recovery.
            accuracy_improvement = (np.abs(pipeline_ratios - 1)- np.abs(same_grid_ratios - 1))

            mean_accuracy_improvement = np.mean(accuracy_improvement)
            std_accuracy_improvement = np.std(accuracy_improvement, ddof=1)

            # Standard error of the mean improvement.
            sem_accuracy_improvement = (std_accuracy_improvement / np.sqrt(n_realizations))

            # Bootstrap 95% confidence interval for the mean improvement.
            n_bootstrap = 10000
            bootstrap_rng = np.random.default_rng(54321)

            bootstrap_means = np.empty(n_bootstrap)

            for i in range(n_bootstrap):
                bootstrap_sample = bootstrap_rng.choice(accuracy_improvement,size=n_realizations,replace=True)
                bootstrap_means[i] = np.mean(bootstrap_sample)

            bootstrap_ci_low, bootstrap_ci_high = np.percentile(bootstrap_means,[2.5, 97.5])

            # Store results so they remain visible after Streamlit reruns.
            st.session_state["monte_carlo_masking_result"] = {
                "n_realizations": n_realizations,
                "mask_fraction": mc_mask_fraction,
                "n_regions": mc_n_regions,
                "pipeline_ratios": pipeline_ratios,
                "same_grid_ratios": same_grid_ratios,
                "pipeline_mean_ratio": pipeline_mean_ratio,
                "pipeline_std_ratio": pipeline_std_ratio,
                "same_grid_mean_ratio": same_grid_mean_ratio,
                "same_grid_std_ratio": same_grid_std_ratio,
                "pipeline_mae": pipeline_mae,
                "same_grid_mae": same_grid_mae,
                "pipeline_rmse": pipeline_rmse,
                "same_grid_rmse": same_grid_rmse,
                "pipeline_better_fraction": pipeline_better_fraction,
                "same_grid_better_fraction": same_grid_better_fraction,
                "mean_paired_difference": mean_paired_difference,
                "std_paired_difference": std_paired_difference,
                "accuracy_improvement": accuracy_improvement,
                "mean_accuracy_improvement": mean_accuracy_improvement,
                "std_accuracy_improvement": std_accuracy_improvement,
                "sem_accuracy_improvement": sem_accuracy_improvement,
                "bootstrap_ci_low": bootstrap_ci_low,
                "bootstrap_ci_high": bootstrap_ci_high,}

            status_text.success("Monte Carlo masking test completed.")


        if "monte_carlo_masking_result" in st.session_state:
            result = st.session_state["monte_carlo_masking_result"]
            st.subheader("Monte Carlo Summary")
            summary = pd.DataFrame({
                "Method": ["Masked Lomb (Pipeline Grid)","Masked Lomb (Original k-grid)"],
                "Mean Ratio": [result["pipeline_mean_ratio"],result["same_grid_mean_ratio"]],
                "Std. Dev.": [result["pipeline_std_ratio"],result["same_grid_std_ratio"]],
                "Mean |Ratio - 1|": [result["pipeline_mae"],result["same_grid_mae"]],
                "RMSE from 1": [result["pipeline_rmse"],result["same_grid_rmse"]],
                "Closer to Original (%)": [100 * result["pipeline_better_fraction"],100 * result["same_grid_better_fraction"]]})

            st.dataframe(summary, hide_index=True, use_container_width=True)

            st.write(f"Realizations = {result['n_realizations']}")
            st.write(f"Masked fraction = {result['mask_fraction']:.4f}")
            st.write(f"Contiguous masked regions = {result['n_regions']}")

            st.subheader("Paired Comparison")
            st.latex(r"\Delta R=R_{\rm same-grid}-R_{\rm pipeline}")
            st.write(f"Mean paired difference = {result['mean_paired_difference']:.6f}")
            st.write(f"Std. dev. of paired difference = {result['std_paired_difference']:.6f}")

            # Plot 1: ratio for each realization.
            realization_number = np.arange(1, result["n_realizations"] + 1)

            fig_ratio = go.Figure()

            fig_ratio.add_trace(go.Scatter(
                x=realization_number,
                y=result["pipeline_ratios"],
                mode="markers",
                name="Pipeline Grid"))

            fig_ratio.add_trace(go.Scatter(
                x=realization_number,
                y=result["same_grid_ratios"],
                mode="markers",
                name="Original k-grid"))

            fig_ratio.add_hline(y=1,
                line_dash="dash",
                line_color="black",
                annotation_text="Ideal Ratio = 1")

            fig_ratio.update_layout(
                title="Masked / Original Power Ratio for Each Realization",
                xaxis_title="White-Noise Realization",
                yaxis_title="Masked / Original Mean P(k)")

            st.plotly_chart(fig_ratio, use_container_width=True)


            # Plot 2: ratio distributions.
            fig_hist = go.Figure()

            fig_hist.add_trace(go.Histogram(
                x=result["pipeline_ratios"],
                name="Pipeline Grid",
                opacity=0.65,
                nbinsx=30))

            fig_hist.add_trace(go.Histogram(
                x=result["same_grid_ratios"],
                name="Original k-grid",
                opacity=0.65,
                nbinsx=30))

            fig_hist.add_vline(x=1,
                line_dash="dash",
                line_color="black",
                annotation_text="Ideal Ratio = 1")

            fig_hist.update_layout(
                title="Distribution of Masked / Original Power Ratios",
                xaxis_title="Masked / Original Mean P(k)",
                yaxis_title="Number of Realizations",
                barmode="overlay")

            st.plotly_chart(fig_hist, use_container_width=True)


            # Plot 3: direct paired comparison.
            fig_paired = go.Figure()

            fig_paired.add_trace(go.Scatter(
                x=result["pipeline_ratios"],
                y=result["same_grid_ratios"],
                mode="markers",
                name="Realizations"))

            all_ratios = np.concatenate([result["pipeline_ratios"],result["same_grid_ratios"]])

            ratio_min = np.min(all_ratios)
            ratio_max = np.max(all_ratios)

            fig_paired.add_trace(go.Scatter(
                x=[ratio_min, ratio_max],
                y=[ratio_min, ratio_max],
                mode="lines",
                name="Equal Ratios",
                line=dict(dash="dash")))

            fig_paired.update_layout(
                title="Paired Comparison of Pipeline and Original k-grid Ratios",
                xaxis_title="Pipeline-Grid Ratio",
                yaxis_title="Original-k-grid Ratio")

            st.plotly_chart(fig_paired, use_container_width=True)

            st.info("""Interpretation:
            • Mean Ratio close to 1 indicates little systematic bias.
            • Smaller Std. Dev. indicates greater realization-to-realization scatter.
            • Smaller Mean |Ratio − 1| and RMSE indicate better recovery of the
              original power-spectrum amplitude.
            • Closer to Original (%) shows how often each method gives the smaller
              amplitude error for the same white-noise realization and mask.
            • The Monte Carlo results should be used to decide whether changing
              the Lomb–Scargle k-grid construction is justified.""")

            st.subheader("Amplitude-Recovery Comparison")

            st.latex(r"D_i= |R_{\rm pipeline,i}-1| - |R_{\rm same-grid,i}-1|")

            st.write(f"Mean improvement = {result['mean_accuracy_improvement']:.6f}")

            st.write(f"Std. dev. of improvement = {result['std_accuracy_improvement']:.6f}")

            st.write(f"Standard error = {result['sem_accuracy_improvement']:.6f}")

            st.write(f"Bootstrap 95% confidence interval = [{result['bootstrap_ci_low']:.6f}, {result['bootstrap_ci_high']:.6f}]")

            if result["bootstrap_ci_low"] > 0:
                st.success(
                    "The original k-grid gives significantly better "
                    "amplitude recovery in this experiment.")

            elif result["bootstrap_ci_high"] < 0:
                st.warning("The pipeline grid gives significantly better amplitude recovery in this experiment.")

            else:
                st.info("The confidence interval includes zero, so this experiment does not show a statistically clear "
                    "difference in amplitude recovery.")
