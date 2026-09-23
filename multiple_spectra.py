#versionn 12
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

st.caption('version 12')
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

cosmo = FlatLambdaCDM(H0=67.8,Om0=0.308,Ob0=0.0482) # Sherwood / Sherwood-Relics cosmology

# SECTION 2 : USER-DEFINED FUNCTIONS

# SECTION 2.1 : GENERAL DATA HANDLING

# 2.1.1 Load FITS spectrum
def load_fits(source):
    with fits.open(source) as hdul: #HDU is Header data unit list. in this case there's only one HDU 
        for hdu in hdul:
            if hdu.data is not None:
                arr = np.array(hdu.data,dtype=np.float64) 
                return (np.squeeze(arr),hdu.header)
    raise ValueError(f"No spectrum found in {source}")

# 2.1.2 Create wavelength array
def wavelength_array(header, n):
    return 10 ** (header["CRVAL1"]+ np.arange(n) * header["CDELT1"])

# 2.1.3 Compute velocity spacing
def velocity_spacing(header):
    return (299792.458* np.log(10)* header["CDELT1"])

# 2.1.4 Assign S/N quality label
def quality_label(snr):
    if snr > 20:
        return "Excellent(>20)"
    elif snr > 10:
        return "Good(>10)"
    elif snr > 5:
        return "Moderate(>5)"
    else:
        return "Poor(<5)"
    
# 2.1.5 pair flux and error spectra
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

# 2.1.6 Compute signal-to-noise statistics
def snr_calculations(n,header,flux,error):
    snr = np.full(n,np.nan) 
    good = (np.isfinite(flux) & np.isfinite(error) & (error > 0))
    snr[good] = (flux[good]/error[good])
    median_snr = float(np.nanmedian(snr))
    masked_fraction = (np.sum(~np.isfinite(flux))/len(flux) * 100)
    return snr,median_snr,masked_fraction

# SECTION 2.2 : COMMON Lyα FOREST PREPROCESSING: Preprocessing steps shared by both the FFT and Lomb–Scargle estimators.

# 2.2.1 Extract the Lyα forest
def extract_lya_forest(wave_obs,flux,error,z,rest_min=1040,rest_max=1180):
    wave_rest = wave_obs / (1 + z)

    good = (np.isfinite(flux)
        & np.isfinite(error)
        & (error > 0))

    forest = ((wave_rest >= rest_min) & (wave_rest <= rest_max))

    mask = forest & good

    return {
        "wave_obs": wave_obs[mask],
        "wave_rest": wave_rest[mask],
        "flux": flux[mask],
        "error": error[mask]
    }

# 2.2.2 Compute flux contrast (global mean normalization)
def flux_contrast(flux_forest):
    Fmean = np.nanmean(flux_forest)
    if not np.isfinite(Fmean):
        return None, None
    if Fmean == 0:
        return None, None
    deltaF = (flux_forest- Fmean) / Fmean
    return (Fmean,deltaF)

# 2.2.3 Construct the velocity coordinate
def velocity_grid(wave_rest):
    c = 299792.458
    velocity = (c* np.log(wave_rest))
    dv = np.median(np.diff(velocity))
    return (velocity,dv)

# 2.2.4 Bin the power spectrum onto logarithmic k bins
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
        logk_center = 0.5 * (K_BIN_EDGES[i] + K_BIN_EDGES[i+1])
        k_center = 10**logk_center
        m = ((logk >= K_BIN_EDGES[i]) &
             (logk <  K_BIN_EDGES[i+1]))

        if np.sum(m) == 0:
            k_bin.append(k_center)
            pk_bin.append(np.nan)
            pk_err.append(np.nan)
            n_modes.append(0)
            continue

        k_bin.append(k_center)
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

# SECTION 2.3 : FFT POWER SPECTRUM ESTIMATOR

# ----- Step 1 : Compute the Fast Fourier Transform (FFT)
def compute_fft(deltaF):
    N = len(deltaF)
    fft_vals = np.fft.rfft(deltaF)
    return (fft_vals,N)

# ----- Step 2 : Construct the Fourier k array
def compute_k(N,dv):
    k = (2* np.pi* np.fft.rfftfreq(N,d=dv))
    return k

# ----- Step 3 : Compute the FFT power spectrum
def compute_power_spectrum(fft_vals,N,dv):
    Pk = (dv / N) * np.abs(fft_vals)**2
    return Pk

# ----- Step 4 : Assemble the generic FFT power-spectrum estimator
def power_spectrum_fft(deltaF, dv):
    fft_vals, N = compute_fft(deltaF)
    k = compute_k(N, dv)
    pk = compute_power_spectrum(fft_vals, N, dv)
    k_bin, pk_bin, pk_err,n_modes = bin_power_spectrum(k,pk)
    return {
        "fft": fft_vals,
        "k": k,
        "pk": pk,
        "k_bin": k_bin,
        "pk_bin": pk_bin,
        "pk_err": pk_err,
        "n_modes": n_modes}

# ----- Step 5 : Assemble the complete Lyα FFT pipeline
def lya_power_spectrum_fft(wave_obs,flux,error,z):
    forest = extract_lya_forest(wave_obs,flux,error,z)
    wave_rest = forest["wave_rest"]
    flux_forest = forest["flux"]
    
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

# SECTION 2.4 : LOMB–SCARGLE POWER SPECTRUM ESTIMATOR

# ----- Step 1 : Compute the rolling-mean continuum
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

# ----- Step 2 : Compute rolling-mean flux contrast (δF)
def flux_contrast_rolling(flux_forest,chi_forest,window_cMpc):
    smooth = rolling_mean_flux(flux_forest,chi_forest,window_cMpc)
    deltaF = (flux_forest / smooth) - 1
    return smooth, deltaF

# ----- Step 3 : Compute the raw Lomb–Scargle periodogram
def power_spectrum_lomb_raw(velocity, deltaF, dv):
    N = len(deltaF)
    k = compute_k(N, dv)
    k = k[1:]
    frequency = k / (2 * np.pi)
    ls = LombScargle(velocity,deltaF,normalization="psd")
    pk = ls.power(frequency)
    pk *= dv    # Convert Astropy PSD normalization to the Lyα FFT normalization.
    return {
        "k": k,
        "pk": pk}

# ----- Step 4 : Estimate the noise power spectrum using Monte Carlo realizations
def estimate_noise_periodogram(velocity,error,smooth,dv,n_realizations=200):
    noise_pk = []
    good = (np.isfinite(error) & np.isfinite(smooth) & (smooth != 0))
    velocity = velocity[good]
    sigma = error[good] / smooth[good]

    for _ in range(n_realizations):
        deltaF_noise = np.random.normal(loc=0.0,scale=sigma)
        ps = power_spectrum_lomb_raw(velocity,deltaF_noise,dv)
        noise_pk.append(ps["pk"])

    noise_pk = np.asarray(noise_pk)
    return {
        "k": ps["k"],
        "pk": np.mean(noise_pk, axis=0)}

# ----- Step 5 : Assemble the complete Lyα Lomb–Scargle pipeline
def lya_power_spectrum_lomb(wave_obs, flux, error, z, window_cMpc, segment_length):


    forest = extract_lya_forest(wave_obs,flux,error,z)
    wave_obs_forest = forest["wave_obs"]
    wave_rest = forest["wave_rest"]
    flux_forest = forest["flux"]
    error_forest = forest["error"]

    if len(flux_forest) < 10:
        return None

    chi_forest = comoving_coordinate(wave_obs_forest)

    
    velocity, dv_forest = velocity_grid(wave_rest)
    smooth, deltaF = flux_contrast_rolling(flux_forest,chi_forest, window_cMpc)

    good = (np.isfinite(flux_forest)
        & np.isfinite(error_forest)
        & (error_forest > 0)
        & np.isfinite(smooth)
        & (smooth != 0))

    if np.sum(good) < 10:
        return None

    segments = split_into_segments(chi_forest,segment_length=segment_length)
    segment_ps = []

    for indices in segments:
        segment_good = good[indices]
        if np.sum(segment_good) < 10:
            continue

        velocity_seg = velocity[indices][segment_good]
        deltaF_seg = deltaF[indices][segment_good]
        error_seg = error_forest[indices][segment_good]
        smooth_seg = smooth[indices][segment_good]
        raw_ps_seg = power_spectrum_lomb_raw(velocity_seg,deltaF_seg,dv_forest)
        noise_ps_seg = estimate_noise_periodogram(velocity_seg,error_seg,smooth_seg,dv_forest)
        pk_corrected = (raw_ps_seg["pk"]- noise_ps_seg["pk"])
        k_bin, pk_bin, pk_err, n_modes = bin_power_spectrum(raw_ps_seg["k"],pk_corrected)
        wave_center = np.mean(wave_obs_forest[indices])
        segment_z = wave_center / 1215.67 - 1.0
        segment_ps.append({
            "z": segment_z,
            "k_bin": k_bin,
            "pk_bin": pk_bin,
            "pk_err": pk_err,
            "n_modes": n_modes})

    average_ps = average_segment_power_spectra(segment_ps)

    if average_ps is None:
        return None

    # --------------------------------------------------
    # 5) Full forest (diagnostic only)
    # --------------------------------------------------
    raw_ps = power_spectrum_lomb_raw(velocity[good],deltaF[good],dv_forest)
    noise_ps = estimate_noise_periodogram(velocity[good],error_forest[good],smooth[good],dv_forest)
    pk_corrected = raw_ps["pk"] - noise_ps["pk"]

    # Optional: useful for diagnostic plots
    k_bin, pk_bin, pk_err, n_modes = bin_power_spectrum(raw_ps["k"],pk_corrected)

    return {
        "wave_rest": wave_rest,
        "flux_forest": flux_forest,
        "smooth": smooth,
        "deltaF": deltaF,
        "velocity": velocity,
        "dv_forest": dv_forest,
        "n_good": np.sum(good),
        "window_cMpc": window_cMpc,
        "n_modes": average_ps["n_modes"],

        # Raw full-forest spectrum (diagnostic)
        "k": raw_ps["k"],
        "pk": raw_ps["pk"],
        "noise_pk": noise_ps["pk"],
        "pk_corrected": pk_corrected,

        # Individual corrected segment spectra
        "segment_ps": segment_ps,

        # Final science result (Boera et al.)
        "k_bin": average_ps["k_bin"],
        "pk_bin": average_ps["pk_bin"],
        "pk_err": average_ps["pk_err"],
    }

# SECTION 2.5 : COMOVING-SPACE UTILITIES

# 2.5.1 Rebin a spectrum
def rebin_spectrum(wave, flux, error, factor=2):
    n = (len(flux) // factor) * factor
    wave = wave[:n]
    flux = flux[:n]
    error = error[:n]
    wave_rebin = wave.reshape(-1, factor).mean(axis=1)
    flux_rebin = flux.reshape(-1, factor).mean(axis=1)
    error_rebin = (np.sqrt(np.sum(error.reshape(-1, factor)**2,axis=1)) / factor)
    return (wave_rebin,flux_rebin,error_rebin)

# 2.5.2 Convert rolling window from cMpc to pixels
def rolling_window_pixels(window_cMpc, chi):    #Convert a physical window (h^-1 cMpc) into an equivalent number of pixels.
    spacing = mean_pixel_spacing(chi)
    window_pixels = int(np.round(window_cMpc / spacing))
    if window_pixels % 2 == 0:
        window_pixels += 1
    return max(3, window_pixels)

# 2.5.3 Compute comoving coordinate
def comoving_coordinate(wave):    #Comoving coordinate of each pixel in h^-1 cMpc.
    z = wave / 1215.67 - 1.0
    chi = cosmo.comoving_distance(z).value
    return chi * cosmo.h

# 2.5.4 Compute mean pixel spacing
def mean_pixel_spacing(chi): #Mean pixel spacing in h^-1 cMpc.
    return np.mean(np.diff(chi))

# 2.5.5 Split the Lyα forest into fixed comoving segments
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

# 2.5.6 Average segment power spectra
def average_segment_power_spectra(segment_ps):

    if len(segment_ps) == 0:
        return None

    k_bin = segment_ps[0]["k_bin"]

    # Shape: (n_segments, n_bins)
    pk = np.array([ps["pk_bin"] for ps in segment_ps])

    # Number of finite measurements contributing to each bin
    valid_counts = np.sum(np.isfinite(pk), axis=0)

    # Initialise outputs
    mean_pk = np.full(pk.shape[1], np.nan)
    std_pk = np.full(pk.shape[1], np.nan)

    # Mean: only where at least one segment contributes
    good_mean = valid_counts > 0
    if np.any(good_mean):
        mean_pk[good_mean] = np.nanmean(pk[:, good_mean],axis=0)

    # Standard deviation: only where at least two segments contribute
    good_std = valid_counts > 1
    if np.any(good_std):
        std_pk[good_std] = np.nanstd(pk[:, good_std],axis=0,ddof=1)

    # Total contributing modes
    total_modes = np.sum([ps["n_modes"] for ps in segment_ps],axis=0)
    
    return {
        "k_bin": k_bin,
        "pk_bin": mean_pk,
        "pk_err": std_pk,
        "n_modes": total_modes}

# 2.5.7 Group segments into redshift bins
def group_segments_by_redshift(all_segments, redshift_bins):
    grouped = {}
    for zmin, zmax in redshift_bins:
        key = (zmin, zmax)
        grouped[key] = [seg for seg in all_segments if zmin <= seg["z"] < zmax]
    return grouped

# SECTION 2.6 : VISUALIZATION UTILITIES

# 2.6.1 Plot download configuration
def plotly_download_config(quasar_name,graph_name):
    return {"toImageButtonOptions": {"format": "png","filename":f"{quasar_name}_{graph_name}","height": 800,"width": 1200,"scale": 2}}

# SECTION 3 : ANALYSIS PIPELINE

# SECTION 3.1 : USER INPUTS IN SIDEBAR

window_cMpc = st.sidebar.number_input("Rolling Mean Window (h⁻¹ cMpc)",min_value=10.0,max_value=100.0,value=40.0,step=5.0)
st.sidebar.caption("Notation: cMpc = comoving Mpc")

segment_length = st.sidebar.number_input("Segment Length (h⁻¹ cMpc)",min_value=5.0,max_value=50.0,value=10.0,step=1.0)

logk_min = st.sidebar.number_input("Minimum log10(k)",value=-2.2,step=0.1)
logk_max = st.sidebar.number_input("Maximum log10(k)",value=-0.7,step=0.1)
delta_logk = st.sidebar.number_input("Δlog10(k)",value=0.1,step=0.01)

K_BIN_EDGES = np.arange(logk_min,logk_max + delta_logk,delta_logk)

st.sidebar.markdown("---")
st.sidebar.markdown("### Cosmology Conversion")
st.sidebar.latex(r"\chi(z)=\int_0^z\frac{c\,dz'}{H(z')}")
st.sidebar.latex(r"\chi_{h^{-1}}=\chi\,h")
st.sidebar.latex(r"\Delta\chi=\chi_{i+1}-\chi_i")
st.sidebar.latex(r"N_{\rm pix}=\frac{L_{\rm cMpc}}{\langle\Delta\chi\rangle}")


# SECTION 3.2 : LOAD AND PREPARE SPECTRA


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

# Collect all 10 h^-1 cMpc segments from every quasar

all_segments = []
for spec in spectra:

    if spec["ps_lomb"] is None:
        continue

    all_segments.extend(spec["ps_lomb"]["segment_ps"])

REDSHIFT_BINS = [
    (4.2, 4.6),
    (4.6, 5.0),
    (5.0, 5.4),
    (5.4, 5.8)]

grouped_segments = group_segments_by_redshift(all_segments,REDSHIFT_BINS)

# Average power spectrum in each redshift bin
redshift_results = {}

for zbin, segments in grouped_segments.items():

    if len(segments) == 0:
        continue

    redshift_results[zbin] = average_segment_power_spectra(segments)




# SECTION 4 : RESULTS AND VISUALISATION

#-----section 4.0: Setup-------------------------
st.title("Power spectrum")

#-----section 4.1: combined result-------------------------
st.header("Combined Lyα Forest Power Spectrum")
st.caption("Average Lomb–Scargle power spectrum grouped by redshift.")

fig = go.Figure()

for (zmin, zmax), result in redshift_results.items():
    valid = (np.isfinite(result["pk_bin"]) & (result["pk_bin"] > 0))
    fig.add_trace(go.Scatter(
            x=np.log10(result["k_bin"][valid]),
            y=np.log10(result["k_bin"][valid] * result["pk_bin"][valid]/ np.pi),
            mode="markers+lines",
            name=f"{zmin:.1f} ≤ z < {zmax:.1f}",

            error_y=dict(type="data",array=(result["pk_err"][valid] / (result["pk_bin"][valid]* np.log(10))),visible=True,),))

fig.update_layout(title="Mean Lyα Forest Power Spectrum",
    xaxis_title="log₁₀(k / km⁻¹ s)",
    yaxis_title="log₁₀(kP(k)/π)",
    legend_title="Redshift bin")

st.plotly_chart(fig, width="stretch")

summary = []

for (zmin, zmax), segments in grouped_segments.items():
    summary.append({
        "Redshift bin": f"{zmin:.1f}–{zmax:.1f}",
        "Segments": len(segments),})

st.dataframe(pd.DataFrame(summary), width="stretch")

summary_df = pd.DataFrame(summary_rows)
st.header("Dataset Summary")
st.dataframe(summary_df,width='stretch')
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
                st.metric("Total Modes",int(np.sum(ps_fft["n_modes"])))
                st.metric("Mean Modes / Bin",f"{np.mean(ps_lomb['n_modes']):.1f}")

            with col2:
                st.write("")
                st.markdown("### Using Lomb-Scargle Periodogram")
                st.metric("Good Pixels",ps_lomb["n_good"])
                st.metric("Forest dv",f"{ps_lomb['dv_forest']:.2f}")
                st.metric("Segment Length", f"{spec['segment_length']:.1f} h⁻¹ cMpc")
                st.metric("Segments", len(ps_lomb["segment_ps"]))
                st.metric("Total Modes",int(np.sum(ps_lomb["n_modes"])))
                st.metric("Mean Modes / Bin",f"{np.mean(ps_lomb['n_modes']):.1f}")


            st.dataframe(pd.DataFrame({
                "log10(k)": np.log10(ps_lomb["k_bin"]),
                "Modes": ps_lomb["n_modes"]}))
            
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

                st.plotly_chart(fig_fft,width='stretch',
                    config=plotly_download_config(spec["object"],"FFTPowerSpectrum"))
            
                fig_lomb = go.Figure()
                                     
                # Raw spectrum
                fig_lomb.add_trace(go.Scatter(
                    x=np.log10(ps_lomb["k"]),
                    y=np.log10(ps_lomb["k"] * ps_lomb["pk"] / np.pi),
                    mode="lines",
                    name="Raw"
                ))

                # Noise spectrum
                fig_lomb.add_trace(go.Scatter(
                    x=np.log10(ps_lomb["k"]),
                    y=np.log10(ps_lomb["k"] * ps_lomb["noise_pk"] / np.pi),
                    mode="lines",
                    name="Noise",
                    line=dict(dash="dot")
                ))

                # Noise-corrected spectrum
                valid = ps_lomb["pk_corrected"] > 0

                fig_lomb.add_trace(go.Scatter(
                    x=np.log10(ps_lomb["k"][valid]),
                    y=np.log10(
                        ps_lomb["k"][valid]
                        * ps_lomb["pk_corrected"][valid]
                        / np.pi
                    ),
                    mode="lines",
                    name="Corrected"
                ))

                fig_lomb.update_layout(title=(f"Lomb-Scargle Power Spectrum (Window={window_pixels} px, Bins=20)"),
                    xaxis_title="log₁₀(k / km⁻¹ s)",
                    yaxis_title="log₁₀(kP(k)/π)")

                st.plotly_chart(fig_lomb,width='stretch',
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

                st.plotly_chart(fig_fft_bin,width='stretch',
                    config=plotly_download_config(spec["object"],"FFT_Binned_PowerSpectrum"))

                # ==========================
                # Lomb-Scargle Binned Power Spectrum
                # ==========================

                fig_lomb_bin = go.Figure()

                # Masks
                valid = np.isfinite(ps_lomb["pk_bin"])
                empty = ~valid

                # --------------------------
                # 1. Line trace (contains NaNs -> gaps remain)
                # --------------------------
                fig_lomb_bin.add_trace(
                    go.Scatter(
                        x=np.log10(ps_lomb["k_bin"]),
                        y=np.log10(ps_lomb["k_bin"] * ps_lomb["pk_bin"] / np.pi),
                        mode="lines",
                        line=dict(color="royalblue"),
                        hoverinfo="skip",
                        showlegend=False
                    )
                )

                # --------------------------
                # 2. Measured points
                # --------------------------
                fig_lomb_bin.add_trace(
                    go.Scatter(
                        x=np.log10(ps_lomb["k_bin"][valid]),
                        y=np.log10(
                            ps_lomb["k_bin"][valid]
                            * ps_lomb["pk_bin"][valid]
                            / np.pi
                        ),
                        mode="markers+text",
                        marker=dict(
                            size=8,
                            color="royalblue"
                        ),
                        text=[f"n={int(n)}" for n in ps_lomb["n_modes"][valid]],
                        textposition="top center",
                        textfont=dict(size=10),
                        error_y=dict(
                            type="data",
                            array=(
                                ps_lomb["pk_err"][valid]
                                / (ps_lomb["pk_bin"][valid] * np.log(10))
                            ),
                            visible=True
                        ),
                        name="Lomb Binned"
                    )
                )

                # --------------------------
                # 3. Empty bins (n = 0)
                # --------------------------
                y_cross = (np.nanmin(np.log10(ps_lomb["k_bin"][valid]* ps_lomb["pk_bin"][valid]/ np.pi)) - 0.15)

                fig_lomb_bin.add_trace(go.Scatter(
                        x=np.log10(ps_lomb["k_bin"][empty]),
                        y=np.full(np.sum(empty), y_cross),
                        mode="markers+text",
                        marker=dict(
                            symbol="x",
                            size=12,
                            color="gray",
                            line=dict(width=2)),
                        text=["n=0"] * np.sum(empty),
                        textposition="top center",
                        textfont=dict(size=10,color="gray"),
                        hovertemplate="No modes in this logarithmic k-bin<extra></extra>",
                        showlegend=False))

                fig_lomb_bin.update_layout(title=f"Lomb-Scargle Binned Power Spectrum (Window={window_pixels} px, Bins=20)",
                    xaxis_title="log₁₀(k / km⁻¹ s)",
                    yaxis_title="log₁₀(kP(k)/π)")

                st.plotly_chart(fig_lomb_bin,width='stretch',config=plotly_download_config(spec["object"],"LombScargle_binned_PowerSpectrum"))

                
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
                    title=(
                        f"FFT vs Lomb-Scargle (Boera Method)\n"
                        f"Rolling Mean = {window_cMpc:.0f} h⁻¹ cMpc, "
                        f"Segment Length = {segment_length:.0f} h⁻¹ cMpc"
                    ),
                    xaxis_title="log₁₀(k / km⁻¹ s)",
                    yaxis_title="log₁₀(kP(k)/π)")

                st.plotly_chart(fig_compare,width='stretch',
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

                st.plotly_chart(fig,width='stretch',config=plotly_download_config(spec["object"],"LargeScalePowerStability"))


            #------------------------------------------------------    
#------------------------------------------------------
            # Data Tables
            with st.expander("Data Table"):

                # ==========================
                # Raw Power Spectrum Tables
                # ==========================
                st.subheader("Raw Power Spectrum Tables")

                fft_raw_df = pd.DataFrame({
                    "log10(k)": np.log10(ps_fft["k"]),
                    "log10(kP(k)/π)": np.log10(
                        ps_fft["k"] * ps_fft["pk"] / np.pi
                    )
                })

                # Robust logarithms for Lomb spectra
                raw = np.full_like(ps_lomb["pk"], np.nan, dtype=float)
                noise = np.full_like(ps_lomb["noise_pk"], np.nan, dtype=float)
                corrected = np.full_like(ps_lomb["pk_corrected"], np.nan, dtype=float)

                raw_mask = ps_lomb["pk"] > 0
                noise_mask = ps_lomb["noise_pk"] > 0
                corrected_mask = ps_lomb["pk_corrected"] > 0

                raw[raw_mask] = np.log10(
                    ps_lomb["k"][raw_mask]
                    * ps_lomb["pk"][raw_mask]
                    / np.pi
                )

                noise[noise_mask] = np.log10(
                    ps_lomb["k"][noise_mask]
                    * ps_lomb["noise_pk"][noise_mask]
                    / np.pi
                )

                corrected[corrected_mask] = np.log10(
                    ps_lomb["k"][corrected_mask]
                    * ps_lomb["pk_corrected"][corrected_mask]
                    / np.pi
                )

                lomb_raw_df = pd.DataFrame({
                    "log10(k)": np.log10(ps_lomb["k"]),
                    "Raw": raw,
                    "Noise": noise,
                    "Corrected": corrected
                })

                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("### FFT Raw")
                    st.dataframe(
                        fft_raw_df,
                        height=300,
                        width='stretch'
                    )

                with col2:
                    st.markdown("### Lomb-Scargle Raw")
                    st.dataframe(
                        lomb_raw_df,
                        height=300,
                        width='stretch'
                    )

                # ==========================
                # Binned Power Spectrum Tables
                # ==========================
                st.subheader("Binned Power Spectrum Tables")

                fft_bin_df = pd.DataFrame({
                    "log10(k)": np.log10(ps_fft["k_bin"]),
                    "log10(kP(k)/π)": np.log10(
                        ps_fft["k_bin"] * ps_fft["pk_bin"] / np.pi
                    ),
                    "σ(P)": ps_fft["pk_err"]
                })

                binned = np.full_like(ps_lomb["pk_bin"], np.nan, dtype=float)

                valid = ps_lomb["pk_bin"] > 0

                binned[valid] = np.log10(
                    ps_lomb["k_bin"][valid]
                    * ps_lomb["pk_bin"][valid]
                    / np.pi
                )

                lomb_bin_df = pd.DataFrame({
                    "log10(k)": np.log10(ps_lomb["k_bin"]),
                    "log10(kP(k)/π)": binned,
                    "σ(P)": ps_lomb["pk_err"],
                    "Modes": ps_lomb["n_modes"]
                })

                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("### FFT Binned")
                    st.dataframe(
                        fft_bin_df,
                        width='stretch'
                    )

                with col2:
                    st.markdown("### Lomb-Scargle Binned")
                    st.dataframe(
                        lomb_bin_df,
                        width='stretch'
                    )

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
                    
                st.plotly_chart(fig_roll,width='stretch')

                # Plot B : Flux Contrast
                st.markdown("#### Rolling-Mean Normalized Flux Contrast")
                fig_delta = go.Figure()
                fig_delta.add_trace(go.Scatter(
                        x=ps_lomb["wave_rest"],
                        y=ps_lomb["deltaF"],
                        mode="lines",name="δF"))

                fig_delta.update_layout(title=f"δF = Flux / Rolling Mean - 1 (Window = {window_pixels} pixels)",xaxis_title="Rest Wavelength (Å)",yaxis_title="δF")
                st.plotly_chart(fig_delta,width='stretch')

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
            st.plotly_chart(fig1,width='stretch')
            
        #-----section 4.2.7:Signal-to-Noise Plot
        with st.expander("Signal to Noise Ratio",expanded=False):
            smask = np.isfinite(snr)
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=wave[smask],y=snr[smask],mode="lines",name="SNR"))

            fig2.update_layout(title="Signal-to-Noise Ratio",
                xaxis_title="Observed Wavelength (Å)",
                yaxis_title="S/N")
            st.plotly_chart(fig2,width='stretch')
        
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
    N = st.number_input("Number of points", value=5000)

    dv = st.number_input("dv (km/s)", value=2.5)

    sigma = st.number_input(r"$\sigma$", value=0.05)

    z_test = 5.0

    wave_rest = np.linspace(1040,1180,N)
    wave_obs = wave_rest * (1 + z_test)

    true_flux = np.ones(N)

    noise = np.random.normal(0, sigma, N)

    flux = true_flux + noise

    error = sigma * np.ones(N)

    velocity = np.arange(N) * dv
    deltaF = noise

    expected_power = sigma**2 * dv

    # FFT unit test
    fft_test = power_spectrum_fft(deltaF, dv)

    # Full Boera pipeline
    ps_lomb = lya_power_spectrum_lomb(
        wave_obs,
        flux,
        error,
        z_test,
        window_cMpc,
        segment_length
    )
    st.subheader("Validation Statistics")

    st.write(f"σ = {sigma:.4f}")

    st.write(f"Expected FFT <P(k)> = σ²Δv = {expected_power:.4f}")

    st.write(f"FFT Mean = {np.mean(fft_test['pk']):.4f}")

    st.write("")

    st.markdown("### Corrected Lomb Pipeline")

    st.write(f"Mean Raw Power = {np.mean(ps_lomb['pk']):.4f}")

    st.write(f"Mean Noise Power = {np.mean(ps_lomb['noise_pk']):.4f}")

    st.write(f"Mean Corrected Power = {np.mean(ps_lomb['pk_corrected']):.4f}")

    st.write(f"Mean Final Binned Power = {np.nanmean(ps_lomb['pk_bin']):.4f}")
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

        st.plotly_chart(fig, width='stretch')
        
    #---------------------------------------------------------
    # B) Binned P(k)
    #---------------------------------------------------------
    with st.expander("B) FFT vs Corrected Lomb", expanded=True):

        fig = go.Figure()

        # FFT
        fig.add_trace(
            go.Scatter(
                x=np.log10(fft_test["k_bin"]),
                y=np.log10(fft_test["k_bin"] * fft_test["pk_bin"] / np.pi),
                mode="markers+lines",
                name="FFT"
            )
        )

        valid = ps_lomb["pk_bin"] > 0

        fig.add_trace(
            go.Scatter(
                x=np.log10(ps_lomb["k_bin"][valid]),
                y=np.log10(
                    ps_lomb["k_bin"][valid]
                    * ps_lomb["pk_bin"][valid]
                    / np.pi
                ),
                mode="markers+lines",
                name="Corrected Lomb"
            )
        )

        fig.add_hline(
            y=np.log10(expected_power),
            line_dash="dash",
            annotation_text="σ²Δv"
        )

        fig.update_layout(
            title="FFT vs Corrected Lomb Pipeline",
            xaxis_title="log10(k)",
            yaxis_title="log10(kP(k)/π)"
        )

        st.plotly_chart(fig, width='stretch')

    with st.expander("C) Raw / Noise / Corrected Lomb", expanded=True):

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=np.log10(ps_lomb["k"]),
                y=np.log10(ps_lomb["k"] * ps_lomb["pk"] / np.pi),
                mode="lines",
                name="Raw"
            )
        )

        fig.add_trace(
            go.Scatter(
                x=np.log10(ps_lomb["k"]),
                y=np.log10(ps_lomb["k"] * ps_lomb["noise_pk"] / np.pi),
                mode="lines",
                line=dict(dash="dot"),
                name="Noise"
            )
        )

        valid = ps_lomb["pk_corrected"] > 0

        fig.add_trace(
            go.Scatter(
                x=np.log10(ps_lomb["k"][valid]),
                y=np.log10(
                    ps_lomb["k"][valid]
                    * ps_lomb["pk_corrected"][valid]
                    / np.pi
                ),
                mode="lines",
                name="Corrected"
            )
        )

        fig.update_layout(
            title="Corrected Lomb Pipeline",
            xaxis_title="log10(k)",
            yaxis_title="log10(kP(k)/π)"
        )

        st.plotly_chart(fig, width='stretch')

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
            # FFT plotting arrays
            valid_fft = (
                np.isfinite(fft_test["k_bin"])
                & np.isfinite(fft_test["pk_bin"])
                & (fft_test["pk_bin"] > 0)
            )

            fft_x = np.log10(
                fft_test["k_bin"][valid_fft]
            )

            fft_y = np.log10(
                fft_test["pk_bin"][valid_fft]
            )

            valid_fft_rebin = (
                np.isfinite(fft_rebin["k_bin"])
                & np.isfinite(fft_rebin["pk_bin"])
                & (fft_rebin["pk_bin"] > 0)
            )

            fft_rebin_x = np.log10(
                fft_rebin["k_bin"][valid_fft_rebin]
            )

            fft_rebin_y = np.log10(
                fft_rebin["pk_bin"][valid_fft_rebin]
            )
            wave_rest_rebin = wave_rest[:n_use].reshape(-1,b).mean(axis=1)

            wave_obs_rebin = wave_rest_rebin * (1 + z_test)

            flux_rebin = flux[:n_use].reshape(-1,b).mean(axis=1)

            error_rebin = (
                np.sqrt(
                    np.sum(error[:n_use].reshape(-1,b)**2, axis=1)
                ) / b
            )

            ps_lomb_rebin = lya_power_spectrum_lomb(
                wave_obs_rebin,
                flux_rebin,
                error_rebin,
                z_test,
                window_cMpc,
                segment_length
            )
            # Measured variances
            variance_original = np.var(deltaF)
            variance_rebin = np.var(deltaF_rebin)

            #-----------------------------------------------------
            # FFT statistics
            #-----------------------------------------------------

            fft_original_mean = np.mean(fft_test["pk"])
            fft_rebin_mean = np.mean(fft_rebin["pk"])

            fft_expected_rebin = fft_original_mean

            fft_ratio = fft_rebin_mean / fft_expected_rebin

            fft_percent_change = (
                100
                * (fft_rebin_mean - fft_original_mean)
                / fft_original_mean
            )

            #-----------------------------------------------------
            # Corrected Lomb statistics
            #-----------------------------------------------------

            lomb_original_mean = np.nanmean(ps_lomb["pk_bin"])

            lomb_rebin_mean = np.nanmean(ps_lomb_rebin["pk_bin"])

            lomb_expected_rebin = lomb_original_mean

            lomb_ratio = lomb_rebin_mean / lomb_expected_rebin

            lomb_percent_change = (
                100
                * (lomb_rebin_mean - lomb_original_mean)
                / lomb_original_mean
            )

            #-----------------------------------------------------
            # Plotting arrays
            #-----------------------------------------------------

            valid_original = (
                np.isfinite(ps_lomb["k_bin"])
                & np.isfinite(ps_lomb["pk_bin"])
                & (ps_lomb["pk_bin"] > 0)
            )

            lomb_x = np.log10(ps_lomb["k_bin"][valid_original])

            lomb_y = np.log10(
                ps_lomb["pk_bin"][valid_original]
            )

            valid_rebin = (
                np.isfinite(ps_lomb_rebin["k_bin"])
                & np.isfinite(ps_lomb_rebin["pk_bin"])
                & (ps_lomb_rebin["pk_bin"] > 0)
            )

            lomb_rebin_x = np.log10(
                ps_lomb_rebin["k_bin"][valid_rebin]
            )

            lomb_rebin_y = np.log10(
                ps_lomb_rebin["pk_bin"][valid_rebin]
            )

            #-----------------------------------------------------
            # Store results
            #-----------------------------------------------------

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

                "lomb_x": lomb_x,
                "lomb_y": lomb_y,

                "lomb_rebin_x": lomb_rebin_x,
                "lomb_rebin_y": lomb_rebin_y,
            }


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

            st.dataframe(rebin_stats,hide_index=True,width='stretch')

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
                x=result["lomb_x"],
                y=result["lomb_y"],
                mode="markers+lines",
                name="Corrected Lomb Original",
                line=dict(dash="dot")
            ))

            # Rebinned Lomb
            fig.add_trace(go.Scatter(
                    x=result["lomb_rebin_x"],y=result["lomb_rebin_y"],
                    mode="markers+lines",
                    name="Corrected Lomb Rebinned"))

            fig.add_hline(y=np.log10(expected_power),
                line_dash="dash",line_color="black",annotation_text="σ²Δv")

            fig.update_layout(
                title=("White Noise Power Spectrum Before and After Rebinning"),
                xaxis_title="log₁₀(k / km⁻¹ s)",
                yaxis_title="log₁₀(P(k))")

            st.plotly_chart(fig,width='stretch')
 
    #---------------------------------------------------------
    # D) White Noise Masking Test
    #---------------------------------------------------------
    with st.expander("D) White Noise Masking Test", expanded=False):

        st.markdown("""
    This test checks whether masking contiguous regions changes the
    final corrected Lomb–Scargle power spectrum.

    The complete Boera pipeline is run for

    • the original spectrum

    • the masked spectrum

    The corrected, binned spectra should agree within statistical scatter.
    """)

        with st.form("masking_form"):

            mask_fraction = st.slider(
                "Masked fraction",
                0.05,
                0.50,
                0.20,
                0.05
            )

            n_regions = st.number_input(
                "Number of masked regions",
                1,
                20,
                5
            )

            run_mask = st.form_submit_button("Run Masking Test")

        if run_mask:

            rng = np.random.default_rng(42)

            Npix = len(flux)

            n_mask = int(mask_fraction * Npix)

            mask = np.zeros(Npix,dtype=bool)

            section_edges = np.linspace(
                0,
                Npix,
                n_regions+1,
                dtype=int
            )

            base = n_mask // n_regions
            remainder = n_mask % n_regions

            sizes = np.full(n_regions,base)

            sizes[:remainder]+=1

            for i in range(n_regions):

                start_section = section_edges[i]
                stop_section = section_edges[i+1]

                gap = sizes[i]

                start = rng.integers(
                    start_section,
                    stop_section-gap+1
                )

                mask[start:start+gap]=True

            wave_mask = wave_obs[~mask]

            flux_mask = flux[~mask]

            error_mask = error[~mask]

            ps_mask = lya_power_spectrum_lomb(
                wave_mask,
                flux_mask,
                error_mask,
                z_test,
                window_cMpc,
                segment_length
            )

            ratio = (
                np.nanmean(ps_mask["pk_bin"])
                /
                np.nanmean(ps_lomb["pk_bin"])
            )

            st.write(f"Masked fraction = {mask_fraction:.2f}")

            st.write(f"Mean power ratio = {ratio:.3f}")

            fig = go.Figure()

            valid1 = ps_lomb["pk_bin"]>0

            valid2 = ps_mask["pk_bin"]>0

            fig.add_trace(
                go.Scatter(
                    x=np.log10(ps_lomb["k_bin"][valid1]),
                    y=np.log10(
                        ps_lomb["k_bin"][valid1]
                        *ps_lomb["pk_bin"][valid1]
                        /np.pi
                    ),
                    mode="lines",
                    name="Original"
                )
            )

            fig.add_trace(
                go.Scatter(
                    x=np.log10(ps_mask["k_bin"][valid2]),
                    y=np.log10(
                        ps_mask["k_bin"][valid2]
                        *ps_mask["pk_bin"][valid2]
                        /np.pi
                    ),
                    mode="lines",
                    name="Masked"
                )
            )

            fig.update_layout(
                title="Corrected Lomb Pipeline After Masking",
                xaxis_title="log10(k)",
                yaxis_title="log10(kP(k)/π)"
            )

            st.plotly_chart(fig,width='stretch')



    #---------------------------------------------------------
    # E) Monte Carlo Masking Validation
    #---------------------------------------------------------
    with st.expander("E) Monte Carlo Masking Validation",expanded=False):

        n_realizations = st.number_input(
            "Number of realizations",
            10,
            500,
            100,
            10
        )

        if st.button("Run Monte Carlo"):

            ratios=[]

            progress=st.progress(0)

            rng=np.random.default_rng(1234)

            for r in range(n_realizations):

                noise=rng.normal(
                    0,
                    sigma,
                    N
                )

                flux_mc=1+noise

                error_mc=sigma*np.ones(N)

                ps_original=lya_power_spectrum_lomb(
                    wave_obs,
                    flux_mc,
                    error_mc,
                    z_test,
                    window_cMpc,
                    segment_length
                )

                mask=np.ones(N,dtype=bool)

                n_remove=int(0.2*N)

                start=rng.integers(
                    0,
                    N-n_remove
                )

                mask[start:start+n_remove]=False

                ps_mask=lya_power_spectrum_lomb(
                    wave_obs[mask],
                    flux_mc[mask],
                    error_mc[mask],
                    z_test,
                    window_cMpc,
                    segment_length
                )

                ratios.append(
                    np.nanmean(ps_mask["pk_bin"])
                    /
                    np.nanmean(ps_original["pk_bin"])
                )

                progress.progress((r+1)/n_realizations)

            ratios=np.asarray(ratios)

            st.write(f"Mean ratio = {np.mean(ratios):.4f}")

            st.write(f"Std = {np.std(ratios,ddof=1):.4f}")

            fig=go.Figure()

            fig.add_trace(
                go.Histogram(
                    x=ratios,
                    nbinsx=25
                )
            )

            fig.add_vline(
                x=1,
                line_dash="dash",
                annotation_text="Ideal"
            )

            fig.update_layout(
                title="Monte Carlo Masking Test",
                xaxis_title="Masked / Original Mean Power",
                yaxis_title="Count"
            )

            st.plotly_chart(fig,width='stretch')
