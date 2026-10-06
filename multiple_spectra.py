
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

st.caption('version 13')
st.set_page_config(page_title=".fit file plots",layout="wide")

SPEC_DIR = "spec" #name of the folder
redshifts = {
    "J0306+1853_HIRES": 5.36,
    "J0957+0610_UVES": 5.17,
    "J0131-0321_HIRES": 5.12,
    "J1659+2709_HIRES": 5.32,
    "J1204-0021_HIRES": 5.09,
    "J0231-0728_HIRES": 5.42,
    "J2111-0156_HIRES": 4.89,
    "J0011+1446_HIRES": 4.94,
    "J1101+0531_UVES": 5.05,
    "J0741+2520_HIRES": 5.19,
    "J1425+0827_UVES": 4.95,
    "J1008-0212_UVES": 5.04,
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
            N = np.sum(m)
            std = np.std(pk[m], ddof=1)
            pk_err.append(std/np.sqrt(N))
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
def compute_k_ls(L_velocity, dv):
    """L_velocity : velocity length of the full section (km/s)
    dv : pixel spacing (km/s)
    """

    k_min = 2 * np.pi / L_velocity
    k_nyquist = np.pi / dv

    n_max = int(np.floor(k_nyquist / k_min))

    n = np.arange(1, n_max + 1)

    k = n * k_min

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

def power_spectrum_lomb_raw(velocity, deltaF, dv, L_velocity):

    k = compute_k_ls(L_velocity, dv)

    frequency = k / (2 * np.pi)

    ls = LombScargle(velocity,deltaF,normalization="psd")

    pk = ls.power(frequency)

    pk *= dv

    return {"k": k,"pk": pk}
# ----- Step 4 : Estimate the noise power spectrum using Monte Carlo realizations


noise_realisations=st.sidebar.number_input("n realisatios for noise",value=1)
def estimate_noise_periodogram(velocity,error,smooth,dv,L_velocity,n_realizations=noise_realisations):


    noise_pk = []
    good = (np.isfinite(error) & np.isfinite(smooth) & (smooth != 0))
    velocity = velocity[good]
    sigma = error[good] / smooth[good]

    for _ in range(n_realizations):
        deltaF_noise = np.random.normal(loc=0.0,scale=sigma)
        ps = power_spectrum_lomb_raw(velocity,deltaF_noise,dv,L_velocity)
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
    L_velocity_full = velocity[-1] - velocity[0] + dv_forest
    for indices in segments:
        
        velocity_section = velocity[indices]
        L_velocity = velocity_section[-1] - velocity_section[0] + dv_forest
        
        segment_good = good[indices]
        
        if np.sum(segment_good) < 10:
            continue

        velocity_seg = velocity[indices][segment_good]
        deltaF_seg = deltaF[indices][segment_good]
        error_seg = error_forest[indices][segment_good]
        smooth_seg = smooth[indices][segment_good]
        raw_ps_seg = power_spectrum_lomb_raw(velocity_seg,deltaF_seg,dv_forest,L_velocity)

        noise_ps_seg = estimate_noise_periodogram(velocity_seg,error_seg,smooth_seg,dv_forest,L_velocity)
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
    raw_ps = power_spectrum_lomb_raw(velocity[good],deltaF[good],dv_forest,L_velocity_full)
    noise_ps = estimate_noise_periodogram(velocity[good],error_forest[good],smooth[good],dv_forest,L_velocity_full)
    pk_corrected = raw_ps["pk"] - noise_ps["pk"]

    # Optional: useful for diagnostic plots
    k_bin, pk_bin, pk_err, n_modes = bin_power_spectrum(raw_ps["k"],pk_corrected)

    return {
        "wave_obs": wave_obs_forest,
        "wave_rest": wave_rest,
        "flux_forest": flux_forest,
        "smooth": smooth,
        "deltaF": deltaF,
        "chi_forest": chi_forest,
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
def split_into_segments(chi, segment_length=20.0):
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

segment_length = st.sidebar.number_input("Segment Length (h⁻¹ cMpc)",min_value=5.0,max_value=50.0,value=20.0,step=1.0)

logk_min = st.sidebar.number_input("Minimum log10(k)",value=-2.2,step=0.1)
logk_max = st.sidebar.number_input("Maximum log10(k)",value=-0.7,step=0.1)
delta_logk = st.sidebar.number_input("Δlog10(k)",value=0.1,step=0.01)


#K_BIN_EDGES = np.arange(logk_min,logk_max + delta_logk,delta_logk)



# 1. Compute the exact number of steps/bins needed
# We add a tiny epsilon (1e-9) to prevent rounding down due to float precision
num_steps = int(np.round((logk_max - logk_min) / delta_logk))

# 2. Number of edges is always number of steps + 1
num_edges = num_steps + 1

# 3. Reliably generate the array including the exact endpoint
#K_BIN_EDGES = np.linspace(logk_min, logk_max, num_edges)
K_BIN_CENTERS = np.arange(-2.2, -0.7 + 0.001, 0.1)

K_BIN_EDGES = np.concatenate([
    [K_BIN_CENTERS[0] - 0.05],
    0.5 * (K_BIN_CENTERS[:-1] + K_BIN_CENTERS[1:]),
    [K_BIN_CENTERS[-1] + 0.05]])

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
    (4.0, 4.4),
    (4.4, 4.8),
    (4.8, 5.2)]

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

showfft=st.checkbox("Show FFT")


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
        
        if (ps_lomb is not None):
            col1, col2 = st.columns(2)
            st.markdown("### Using Lomb-Scargle Periodogram")

            with col1:
                st.metric("Good Pixels",ps_lomb["n_good"])
                st.metric("Forest dv",f"{ps_lomb['dv_forest']:.2f}")
                st.metric("Segment Length", f"{spec['segment_length']:.1f} h⁻¹ cMpc")
                

            with col2:
                st.write("")
                st.metric("Segments", len(ps_lomb["segment_ps"]))
                st.metric("Total Modes",int(np.sum(ps_lomb["n_modes"])))
                st.metric("Mean Modes / Bin",f"{np.mean(ps_lomb['n_modes']):.1f}")


            st.dataframe(pd.DataFrame({
                "log10(k)": np.log10(ps_lomb["k_bin"]),
                "Modes": ps_lomb["n_modes"]}))
            
        #-----section 4.2.4: Lyα Power Spectrum Plot
        if ps_lomb is not None :

            # ----- Step 1: Observed Spectrum -----
            with st.expander("1. Observed Spectrum", expanded=False):

                fig_obs = go.Figure()

                fig_obs.add_trace(go.Scatter(
                        x=wave,
                        y=flux,
                        mode="lines",
                        name="Observed Flux",
                        error_y=dict(type="data",array=error,visible=True)))

                fig_obs.update_layout(
                    title="Observed Spectrum from FITS",
                    xaxis_title="Wavelength (Å)",
                    yaxis_title="Flux",
                    hovermode="x unified")

                st.plotly_chart(fig_obs,width="stretch",config=plotly_download_config(spec["object"],"Step_1_Observed_Spectrum"))


            # ----- Step 2: Lyα Forest + Rolling Mean -----
            with st.expander("2. Lyα Forest and Rolling Mean", expanded=False):

                wave_lya = ps_lomb["wave_obs"]
                flux_lya = ps_lomb["flux_forest"]
                smooth_lya = ps_lomb["smooth"]

                fig_lya = go.Figure()

                # Observed flux in Lyα forest
                fig_lya.add_trace(
                    go.Scatter(
                        x=wave_lya,
                        y=flux_lya,
                        mode="lines",
                        name="Observed Flux",
                        line=dict(width=1)))

                # Rolling mean
                fig_lya.add_trace(
                    go.Scatter(
                        x=wave_lya,
                        y=smooth_lya,
                        mode="lines",
                        name=f"Rolling Mean ({window_cMpc:.0f} $h^{{-1}}$ cMpc)",
                        line=dict(width=2)))

                fig_lya.update_layout(
                    title="Lyα Forest and Rolling Mean",
                    xaxis_title="Observed Wavelength (Å)",
                    yaxis_title="Flux",
                    hovermode="x unified")

                st.plotly_chart(
                    fig_lya,
                    width="stretch",
                    config=plotly_download_config(
                        spec["object"],
                        "Step_2_Lya_Forest_Rolling_Mean"))


            # ============================================================
            # STEP 3 : FLUX CONTRAST
            # ============================================================

            with st.expander("3. Flux Contrast", expanded=False):

                wave_lya = ps_lomb["wave_obs"]
                deltaF = ps_lomb["deltaF"]
                chi_lya = ps_lomb["chi_forest"]

                # Shift the comoving coordinate so the forest starts at 0
                chi_relative = chi_lya - chi_lya[0]

                fig_deltaF = go.Figure()

                fig_deltaF.add_trace(go.Scatter(
                        x=chi_relative,
                        y=deltaF,
                        mode="lines",
                        name="Flux Contrast",
                        line=dict(width=1)))

                # Reference line: deltaF = 0
                fig_deltaF.add_hline(
                    y=0,
                    line_dash="dash",
                    line_width=1)

                fig_deltaF.update_layout(
                    title="Flux Contrast in the Lyα Forest",
                    xaxis_title="Comoving Distance ($h^{-1}$ cMpc)",
                    yaxis_title=r"$\delta_F$",
                    hovermode="x unified")

                st.plotly_chart(fig_deltaF,width="stretch",config=plotly_download_config(spec["object"],"Step_3_Flux_Contrast"))


            # ============================================================
            # STEP 4 : LOMB-SCARGLE POWER SPECTRUM
            # ============================================================

            with st.expander("4. Noise Corrected Lomb–Scargle Power Spectrum", expanded=False):

                segment_ps = ps_lomb["segment_ps"]

                fig_ls = go.Figure()

                # --------------------------------------------------------
                # Individual segment spectra
                # --------------------------------------------------------
                for i, seg in enumerate(segment_ps):

                    k = seg["k_bin"]
                    pk = seg["pk_bin"]

                    valid = (np.isfinite(k)
                        & np.isfinite(pk)
                        & (k > 0)
                        & (pk > 0))

                    log_k = np.log10(k[valid])
                    log_kP = np.log10(k[valid] * pk[valid] / np.pi)

                    fig_ls.add_trace(
                        go.Scatter(
                            x=log_k,
                            y=log_kP,
                            mode="lines",
                            name=f"Segment {i+1}",
                            line=dict(width=1),
                            opacity=0.4,
                            showlegend=False))

                # --------------------------------------------------------
                # Averaged spectrum
                # --------------------------------------------------------
                k_avg = ps_lomb["k_bin"]
                pk_avg = ps_lomb["pk_bin"]

                valid_avg = (
                    np.isfinite(k_avg)
                    & np.isfinite(pk_avg)
                    & (k_avg > 0)
                    & (pk_avg > 0))

                log_k_avg = np.log10(k_avg[valid_avg])
                log_kP_avg = np.log10(k_avg[valid_avg] * pk_avg[valid_avg] / np.pi)

                fig_ls.add_trace(
                    go.Scatter(
                        x=log_k_avg,
                        y=log_kP_avg,
                        mode="lines+markers",
                        name="Average",
                        line=dict(width=3),
                        marker=dict(size=6)))

                fig_ls.update_layout(
                    title="Lomb–Scargle Flux Power Spectrum",
                    xaxis_title=r"$\log_{10}(k/\mathrm{km}^{-1}\,\mathrm{s})$",
                    yaxis_title=r"$\log_{10}[kP(k)/\pi]$",
                    hovermode="x unified")

                st.plotly_chart(
                    fig_ls,
                    width="stretch",
                    config=plotly_download_config(
                        spec["object"],
                        "Step_4_Lomb_Scargle_Power_Spectrum"))

            with st.expander("5. Noise Power Spectrum", expanded=False):

                k_noise = ps_lomb["k"]
                pk_noise = ps_lomb["noise_pk"]

                valid_noise = (
                    np.isfinite(k_noise)
                    & np.isfinite(pk_noise)
                    & (k_noise > 0)
                    & (pk_noise > 0)
                )

                fig_noise = go.Figure()

                fig_noise.add_trace(go.Scatter(
                    x=np.log10(k_noise[valid_noise]),
                    y=np.log10(
                        k_noise[valid_noise]
                        * pk_noise[valid_noise]
                        / np.pi
                    ),
                    mode="lines",
                    name="Noise Power",
                    line=dict(width=2)
                ))

                fig_noise.update_layout(
                    title="Estimated Noise Power Spectrum",
                    xaxis_title=r"$\log_{10}(k/\mathrm{km}^{-1}\,\mathrm{s})$",
                    yaxis_title=r"$\log_{10}[kP_N(k)/\pi]$",
                    hovermode="x unified"
                )

                st.plotly_chart(
                    fig_noise,
                    width="stretch",
                    config=plotly_download_config(
                        spec["object"],
                        "Step_5_Noise_Power_Spectrum"
                    )
                )


            with st.expander("6. Final Noise-Corrected Power Spectrum", expanded=False):

                k_final = ps_lomb["k_bin"]
                pk_final = ps_lomb["pk_bin"]
                pk_err = ps_lomb["pk_err"]

                valid_final = (
                    np.isfinite(k_final)
                    & np.isfinite(pk_final)
                    & (k_final > 0)
                    & (pk_final > 0)
                )

                log_k = np.log10(k_final[valid_final])
                log_kP = np.log10(
                    k_final[valid_final]
                    * pk_final[valid_final]
                    / np.pi
                )

                # Convert linear P(k) error to log10[kP/pi] error
                valid_err = (
                    valid_final
                    & np.isfinite(pk_err)
                    & (pk_err >= 0)
                )

                log_err = (
                    pk_err[valid_err]
                    / pk_final[valid_err]
                    / np.log(10)
                )

                fig_final = go.Figure()

                fig_final.add_trace(go.Scatter(
                    x=log_k,
                    y=log_kP,
                    mode="lines+markers",
                    name="Noise-corrected power",
                    error_y=dict(
                        type="data",
                        array=log_err,
                        visible=True
                    ),
                    line=dict(width=3),
                    marker=dict(size=6)
                ))

                fig_final.update_layout(
                    title="Final Noise-Corrected Lomb–Scargle Power Spectrum",
                    xaxis_title=r"$\log_{10}(k/\mathrm{km}^{-1}\,\mathrm{s})$",
                    yaxis_title=r"$\log_{10}[kP(k)/\pi]$",
                    hovermode="x unified"
                )

                st.plotly_chart(
                    fig_final,
                    width="stretch",
                    config=plotly_download_config(
                        spec["object"],
                        "Step_6_Final_Power_Spectrum"
                    )
                )












#-----section 4.2.4: Rolling Mean Normalization
        if ps_lomb is not None:

            with st.expander("1. Rolling Mean Normalization",expanded=True):


                fig_roll = go.Figure()

                fig_roll.add_trace(
                    go.Scatter(
                        x=ps_lomb["wave_rest"],
                        y=ps_lomb["flux_forest"],
                        mode="lines",
                        name="Forest Flux"))

                fig_roll.add_trace(
                    go.Scatter(
                        x=ps_lomb["wave_rest"],
                        y=ps_lomb["smooth"],
                        mode="lines",
                        name=f"Rolling Mean ({window_pixels} px)"))

                fig_roll.update_layout(
                    title=(
                        f"Lyα Forest Flux and Rolling Mean "
                        f"(Window = {window_cMpc:.1f} h⁻¹ cMpc)"),
                    xaxis_title="Rest Wavelength (Å)",
                    yaxis_title="Flux")

                st.plotly_chart(
                    fig_roll,
                    width="stretch",
                    config=plotly_download_config(spec["object"],"RollingMeanNormalization"))

                st.markdown("#### Flux Contrast")

                fig_delta = go.Figure()

                fig_delta.add_trace(
                    go.Scatter(
                        x=ps_lomb["wave_rest"],
                        y=ps_lomb["deltaF"],
                        mode="lines",
                        name="δF"
                    )
                )

                fig_delta.add_hline(
                    y=0,
                    line_dash="dash",
                    annotation_text="δF = 0"
                )

                fig_delta.update_layout(
                    title="Rolling-Mean Normalized Flux Contrast",
                    xaxis_title="Rest Wavelength (Å)",
                    yaxis_title="δF")

                st.plotly_chart(
                    fig_delta,
                    width="stretch",
                    config=plotly_download_config(spec["object"],"FluxContrast"))



       


            #B) Binned Power Spectra
            with st.expander("B) Binned Power Spectra", expanded=True):
                if showfft==True: 
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
                        showlegend=False))

                # --------------------------
                # 2. Measured points
                # --------------------------
                fig_lomb_bin.add_trace(
                    go.Scatter(
                        x=np.log10(ps_lomb["k_bin"][valid]),
                        y=np.log10(
                            ps_lomb["k_bin"][valid]* ps_lomb["pk_bin"][valid]/ np.pi),
                        mode="markers+text",
                        marker=dict(size=8,color="royalblue"),
                        text=[f"n={int(n)}" for n in ps_lomb["n_modes"][valid]],
                        textposition="top center",
                        textfont=dict(size=10),
                        error_y=dict(
                            type="data",
                            array=(ps_lomb["pk_err"][valid]/ (ps_lomb["pk_bin"][valid] * np.log(10))),
                            visible=True),name="Lomb Binned"))

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

                
            

            # D) Large-scale Power Stability Test
            with st.expander("D) Large-scale Power Stability Test"):

                wave2, flux2, error2 = rebin_spectrum(wave,flux,error,factor=2)
                ps_lomb2 = lya_power_spectrum_lomb(wave2,flux2,error2,
                    spec["z"],spec["window_cMpc"],segment_length)
                fig = go.Figure()

           

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


