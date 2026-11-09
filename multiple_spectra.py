
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
import pickle


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


#noise_realisations=st.sidebar.number_input("n realisatios for noise",value=1)

#def estimate_noise_periodogram(velocity,error,smooth,dv,L_velocity,n_realizations=noise_realisations):
 #   noise_pk = []
  #  good = (np.isfinite(error) & np.isfinite(smooth) & (smooth != 0))
   # velocity = velocity[good]
    #sigma = error[good] / smooth[good]

    #for _ in range(n_realizations):
     #   deltaF_noise = np.random.normal(loc=0.0,scale=sigma)
      #  ps = power_spectrum_lomb_raw(velocity,deltaF_noise,dv,L_velocity)
       # noise_pk.append(ps["pk"])

  #  noise_pk = np.asarray(noise_pk)
   # return {
    #    "k": ps["k"],
     #   "pk": np.mean(noise_pk, axis=0)}

def estimate_noise_high_k(raw_ps, logk_noise_min=-0.2):
    k = raw_ps["k"]
    pk = raw_ps["pk"]

    valid = np.isfinite(k) & np.isfinite(pk) & (k > 0)
    high_k = valid & (np.log10(k) >= logk_noise_min)

    if np.sum(high_k) < 2:
        P_noise = np.nan
    else:
        # Current approximation: constant noise floor from the mean high-k power.
        P_noise = float(np.mean(pk[high_k]))

    noise_pk = np.full_like(pk, P_noise, dtype=float)

    return {
        "k": k,
        "pk": noise_pk,
        "P_noise": P_noise,
        "high_k_mask": high_k,
        "n_high_k": int(np.sum(high_k)),
        "logk_noise_min": float(logk_noise_min),
    }

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

        #noise_ps_seg = estimate_noise_periodogram(velocity_seg,error_seg,smooth_seg,dv_forest,L_velocity)
        noise_ps_seg = estimate_noise_high_k(raw_ps_seg)
        pk_corrected = raw_ps_seg["pk"] - noise_ps_seg["pk"]

        # Binned diagnostic stages. Keep signed values after noise subtraction.
        k_debug, pk_raw_bin, pk_raw_err, n_raw_modes = bin_power_for_debug(
            raw_ps_seg["k"], raw_ps_seg["pk"])
        _, pk_noise_bin, _, n_noise_modes = bin_power_for_debug(
            raw_ps_seg["k"], noise_ps_seg["pk"])
        _, pk_sub_bin, pk_sub_err, n_sub_modes = bin_power_for_debug(
            raw_ps_seg["k"], pk_corrected)

        # Science spectrum retains the existing positive-power binning.
        k_bin, pk_bin, pk_err, n_modes = bin_power_spectrum(
            raw_ps_seg["k"], pk_corrected)

        wave_center = np.mean(wave_obs_forest[indices])
        segment_z = wave_center / 1215.67 - 1.0
        segment_ps.append({
            "z": segment_z,
            "k_bin": k_bin,
            "pk_bin": pk_bin,
            "pk_err": pk_err,
            "n_modes": n_modes,
            # Diagnostic arrays, all binned on the same science k bins:
            "k_debug_bin": k_debug,
            "pk_raw_bin": pk_raw_bin,
            "pk_raw_err_bin": pk_raw_err,
            "pk_noise_bin": pk_noise_bin,
            "pk_sub_bin": pk_sub_bin,
            "pk_sub_err_bin": pk_sub_err,
            "n_raw_modes": n_raw_modes,
            "n_noise_modes": n_noise_modes,
            "n_sub_modes": n_sub_modes,
            "P_noise": noise_ps_seg["P_noise"],
            "n_high_k": noise_ps_seg["n_high_k"],
            "logk_noise_min": noise_ps_seg["logk_noise_min"],
        })

    average_ps = average_segment_power_spectra(segment_ps)

    if average_ps is None:
        return None

    # --------------------------------------------------
    # 5) Full forest (diagnostic only)
    # --------------------------------------------------
    raw_ps = power_spectrum_lomb_raw(velocity[good],deltaF[good],dv_forest,L_velocity_full)
    #noise_ps = estimate_noise_periodogram(velocity[good],error_forest[good],smooth[good],dv_forest,L_velocity_full)
    noise_ps= estimate_noise_high_k(raw_ps)
    pk_corrected = raw_ps["pk"] - noise_ps["pk"]

    # Binned full-forest diagnostic stages (not used as the science result).
    k_debug, pk_raw_bin, pk_raw_err_bin, n_raw_modes = bin_power_for_debug(
        raw_ps["k"], raw_ps["pk"])
    _, pk_noise_bin, _, n_noise_modes = bin_power_for_debug(
        raw_ps["k"], noise_ps["pk"])
    _, pk_sub_bin, pk_sub_err_bin, n_sub_modes = bin_power_for_debug(
        raw_ps["k"], pk_corrected)

    k_bin, pk_bin, pk_err, n_modes = bin_power_spectrum(
        raw_ps["k"], pk_corrected)

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
        "P_noise": noise_ps["P_noise"],
        "n_high_k": noise_ps["n_high_k"],
        "logk_noise_min": noise_ps["logk_noise_min"],
        "k_debug_bin": k_debug,
        "pk_raw_bin": pk_raw_bin,
        "pk_raw_err_bin": pk_raw_err_bin,
        "pk_noise_bin": pk_noise_bin,
        "pk_sub_bin": pk_sub_bin,
        "pk_sub_err_bin": pk_sub_err_bin,
        "n_raw_modes": n_raw_modes,
        "n_noise_modes": n_noise_modes,
        "n_sub_modes": n_sub_modes,

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
        #std_pk[good_std] = np.nanstd(pk[:, good_std],axis=0,ddof=1)

        std_pk[good_std] = (np.nanstd(pk[:, good_std],axis=0,ddof=1)/ np.sqrt(valid_counts[good_std]))


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

# SECTION 2.7 : THE PLOTS STUFF 

def bin_power_for_debug(k, pk):
    """
    Bin power in linear units while retaining negative noise-subtracted values.
    This is for diagnostics; the existing science binning function is unchanged.
    """
    k = np.asarray(k, dtype=float)
    pk = np.asarray(pk, dtype=float)
    n_bins = len(K_BIN_EDGES) - 1

    k_bin = 10 ** (0.5 * (K_BIN_EDGES[:-1] + K_BIN_EDGES[1:]))
    pk_bin = np.full(n_bins, np.nan)
    pk_err = np.full(n_bins, np.nan)
    n_modes = np.zeros(n_bins, dtype=int)

    valid_k = np.isfinite(k) & (k > 0)
    logk = np.full(k.shape, np.nan)
    logk[valid_k] = np.log10(k[valid_k])

    for i in range(n_bins):
        m = (
            valid_k
            & np.isfinite(pk)
            & (logk >= K_BIN_EDGES[i])
            & (logk < K_BIN_EDGES[i + 1])
        )
        values = pk[m]
        n_modes[i] = values.size

        if values.size == 0:
            continue

        pk_bin[i] = np.mean(values)
        if values.size > 1:
            pk_err[i] = np.std(values, ddof=1) / np.sqrt(values.size)

    return k_bin, pk_bin, pk_err, n_modes



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
            #"header": header,
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





fig2 = go.Figure()

for (zmin, zmax), result in redshift_results.items():
    valid = (np.isfinite(result["pk_bin"]) & (result["pk_bin"] > 0))
    fig2.add_trace(go.Scatter(
            x=np.log10(result["k_bin"][valid]),
            y=np.log10(result["k_bin"][valid] * result["pk_bin"][valid]/ (0.76*np.pi)),
            mode="markers+lines",
            name=f"{zmin:.1f} ≤ z < {zmax:.1f}",

            error_y=dict(type="data",array=(result["pk_err"][valid] / (result["pk_bin"][valid]* np.log(10))),visible=True,),))

fig2.update_layout(title="Mean Lyα Forest Power Spectrum 2",
    xaxis_title="log₁₀(k / km⁻¹ s)",
    yaxis_title="log₁₀(kP(k)/π)",
    legend_title="Redshift bin")

st.plotly_chart(fig2, width="stretch")






summary = []

for (zmin, zmax), segments in grouped_segments.items():
    summary.append({
        "Redshift bin": f"{zmin:.1f}–{zmax:.1f}",
        "Segments": len(segments),})

st.dataframe(pd.DataFrame(summary), width="stretch")

summary_df = pd.DataFrame(summary_rows)
st.header("Dataset Summary")
st.dataframe(summary_df,width='stretch')




#showfft=st.checkbox("Show FFT")


#st.header("Individual Spectra")


spectra_bytes = pickle.dumps(

    spectra,

    protocol=pickle.HIGHEST_PROTOCOL)

st.download_button(

    label="Download spectra dictionary",

    data=spectra_bytes,

    file_name="spectra.pkl",

    mime="application/octet-stream",

    key="download_spectra_dictionary")






#________________________
#________________________
#________________________
#________________________


# SECTION 4 : RESULTS AND VISUALISATION

#-----section 4.0: Setup-------------------------
st.title("Power spectrum")

#-----section 4.1: combined result-------------------------

fig = go.Figure()

for (zmin, zmax), result in redshift_results.items():
    valid = (np.isfinite(result["pk_bin"]) & (result["pk_bin"] > 0))
    fig.add_trace(go.Scatter(
            x=np.log10(result["k_bin"][valid]),
            y=np.log10(result["k_bin"][valid] * result["pk_bin"][valid]/ np.pi),
            mode="markers+lines",
            name=f"{zmin:.1f} ≤ z < {zmax:.1f}",

            error_y=dict(type="data",array=(result["pk_err"][valid] / (result["pk_bin"][valid]* np.log(10))),visible=True,),))






labels = [
    f"{i}: {spec['object']} (z = {spec['z']})"
    for i, spec in enumerate(spectra)
]

selected_label = st.selectbox("Select a quasar", labels)

st.header(f"Plots for {selected_label}")
selected_index = labels.index(selected_label)
spec = spectra[selected_index]

#-----section 4.2.1: Extract Stored Data
flux = spec["flux"]
error = spec["error"]
wave = spec["wave"]
snr = spec["snr"]
ps_fft = spec["ps_fft"]

window_cMpc = spec.get("window_cMpc", window_cMpc if "window_cMpc" in locals() else 40.0)
ps_lomb = lya_power_spectrum_lomb(wave, flux, error, spec["z"], window_cMpc, segment_length)

if ps_lomb is not None:
    window_pixels = spec.get("window_pixels", np.nan)

with st.expander("Graph 1A:noise subtraction (bin orignal)",expanded=False):
    k = np.asarray(ps_lomb["k"], dtype=float)
    pk_raw = np.asarray(ps_lomb["pk"], dtype=float)
    pk_noise = np.asarray(ps_lomb["noise_pk"], dtype=float)
    P_noise = float(np.nanmedian(pk_noise))


    st.write("### Noise Subtraction Diagnostics")

    # Retrieve the actual arrays from ps_lomb
    k = ps_lomb["k"]
    pk_original = ps_lomb["pk"]
    pk_corrected = ps_lomb["pk_corrected"]
    noise_pk = ps_lomb["noise_pk"]

    st.write("Raw k range:", float(np.nanmin(k)), float(np.nanmax(k)))
    st.write("Total raw k points:", len(k))

    original_valid = ()
    corrected_valid = ()

    # Use the diagnostic bins already calculated in the pipeline
    k_bin = np.asarray(ps_lomb["k_debug_bin"], dtype=float)
    pk_raw_bin = np.asarray(ps_lomb["pk_raw_bin"], dtype=float)
    pk_noise_bin = np.asarray(ps_lomb["pk_noise_bin"], dtype=float)
    pk_corrected_bin = np.asarray(ps_lomb["pk_sub_bin"], dtype=float)
    # Keep only positive powers for the logarithmic y-axis.
    raw_ok = (k_bin > 0) & (pk_raw_bin > 0)
    corrected_ok = (k_bin > 0) & (pk_corrected_bin > 0)

    logk_raw = np.log10(k_bin[raw_ok])
    logk_corrected = np.log10(k_bin[corrected_ok])

    fig_noise_subtraction = go.Figure()

    # Original binned power: black
    fig_noise_subtraction.add_trace(go.Scatter(
        x=logk_raw,
        y=np.log10(pk_raw_bin[raw_ok]),
        mode="lines",
        name="Original",
        line=dict(color="black", width=2)))

    # Corrected binned power: green
    fig_noise_subtraction.add_trace(go.Scatter(
        x=logk_corrected,
        y=np.log10(pk_corrected_bin[corrected_ok]),
        mode="lines",
        name="Corrected",
        line=dict(color="green", width=2)))

    # Constant noise level: red dashed
    fig_noise_subtraction.add_hline(
        y=np.log10(P_noise),
        line_color="red",
        line_width=2,
        line_dash="dash",
        annotation_text="",)

    # Add a legend entry for the horizontal noise line.
    fig_noise_subtraction.add_trace(go.Scatter(
        x=[None],
        y=[None],
        mode="lines",
        name="Noise level",
        line=dict(color="red", width=2, dash="dash")))

    fig_noise_subtraction.update_layout(
        title="Noise subtraction ",
        xaxis_title="log₁₀(k / km⁻¹ s)",
        yaxis_title="log₁₀(P(k))",
        hovermode="x unified",)

    st.plotly_chart(fig_noise_subtraction,width="stretch",config=plotly_download_config(spec["object"],"Boera_Figure5_Noise_Subtraction"))




with st.expander("Graph 1B: Noise subtraction (binned extended)", expanded=True):


    # Use precomputed values from spectra.pkl
    k = np.asarray(ps_lomb["k"], dtype=float)
    pk_original = np.asarray(ps_lomb["pk"], dtype=float)
    pk_corrected = np.asarray(ps_lomb["pk_corrected"], dtype=float)
    noise_pk = np.asarray(ps_lomb["noise_pk"], dtype=float)

    # --------------------------------------------------
    # 1. Select valid original power and noise values
    # --------------------------------------------------
    valid_raw = (
        np.isfinite(k) &
        (k > 0) &
        np.isfinite(pk_original) &
        (pk_original > 0)
    )

    logk_raw = np.log10(k[valid_raw])
    pk_raw = pk_original[valid_raw]

    valid_noise = np.isfinite(noise_pk) & (noise_pk > 0)

    if not np.any(valid_noise):
        st.warning("No valid stored noise estimates are available.")
    else:
        P_noise = float(np.median(noise_pk[valid_noise]))

        # --------------------------------------------------
        # 2. Common bin edges, extending to high k
        # --------------------------------------------------
        bin_edges = np.arange(-2.25, 0.151, 0.1)

        bin_centres = []
        bin_original = []
        bin_corrected = []

        for left, right in zip(bin_edges[:-1], bin_edges[1:]):
            mask = (
                (logk_raw >= left) &
                (logk_raw < right)
            )

            if not np.any(mask):
                continue

            # Arithmetic mean in LINEAR power units
            P_original_bin = np.mean(pk_raw[mask])

            # Subtract noise in linear units
            P_corrected_bin = P_original_bin - P_noise

            bin_centres.append(0.5 * (left + right))
            bin_original.append(P_original_bin)
            bin_corrected.append(P_corrected_bin)

        bin_centres = np.asarray(bin_centres)
        bin_original = np.asarray(bin_original)
        bin_corrected = np.asarray(bin_corrected)

        # --------------------------------------------------
        # 3. Plot both spectra using identical bins
        # --------------------------------------------------
        fig_noise_subtraction = go.Figure()

        # Original spectrum
        positive_original = bin_original > 0

        fig_noise_subtraction.add_trace(go.Scatter(
            x=bin_centres[positive_original],
            y=np.log10(bin_original[positive_original]),
            mode="lines+markers",
            name="Original P(k)",
            line=dict(color="black", width=1.5),
            marker=dict(size=5)
        ))

        # Noise-subtracted spectrum.
        # Non-positive values cannot be shown on a log scale.
        positive_corrected = bin_corrected > 0

        fig_noise_subtraction.add_trace(go.Scatter(
            x=bin_centres[positive_corrected],
            y=np.log10(bin_corrected[positive_corrected]),
            mode="lines+markers",
            name="Noise-subtracted P(k)",
            line=dict(color="green", width=1.5),
            marker=dict(size=5)
        ))

        # Constant stored noise level
        fig_noise_subtraction.add_hline(
            y=np.log10(P_noise),
            line_color="red",
            line_width=2,
            line_dash="dash",
            annotation_text="Noise level",
            annotation_position="top left"
        )

        # --------------------------------------------------
        # 4. Format the graph
        # --------------------------------------------------
        fig_noise_subtraction.update_layout(
            title="Original Power Spectrum and Noise Subtraction",
            xaxis_title="log₁₀(k / km⁻¹ s)",
            yaxis_title="log₁₀(P(k))",
            hovermode="x unified",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="left",
                x=0
            )
        )

        fig_noise_subtraction.update_xaxes(
            range=[-2.25, 0.15]
        )

        st.plotly_chart(
            fig_noise_subtraction,
            width="stretch",
            config=plotly_download_config(
                spec["object"],
                "Boera_Figure5_Noise_Subtraction"
            )
        )






with st.expander("Graph 2: Flux vs wavelength with rolling mean",expanded=True):
    ps_lomb = spec["ps_lomb"]

    # Observed-frame Lyα forest data and stored rolling mean
    wave = np.asarray(ps_lomb["wave_obs"], dtype=float)
    flux = np.asarray(ps_lomb["flux_forest"], dtype=float)
    rolling_mean = np.asarray(ps_lomb["smooth"], dtype=float)

    fig_flux = go.Figure()

    fig_flux.add_trace(go.Scatter(
        x=wave,
        y=flux,
        mode="lines",
        name="Flux",
        line=dict(color="black", width=1)))

    fig_flux.add_trace(go.Scatter(
        x=wave,
        y=rolling_mean,
        mode="lines",
        name="Rolling mean (40 h⁻¹ cMpc)",
        line=dict(color="red", width=2)))

    fig_flux.update_layout(
        title="Lyα Forest Flux and Rolling Mean",
        xaxis=dict(
            title="Observed wavelength (Å)",
            type="log"),
        yaxis_title="Flux",
        hovermode="x unified")

    st.plotly_chart(fig_flux,
        width="stretch",
        config=plotly_download_config(
            spec["object"],
            "Flux_vs_Wavelength_Rolling_Mean"))









with st.expander("Graph 3: Power Spectrum by Segment",expanded=True):
    segment_ps = ps_lomb["segment_ps"]

    if len(segment_ps) == 0:
        st.info("No segments available for this quasar.")

    else:
        n_segments = len(segment_ps)
        ncols = 3
        nrows = (n_segments + ncols - 1) // ncols

        fig_segments = make_subplots(
            rows=nrows,
            cols=ncols,
            subplot_titles=[
                f"Segment {i + 1}"
                for i in range(n_segments)
            ],
            horizontal_spacing=0.08,
            vertical_spacing=0.12
        )

        for i, seg in enumerate(segment_ps):
            row = i // ncols + 1
            col = i % ncols + 1

            k = np.asarray(seg["k_bin"], dtype=float)
            pk = np.asarray(seg["pk_bin"], dtype=float)

            valid = (
                np.isfinite(k)
                & np.isfinite(pk)
                & (k > 0)
                & (pk > 0)
            )

            log_k = np.log10(k[valid])
            log_kP = np.log10(k[valid] * pk[valid] / np.pi)

            fig_segments.add_trace(
                go.Scatter(
                    x=log_k,
                    y=log_kP,
                    mode="lines+markers",
                    name=f"Segment {i + 1}",
                    marker=dict(size=4),
                    line=dict(width=1.5),
                    showlegend=False
                ),
                row=row,
                col=col
            )

        fig_segments.update_xaxes(title_text="log₁₀(k / (km⁻¹ s))",matches="x")

        fig_segments.update_yaxes(title_text="log₁₀[kP(k) / π]",matches="y")

        fig_segments.update_layout(
            title=f"{spec['object']}: Power Spectrum by Segment",
            height=300 * nrows,
            showlegend=False,
            hovermode="closest"
        )

        st.plotly_chart(
            fig_segments,
            width="stretch",
            config=plotly_download_config(
                spec["object"],
                "Lomb_Scargle_Power_Spectrum_By_Segment"
            )
        )


# ============================================================
# OVERALL POWER SPECTRUM FOR THIS QUASAR
# ============================================================

with st.expander(" Quasar Power Spectrum",expanded=True):
    k_avg = np.asarray(ps_lomb["k_bin"], dtype=float)
    pk_avg = np.asarray(ps_lomb["pk_bin"], dtype=float)
    pk_err = np.asarray(ps_lomb["pk_err"], dtype=float)

    valid = (
        np.isfinite(k_avg)
        & np.isfinite(pk_avg)
        & (k_avg > 0)
        & (pk_avg > 0)
    )

    fig_overall = go.Figure()

    fig_overall.add_trace(
        go.Scatter(
            x=np.log10(k_avg[valid]),
            y=np.log10(k_avg[valid] * pk_avg[valid] / np.pi),
            error_y=dict(
                type="data",
                array=(
                    pk_err[valid]
                    / (pk_avg[valid] * np.log(10))
                ),
                visible=True
            ),
            mode="lines+markers",
            name="Overall spectrum",
            line=dict(color="black", width=2),
            marker=dict(size=6)
        )
    )

    fig_overall.update_layout(
        title=f"{spec['object']}: Overall Lyα Flux Power Spectrum",
        xaxis_title="log₁₀(k / (km⁻¹ s))",
        yaxis_title="log₁₀[kP(k) / π]",
        hovermode="x unified"
    )

    st.plotly_chart(
        fig_overall,
        width="stretch",
        config=plotly_download_config(
            spec["object"],
            "Overall_Quasar_Power_Spectrum"
        )
    )


#---------------------------------



# Collect segments and preserve their parent quasar names

all_segments = []

for spec in spectra:
    if spec["ps_lomb"] is None:
        continue

    for i, seg in enumerate(spec["ps_lomb"]["segment_ps"]):
        segment = seg.copy()
        segment["quasar"] = spec["object"]
        segment["segment_number"] = i + 1
        all_segments.append(segment)

REDSHIFT_BINS = [
    (4.0, 4.4),
    (4.4, 4.8),
    (4.8, 5.2),
]

grouped_segments = group_segments_by_redshift(
    all_segments, REDSHIFT_BINS
)

redshift_results = {}

for zbin, segments in grouped_segments.items():
    if len(segments) == 0:
        continue

    redshift_results[zbin] = average_segment_power_spectra(
        segments
    )


# ============================================================
# PLOT 1: FINAL POWER SPECTRUM FOR EACH REDSHIFT BIN
# ============================================================

st.header("Final Power Spectrum by Redshift")

fig_final = go.Figure()

for (zmin, zmax), result in redshift_results.items():
    k = result["k_bin"]
    pk = result["pk_bin"]
    err = result["pk_err"]

    valid = (
        np.isfinite(k)
        & np.isfinite(pk)
        & (k > 0)
        & (pk > 0)
    )

    # Convert the uncertainty in P to uncertainty in log10(kP/pi)
    valid_err = np.where(
        np.isfinite(err[valid]) & (err[valid] >= 0),
        err[valid] / (pk[valid] * np.log(10)),
        np.nan,
    )

    fig_final.add_trace(go.Scatter(
        x=np.log10(k[valid]),
        y=np.log10(k[valid] * pk[valid] / np.pi),
        mode="lines+markers",
        name=f"{zmin:.1f} ≤ z < {zmax:.1f}",
        error_y=dict(
            type="data",
            array=valid_err,
            visible=True,
        ),
    ))

fig_final.update_layout(
    title="Mean Lyα Forest Power Spectrum",
    xaxis_title="log₁₀(k / (km⁻¹ s))",
    yaxis_title="log₁₀[kP(k) / π]",
    legend_title="Redshift bin",
)

st.plotly_chart(fig_final, width="stretch")


# ============================================================
# PLOT 2: INDIVIDUAL SEGMENT CONTRIBUTIONS BY REDSHIFT
# ============================================================

st.header("Segment Contributions to Each Redshift Bin")

for (zmin, zmax), segments in grouped_segments.items():

    if len(segments) == 0:
        st.info(f"No segments in {zmin:.1f} ≤ z < {zmax:.1f}")
        continue

    result = redshift_results[(zmin, zmax)]

    fig_segments = go.Figure()

    # Individual segment spectra
    for seg in segments:
        k = seg["k_bin"]
        pk = seg["pk_bin"]

        valid = (
            np.isfinite(k)
            & np.isfinite(pk)
            & (k > 0)
            & (pk > 0)
        )

        fig_segments.add_trace(go.Scatter(
            x=np.log10(k[valid]),
            y=np.log10(k[valid] * pk[valid] / np.pi),
            mode="lines",
            name=(
                f"{seg['quasar']} — "
                f"segment {seg['segment_number']}"
            ),
            line=dict(width=1),
            opacity=0.35,
            showlegend=True,
        ))

    # Final mean spectrum for this redshift bin
    k = result["k_bin"]
    pk = result["pk_bin"]
    err = result["pk_err"]

    valid = (
        np.isfinite(k)
        & np.isfinite(pk)
        & (k > 0)
        & (pk > 0)
    )

    mean_err = np.where(
        np.isfinite(err[valid]) & (err[valid] >= 0),
        err[valid] / (pk[valid] * np.log(10)),
        np.nan,
    )

    fig_segments.add_trace(go.Scatter(
        x=np.log10(k[valid]),
        y=np.log10(k[valid] * pk[valid] / np.pi),
        mode="lines+markers",
        name="Redshift-bin mean",
        line=dict(width=4, color="black"),
        marker=dict(size=6, color="black"),
        error_y=dict(
            type="data",
            array=mean_err,
            visible=True,
        ),
    ))

    fig_segments.update_layout(
        title=(
            f"Segment Contributions: "
            f"{zmin:.1f} ≤ z < {zmax:.1f}"
        ),
        xaxis_title="log₁₀(k / (km⁻¹ s))",
        yaxis_title="log₁₀[kP(k) / π]",
        legend_title="Quasar and segment",
        height=650,
    )

    st.plotly_chart(fig_segments, width="stretch")



