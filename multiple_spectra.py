"""Combined Ly-alpha forest power-spectrum analysis for multiple quasars.

This script combines the useful parts of multiple_spectra-3.py and
multiple_spectra-2(2).py and fixes the positive-only binning problem.

Important behaviour:
- The combined result is built from Lomb-Scargle spectra of fixed comoving
  segments from every successfully loaded quasar.
- Noise-subtracted power is binned in LINEAR units without rejecting negative
  values. Negative bins are retained in the returned arrays, tables, and CSV.
- Signed power is plotted against log10(k) on a LINEAR y-axis. Conventional
  log10[kP(k)/pi] plots show only positive binned values, because log10 of a
  zero or negative number is undefined.
- The high-k constant noise-floor estimate is retained from the original
  scripts. It is an approximation and must be validated for the dataset.

Run with:
    streamlit run multiple_spectra_combined_signed.py

Expected input files in ./spec:
    <quasar_key>_flux.fits
    <quasar_key>_error.fits

The file prefix before _flux.fits / _error.fits must match a key in REDSHIFTS.
"""

from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from astropy.cosmology import FlatLambdaCDM
from astropy.io import fits
from astropy.timeseries import LombScargle


# -----------------------------------------------------------------------------
# SECTION 1: APP CONFIGURATION AND DATASET
# -----------------------------------------------------------------------------

st.set_page_config(page_title="Combined Lyα Forest Power Spectrum", layout="wide")
st.caption("Combined multi-quasar Lomb–Scargle analysis | signed noise-subtracted power retained")

SPEC_DIR = Path("spec")
LYA_REST_WAVELENGTH = 1215.67  # Angstrom
C_KMS = 299792.458

REDSHIFTS = {
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
    "J0915+4924_HIRES": 5.20,
}

COSMO = FlatLambdaCDM(H0=67.8, Om0=0.308, Ob0=0.0482)

# These are absorber-redshift bins, not quasar-emission-redshift bins.
REDSHIFT_BINS = [
    (4.0, 4.4),
    (4.4, 4.8),
    (4.8, 5.2),
]


# -----------------------------------------------------------------------------
# SECTION 2: GENERAL DATA HANDLING
# -----------------------------------------------------------------------------


def load_fits_1d(path: Path) -> tuple[np.ndarray, fits.Header]:
    """Load the first non-empty FITS HDU and require a one-dimensional array."""
    with fits.open(path, memmap=False) as hdul:
        for hdu in hdul:
            if hdu.data is None:
                continue
            arr = np.asarray(np.squeeze(hdu.data), dtype=float)
            if arr.ndim != 1:
                raise ValueError(
                    f"Expected a 1D spectrum in {path.name}; found shape {arr.shape}."
                )
            return arr.copy(), hdu.header.copy()
    raise ValueError(f"No spectrum data found in {path}")


def wavelength_array(header: fits.Header, n_pixels: int) -> np.ndarray:
    """Build wavelength from the logarithmic FITS WCS used by the input files.

    The input scripts assume CRVAL1/CDELT1 describe log10(wavelength). This
    version also honours CRPIX1 (default 1), which is the standard FITS linear
    WCS reference-pixel convention.
    """
    required = ("CRVAL1", "CDELT1")
    missing = [key for key in required if key not in header]
    if missing:
        raise KeyError(f"Missing FITS wavelength keyword(s): {', '.join(missing)}")

    crval = float(header["CRVAL1"])
    cdelt = float(header["CDELT1"])
    crpix = float(header.get("CRPIX1", 1.0))
    pixel_numbers = np.arange(n_pixels, dtype=float) + 1.0
    log10_wave = crval + (pixel_numbers - crpix) * cdelt
    wave = 10.0**log10_wave

    if not np.all(np.isfinite(wave)) or np.any(wave <= 0):
        raise ValueError("FITS header produced invalid wavelength values.")
    return wave


def native_velocity_spacing(header: fits.Header) -> float:
    """Native velocity pixel spacing in km/s from logarithmic wavelength WCS.

    This is calculated from the original FITS grid rather than from the
    forest array after bad pixels are removed. The latter can contain gaps.
    """
    cdelt = float(header["CDELT1"])
    dv = abs(C_KMS * np.log(10.0) * cdelt)
    if not np.isfinite(dv) or dv <= 0:
        raise ValueError(f"Invalid native velocity spacing: {dv}")
    return dv


def find_flux_error_pairs(folder: Path) -> dict[str, dict[str, Path]]:
    """Find matching *_flux.fits and *_error.fits files."""
    pairs: dict[str, dict[str, Path]] = {}
    if not folder.is_dir():
        return pairs

    for filename in sorted(os.listdir(folder)):
        path = folder / filename
        if not path.is_file() or not filename.lower().endswith(".fits"):
            continue

        if filename.endswith("_flux.fits"):
            key = filename[: -len("_flux.fits")]
            pairs.setdefault(key, {})["flux"] = path
        elif filename.endswith("_error.fits"):
            key = filename[: -len("_error.fits")]
            pairs.setdefault(key, {})["error"] = path

    return {
        key: value
        for key, value in pairs.items()
        if "flux" in value and "error" in value
    }


def snr_calculations(flux: np.ndarray, error: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Return pixel S/N, median S/N, and fraction of unusable input pixels."""
    if flux.shape != error.shape:
        raise ValueError("Flux and error arrays have different shapes.")

    good = np.isfinite(flux) & np.isfinite(error) & (error > 0)
    snr = np.full(flux.shape, np.nan, dtype=float)
    snr[good] = flux[good] / error[good]
    median_snr = float(np.nanmedian(snr)) if np.any(good) else float("nan")
    masked_fraction = 100.0 * float(np.sum(~good)) / max(len(flux), 1)
    return snr, median_snr, masked_fraction


def quality_label(snr: float) -> str:
    if not np.isfinite(snr):
        return "Unavailable"
    if snr > 20:
        return "Excellent (>20)"
    if snr > 10:
        return "Good (>10)"
    if snr > 5:
        return "Moderate (>5)"
    return "Poor (≤5)"


# -----------------------------------------------------------------------------
# SECTION 3: FOREST EXTRACTION AND COORDINATES
# -----------------------------------------------------------------------------


def extract_lya_forest(
    wave_obs: np.ndarray,
    flux: np.ndarray,
    error: np.ndarray,
    quasar_z: float,
    rest_min: float = 1040.0,
    rest_max: float = 1180.0,
) -> dict[str, np.ndarray]:
    """Extract usable pixels in the specified quasar-rest-frame Lyα forest."""
    if not (wave_obs.shape == flux.shape == error.shape):
        raise ValueError("Wavelength, flux, and error arrays must have equal shapes.")

    wave_rest = wave_obs / (1.0 + quasar_z)
    forest = (wave_rest >= rest_min) & (wave_rest <= rest_max)
    good = np.isfinite(wave_obs) & np.isfinite(flux) & np.isfinite(error) & (error > 0)
    mask = forest & good

    return {
        "wave_obs": wave_obs[mask],
        "wave_rest": wave_rest[mask],
        "flux": flux[mask],
        "error": error[mask],
    }


def velocity_grid(wave_rest: np.ndarray, dv_native: float) -> tuple[np.ndarray, float]:
    """Return v=c ln(lambda_rest) and the native detector pixel spacing."""
    if len(wave_rest) < 2:
        raise ValueError("At least two forest pixels are needed for velocity coordinates.")
    velocity = C_KMS * np.log(wave_rest)
    if np.any(np.diff(velocity) <= 0):
        raise ValueError("Forest wavelengths must be strictly increasing.")
    return velocity, float(dv_native)


def comoving_coordinate(wave_obs: np.ndarray) -> np.ndarray:
    """Comoving distance in h^-1 cMpc at the Lyα absorber redshift."""
    absorber_z = wave_obs / LYA_REST_WAVELENGTH - 1.0
    return np.asarray(COSMO.comoving_distance(absorber_z).value * COSMO.h, dtype=float)


def mean_pixel_spacing(chi: np.ndarray) -> float:
    if len(chi) < 2:
        return float("nan")
    return float(np.nanmedian(np.diff(chi)))


def rolling_mean_flux(flux: np.ndarray, chi: np.ndarray, window_cMpc: float) -> np.ndarray:
    """Calculate a centered rolling mean in comoving distance."""
    if len(flux) != len(chi):
        raise ValueError("Flux and comoving-coordinate arrays have different lengths.")
    if window_cMpc <= 0:
        raise ValueError("Rolling-mean window must be positive.")

    half_window = 0.5 * window_cMpc
    smooth = np.full(len(flux), np.nan, dtype=float)
    for i in range(len(flux)):
        left = np.searchsorted(chi, chi[i] - half_window, side="left")
        right = np.searchsorted(chi, chi[i] + half_window, side="right")
        values = flux[left:right]
        values = values[np.isfinite(values)]
        if values.size:
            smooth[i] = float(np.mean(values))
    return smooth


def split_into_segments(chi: np.ndarray, segment_length: float) -> list[np.ndarray]:
    """Split the extracted forest into complete, non-overlapping cMpc segments."""
    if len(chi) == 0:
        return []
    if segment_length <= 0:
        raise ValueError("Segment length must be positive.")
    if np.any(np.diff(chi) < 0):
        raise ValueError("Comoving coordinate must be sorted in increasing order.")

    chi_start = float(chi[0])
    chi_end = float(chi[-1])
    n_segments = int(np.floor((chi_end - chi_start) / segment_length))
    segments: list[np.ndarray] = []

    for index in range(n_segments):
        start = chi_start + index * segment_length
        stop = start + segment_length
        pixel_indices = np.flatnonzero((chi >= start) & (chi < stop))
        if pixel_indices.size:
            segments.append(pixel_indices)
    return segments


# -----------------------------------------------------------------------------
# SECTION 4: SIGNED POWER-SPECTRUM BINNING AND AGGREGATION
# -----------------------------------------------------------------------------


def make_log_k_bins(logk_min: float, logk_max: float, delta_logk: float) -> tuple[np.ndarray, np.ndarray]:
    """Return evenly spaced log10(k) bin centers and corresponding edges."""
    if delta_logk <= 0:
        raise ValueError("Delta log10(k) must be positive.")
    if logk_max <= logk_min:
        raise ValueError("Maximum log10(k) must exceed minimum log10(k).")

    n_intervals = int(np.floor((logk_max - logk_min) / delta_logk + 1e-10))
    centers = logk_min + np.arange(n_intervals + 1, dtype=float) * delta_logk
    if centers.size < 2:
        raise ValueError("Choose a k range that contains at least two bin centers.")

    # Extend each end by half a bin; internal edges are center midpoints.
    edges = np.concatenate((
        [centers[0] - 0.5 * delta_logk],
        0.5 * (centers[:-1] + centers[1:]),
        [centers[-1] + 0.5 * delta_logk],
    ))
    return centers, edges


def bin_power_spectrum_signed(
    k: np.ndarray,
    pk: np.ndarray,
    k_bin_edges: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Bin P(k) in linear units while retaining all finite signed values.

    The validity mask deliberately does NOT contain ``pk > 0``. Negative
    noise-subtracted values contribute to the arithmetic mean and uncertainty.
    Only non-finite powers and non-positive k values are excluded.
    """
    k = np.asarray(k, dtype=float)
    pk = np.asarray(pk, dtype=float)
    if k.shape != pk.shape:
        raise ValueError("k and P(k) arrays must have identical shapes.")

    n_bins = len(k_bin_edges) - 1
    k_bin = 10.0 ** (0.5 * (k_bin_edges[:-1] + k_bin_edges[1:]))
    pk_bin = np.full(n_bins, np.nan, dtype=float)
    pk_err = np.full(n_bins, np.nan, dtype=float)
    n_modes = np.zeros(n_bins, dtype=int)

    valid_k = np.isfinite(k) & (k > 0)
    logk = np.full(k.shape, np.nan, dtype=float)
    logk[valid_k] = np.log10(k[valid_k])

    for i in range(n_bins):
        in_bin = (
            valid_k
            & np.isfinite(pk)
            & (logk >= k_bin_edges[i])
            & (logk < k_bin_edges[i + 1])
        )
        values = pk[in_bin]  # signed values intentionally retained
        n_modes[i] = int(values.size)
        if values.size == 0:
            continue

        pk_bin[i] = float(np.mean(values))
        if values.size > 1:
            pk_err[i] = float(np.std(values, ddof=1) / np.sqrt(values.size))
        else:
            # A single Fourier mode has no within-bin scatter estimate.
            pk_err[i] = np.nan

    return k_bin, pk_bin, pk_err, n_modes


def average_segment_power_spectra(
    segment_ps: list[dict[str, Any]],
    power_key: str = "pk_bin",
    error_key: str | None = "pk_err",
    modes_key: str | None = "n_modes",
) -> dict[str, np.ndarray] | None:
    """Average segment-bin powers in linear units, including negative values.

    Segments are equally weighted, matching the averaging convention in the
    supplied scripts. For bins with multiple segments, the reported error is
    the standard error of the segment-to-segment scatter. For a bin with only
    one contributing segment, its within-segment error is used when available.
    """
    if not segment_ps:
        return None

    k_bin = np.asarray(segment_ps[0]["k_bin"], dtype=float)
    power_rows = []
    error_rows = []
    mode_rows = []

    for segment in segment_ps:
        values = np.asarray(segment.get(power_key, np.full_like(k_bin, np.nan)), dtype=float)
        if values.shape != k_bin.shape:
            raise ValueError(f"Inconsistent k-bin shape for {power_key}.")
        power_rows.append(values)

        if error_key is not None and error_key in segment:
            errors = np.asarray(segment[error_key], dtype=float)
            if errors.shape != k_bin.shape:
                errors = np.full_like(k_bin, np.nan)
        else:
            errors = np.full_like(k_bin, np.nan)
        error_rows.append(errors)

        if modes_key is not None and modes_key in segment:
            modes = np.asarray(segment[modes_key], dtype=float)
            if modes.shape != k_bin.shape:
                modes = np.zeros_like(k_bin)
        else:
            modes = np.zeros_like(k_bin)
        mode_rows.append(modes)

    powers = np.vstack(power_rows)
    within_errors = np.vstack(error_rows)
    modes = np.vstack(mode_rows)

    finite = np.isfinite(powers)
    n_segments = np.sum(finite, axis=0).astype(int)
    mean_power = np.full(k_bin.shape, np.nan, dtype=float)
    mean_error = np.full(k_bin.shape, np.nan, dtype=float)

    has_data = n_segments > 0
    for i in np.flatnonzero(has_data):
        vals = powers[finite[:, i], i]
        mean_power[i] = float(np.mean(vals))
        if vals.size > 1:
            mean_error[i] = float(np.std(vals, ddof=1) / np.sqrt(vals.size))
        else:
            candidate = within_errors[finite[:, i], i]
            if candidate.size and np.isfinite(candidate[0]):
                mean_error[i] = float(candidate[0])

    return {
        "k_bin": k_bin,
        "pk_bin": mean_power,
        "pk_err": mean_error,
        "n_modes": np.sum(modes, axis=0).astype(int),
        "n_segments": n_segments,
    }


def group_segments_by_redshift(
    all_segments: list[dict[str, Any]],
    redshift_bins: list[tuple[float, float]],
) -> dict[tuple[float, float], list[dict[str, Any]]]:
    return {
        (z_min, z_max): [
            seg for seg in all_segments if z_min <= float(seg["z"]) < z_max
        ]
        for z_min, z_max in redshift_bins
    }


# -----------------------------------------------------------------------------
# SECTION 5: LOMB-SCARGLE AND NOISE ESTIMATION
# -----------------------------------------------------------------------------


def compute_k_ls(segment_velocity_length: float, dv_native: float) -> np.ndarray:
    """Construct positive angular wavenumbers in (km/s)^-1."""
    if not np.isfinite(segment_velocity_length) or segment_velocity_length <= 0:
        return np.array([], dtype=float)
    if not np.isfinite(dv_native) or dv_native <= 0:
        return np.array([], dtype=float)

    k_min = 2.0 * np.pi / segment_velocity_length
    k_nyquist = np.pi / dv_native
    n_max = int(np.floor(k_nyquist / k_min))
    if n_max < 1:
        return np.array([], dtype=float)
    return np.arange(1, n_max + 1, dtype=float) * k_min


def power_spectrum_lomb_raw(
    velocity: np.ndarray,
    delta_f: np.ndarray,
    dv_native: float,
    velocity_length: float,
) -> dict[str, np.ndarray]:
    """Calculate the unweighted Lomb-Scargle periodogram on an angular-k grid."""
    velocity = np.asarray(velocity, dtype=float)
    delta_f = np.asarray(delta_f, dtype=float)
    valid = np.isfinite(velocity) & np.isfinite(delta_f)
    velocity = velocity[valid]
    delta_f = delta_f[valid]

    if velocity.size < 10:
        return {"k": np.array([], dtype=float), "pk": np.array([], dtype=float)}
    if np.any(np.diff(velocity) <= 0):
        raise ValueError("Lomb–Scargle velocity coordinates must be increasing.")

    k = compute_k_ls(velocity_length, dv_native)
    if k.size == 0:
        return {"k": k, "pk": np.array([], dtype=float)}

    frequency = k / (2.0 * np.pi)  # cycles per km/s; k is angular wavenumber
    periodogram = LombScargle(
        velocity,
        delta_f,
        normalization="psd",
        fit_mean=True,
        center_data=True,
    )
    pk = np.asarray(periodogram.power(frequency), dtype=float) * dv_native
    return {"k": k, "pk": pk}


def estimate_noise_high_k(
    raw_ps: dict[str, np.ndarray],
    logk_noise_min: float = -0.2,
) -> dict[str, Any]:
    """Estimate a constant white-noise floor from the mean high-k raw power.

    This reproduces the approximation used in the supplied scripts. If fewer
    than two high-k modes are available, the noise estimate is NaN rather than
    silently treating an unmeasured noise floor as zero.
    """
    k = np.asarray(raw_ps["k"], dtype=float)
    pk = np.asarray(raw_ps["pk"], dtype=float)
    valid = np.isfinite(k) & (k > 0) & np.isfinite(pk)
    high_k_mask = valid & (np.log10(k) >= logk_noise_min)

    if np.sum(high_k_mask) < 2:
        p_noise = float("nan")
    else:
        p_noise = float(np.mean(pk[high_k_mask]))

    noise_pk = np.full(pk.shape, p_noise, dtype=float)
    return {
        "k": k,
        "pk": noise_pk,
        "P_noise": p_noise,
        "high_k_mask": high_k_mask,
        "n_high_k": int(np.sum(high_k_mask)),
        "logk_noise_min": float(logk_noise_min),
    }


def _empty_binned(k_bin: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "k_bin": np.asarray(k_bin, dtype=float),
        "pk_bin": np.full(len(k_bin), np.nan, dtype=float),
        "pk_err": np.full(len(k_bin), np.nan, dtype=float),
        "n_modes": np.zeros(len(k_bin), dtype=int),
    }


def lya_power_spectrum_lomb(
    wave_obs: np.ndarray,
    flux: np.ndarray,
    error: np.ndarray,
    quasar_z: float,
    dv_native: float,
    window_cMpc: float,
    segment_length: float,
    k_bin_edges: np.ndarray,
    logk_noise_min: float = -0.2,
) -> dict[str, Any] | None:
    """Run the Lyα forest Lomb–Scargle analysis and retain signed binned power."""
    forest = extract_lya_forest(wave_obs, flux, error, quasar_z)
    if len(forest["flux"]) < 10:
        return None

    wave_obs_forest = forest["wave_obs"]
    wave_rest = forest["wave_rest"]
    flux_forest = forest["flux"]
    error_forest = forest["error"]

    chi_forest = comoving_coordinate(wave_obs_forest)
    velocity, dv = velocity_grid(wave_rest, dv_native)
    smooth = rolling_mean_flux(flux_forest, chi_forest, window_cMpc)

    delta_f = np.full_like(flux_forest, np.nan, dtype=float)
    usable_smooth = np.isfinite(smooth) & (smooth != 0)
    delta_f[usable_smooth] = flux_forest[usable_smooth] / smooth[usable_smooth] - 1.0

    good = (
        np.isfinite(flux_forest)
        & np.isfinite(error_forest)
        & (error_forest > 0)
        & np.isfinite(smooth)
        & (smooth != 0)
        & np.isfinite(delta_f)
    )
    if np.sum(good) < 10:
        return None

    segment_indices = split_into_segments(chi_forest, segment_length)
    segment_ps: list[dict[str, Any]] = []

    for segment_number, indices in enumerate(segment_indices, start=1):
        segment_good = good[indices]
        if np.sum(segment_good) < 10:
            continue

        velocity_section = velocity[indices]
        if velocity_section.size < 2:
            continue
        velocity_length = float(velocity_section[-1] - velocity_section[0] + dv)
        if velocity_length <= 0:
            continue

        segment_velocity = velocity_section[segment_good]
        segment_delta_f = delta_f[indices][segment_good]

        raw_ps = power_spectrum_lomb_raw(
            segment_velocity, segment_delta_f, dv, velocity_length
        )
        if raw_ps["k"].size == 0:
            continue

        noise_ps = estimate_noise_high_k(raw_ps, logk_noise_min)
        corrected_pk = raw_ps["pk"] - noise_ps["pk"]

        k_bin, raw_bin, raw_err, raw_modes = bin_power_spectrum_signed(
            raw_ps["k"], raw_ps["pk"], k_bin_edges
        )
        _, noise_bin, noise_err, noise_modes = bin_power_spectrum_signed(
            raw_ps["k"], noise_ps["pk"], k_bin_edges
        )
        _, corrected_bin, corrected_err, corrected_modes = bin_power_spectrum_signed(
            raw_ps["k"], corrected_pk, k_bin_edges
        )

        segment_wave_center = float(np.mean(wave_obs_forest[indices]))
        segment_z = segment_wave_center / LYA_REST_WAVELENGTH - 1.0

        segment_ps.append({
            "segment_number": segment_number,
            "z": segment_z,
            "k_bin": k_bin,
            # Final noise-subtracted science values: signed and not clipped.
            "pk_bin": corrected_bin,
            "pk_err": corrected_err,
            "n_modes": corrected_modes,
            # Separate diagnostic stages, all in linear power units.
            "pk_raw_bin": raw_bin,
            "pk_raw_err": raw_err,
            "n_raw_modes": raw_modes,
            "pk_noise_bin": noise_bin,
            "pk_noise_err": noise_err,
            "n_noise_modes": noise_modes,
            "pk_sub_bin": corrected_bin.copy(),
            "pk_sub_err": corrected_err.copy(),
            "n_sub_modes": corrected_modes.copy(),
            "P_noise": noise_ps["P_noise"],
            "n_high_k": noise_ps["n_high_k"],
            "logk_noise_min": noise_ps["logk_noise_min"],
            "noise_valid": bool(np.isfinite(noise_ps["P_noise"])),
        })

    if not segment_ps:
        return None

    # Per-quasar mean: equal-weight average of its accepted segments.
    average_ps = average_segment_power_spectra(segment_ps)
    if average_ps is None:
        return None

    # Full-forest diagnostic. This is not the combined science measurement.
    full_velocity_length = float(velocity[-1] - velocity[0] + dv)
    raw_full = power_spectrum_lomb_raw(
        velocity[good], delta_f[good], dv, full_velocity_length
    )
    if raw_full["k"].size:
        noise_full = estimate_noise_high_k(raw_full, logk_noise_min)
        corrected_full = raw_full["pk"] - noise_full["pk"]
        k_full_bin, raw_full_bin, raw_full_err, raw_full_modes = bin_power_spectrum_signed(
            raw_full["k"], raw_full["pk"], k_bin_edges
        )
        _, noise_full_bin, _, _ = bin_power_spectrum_signed(
            raw_full["k"], noise_full["pk"], k_bin_edges
        )
        _, corrected_full_bin, corrected_full_err, corrected_full_modes = bin_power_spectrum_signed(
            raw_full["k"], corrected_full, k_bin_edges
        )
    else:
        noise_full = {"pk": np.array([], dtype=float), "P_noise": np.nan, "n_high_k": 0}
        k_full_bin = 10.0 ** (0.5 * (k_bin_edges[:-1] + k_bin_edges[1:]))
        raw_full_bin = np.full_like(k_full_bin, np.nan)
        raw_full_err = np.full_like(k_full_bin, np.nan)
        raw_full_modes = np.zeros_like(k_full_bin, dtype=int)
        noise_full_bin = np.full_like(k_full_bin, np.nan)
        corrected_full_bin = np.full_like(k_full_bin, np.nan)
        corrected_full_err = np.full_like(k_full_bin, np.nan)
        corrected_full_modes = np.zeros_like(k_full_bin, dtype=int)
        raw_full = {"k": np.array([], dtype=float), "pk": np.array([], dtype=float)}
        corrected_full = np.array([], dtype=float)

    return {
        "wave_obs": wave_obs_forest,
        "wave_rest": wave_rest,
        "flux_forest": flux_forest,
        "error_forest": error_forest,
        "smooth": smooth,
        "deltaF": delta_f,
        "chi_forest": chi_forest,
        "velocity": velocity,
        "dv_forest": dv,
        "n_good": int(np.sum(good)),
        "n_forest_pixels": int(len(flux_forest)),
        "window_cMpc": float(window_cMpc),
        "segment_length": float(segment_length),
        "segment_ps": segment_ps,
        # Per-quasar average of accepted segments, retaining signed values.
        "k_bin": average_ps["k_bin"],
        "pk_bin": average_ps["pk_bin"],
        "pk_err": average_ps["pk_err"],
        "n_modes": average_ps["n_modes"],
        "n_segments_per_bin": average_ps["n_segments"],
        # Full-forest diagnostics.
        "k_full": raw_full["k"],
        "pk_full_raw": raw_full["pk"],
        "pk_full_noise": noise_full["pk"],
        "pk_full_corrected": corrected_full,
        "P_noise_full": noise_full["P_noise"],
        "n_high_k_full": noise_full["n_high_k"],
        "k_full_bin": k_full_bin,
        "pk_full_raw_bin": raw_full_bin,
        "pk_full_raw_err_bin": raw_full_err,
        "pk_full_noise_bin": noise_full_bin,
        "pk_full_corrected_bin": corrected_full_bin,
        "pk_full_corrected_err_bin": corrected_full_err,
        "n_full_modes": corrected_full_modes,
    }


# -----------------------------------------------------------------------------
# SECTION 6: PLOTTING UTILITIES
# -----------------------------------------------------------------------------


def plotly_download_config(filename: str) -> dict[str, Any]:
    safe_name = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in filename)
    return {
        "toImageButtonOptions": {
            "format": "png",
            "filename": safe_name,
            "height": 800,
            "width": 1200,
            "scale": 2,
        }
    }


def signed_y_values(k: np.ndarray, pk: np.ndarray) -> np.ndarray:
    """Return signed dimensionless plotting quantity kP(k)/pi."""
    return np.asarray(k, dtype=float) * np.asarray(pk, dtype=float) / np.pi


def add_signed_trace(
    fig: go.Figure,
    k: np.ndarray,
    pk: np.ndarray,
    name: str,
    error: np.ndarray | None = None,
    mode: str = "lines+markers",
    opacity: float = 1.0,
) -> None:
    """Add a signed linear-y trace; negative power remains visible."""
    k = np.asarray(k, dtype=float)
    pk = np.asarray(pk, dtype=float)
    valid = np.isfinite(k) & (k > 0) & np.isfinite(pk)

    if error is None:
        error_y = None
    else:
        error = np.asarray(error, dtype=float)
        error_y = np.full(np.sum(valid), np.nan, dtype=float)
        if error.shape == k.shape:
            e = error[valid]
            error_y = np.where(np.isfinite(e) & (e >= 0), k[valid] * e / np.pi, np.nan)

    fig.add_trace(go.Scatter(
        x=np.log10(k[valid]),
        y=signed_y_values(k[valid], pk[valid]),
        mode=mode,
        name=name,
        opacity=opacity,
        error_y=(
            dict(type="data", array=error_y, visible=True)
            if error_y is not None else None
        ),
    ))


def add_positive_log_trace(
    fig: go.Figure,
    k: np.ndarray,
    pk: np.ndarray,
    name: str,
    error: np.ndarray | None = None,
    mode: str = "lines+markers",
) -> None:
    """Add only positive binned power to the conventional logarithmic plot."""
    k = np.asarray(k, dtype=float)
    pk = np.asarray(pk, dtype=float)
    valid = np.isfinite(k) & (k > 0) & np.isfinite(pk) & (pk > 0)

    if not np.any(valid):
        return

    error_y = None
    if error is not None:
        error = np.asarray(error, dtype=float)
        if error.shape == k.shape:
            e = error[valid]
            error_y = np.where(
                np.isfinite(e) & (e >= 0),
                e / (pk[valid] * np.log(10.0)),
                np.nan,
            )

    fig.add_trace(go.Scatter(
        x=np.log10(k[valid]),
        y=np.log10(k[valid] * pk[valid] / np.pi),
        mode=mode,
        name=name,
        error_y=(
            dict(type="data", array=error_y, visible=True)
            if error_y is not None else None
        ),
    ))


def make_results_table(
    redshift_results: dict[tuple[float, float], dict[str, Any]],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for zbin, products in redshift_results.items():
        final = products["final"]
        raw = products["raw"]
        noise = products["noise"]
        for i, k in enumerate(final["k_bin"]):
            rows.append({
                "z_min": zbin[0],
                "z_max": zbin[1],
                "log10_k": np.log10(k),
                "k_km-1_s": k,
                "P_raw": raw["pk_bin"][i],
                "P_noise": noise["pk_bin"][i],
                # This column is signed: negative values are intentionally kept.
                "P_corrected": final["pk_bin"][i],
                "P_corrected_error": final["pk_err"][i],
                "n_modes": final["n_modes"][i],
                "n_segments": final["n_segments"][i],
            })
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# SECTION 7: SIDEBAR SETTINGS
# -----------------------------------------------------------------------------

st.sidebar.header("Analysis settings")
window_cMpc = st.sidebar.number_input(
    "Rolling mean window (h⁻¹ cMpc)",
    min_value=10.0,
    max_value=100.0,
    value=40.0,
    step=5.0,
)
segment_length = st.sidebar.number_input(
    "Segment length (h⁻¹ cMpc)",
    min_value=5.0,
    max_value=50.0,
    value=20.0,
    step=1.0,
)
logk_min = st.sidebar.number_input("Minimum log₁₀(k)", value=-2.2, step=0.1)
logk_max = st.sidebar.number_input("Maximum log₁₀(k)", value=-0.7, step=0.1)
delta_logk = st.sidebar.number_input("Δlog₁₀(k)", min_value=0.01, value=0.1, step=0.01)
logk_noise_min = st.sidebar.number_input(
    "High-k noise threshold: log₁₀(k)",
    value=-0.2,
    step=0.05,
    help="The current noise model takes the mean raw power above this threshold as a constant noise floor.",
)

try:
    K_BIN_CENTERS, K_BIN_EDGES = make_log_k_bins(logk_min, logk_max, delta_logk)
except ValueError as exc:
    st.error(str(exc))
    st.stop()


# -----------------------------------------------------------------------------
# SECTION 8: LOAD SPECTRA AND RUN EACH QUASAR
# -----------------------------------------------------------------------------

st.title("Combined Lyα Forest Power Spectrum")
st.markdown(
    "The final combined measurement is an equal-weight mean of accepted "
    "comoving segments within each absorber-redshift bin. All power averaging "
    "is performed in linear units, and negative noise-subtracted values are retained."
)

pairs = find_flux_error_pairs(SPEC_DIR)
if not pairs:
    st.error(
        f"No matching *_flux.fits / *_error.fits pairs were found in '{SPEC_DIR}'. "
        "Create the folder and place the spectra there."
    )
    st.stop()

spectra: list[dict[str, Any]] = []
summary_rows: list[dict[str, Any]] = []
load_errors: list[dict[str, str]] = []

progress = st.progress(0.0, text="Loading spectra and calculating power spectra...")
for pair_index, (key, files) in enumerate(sorted(pairs.items()), start=1):
    try:
        if key not in REDSHIFTS:
            raise KeyError(f"No quasar redshift is defined for file key '{key}'.")

        flux, flux_header = load_fits_1d(files["flux"])
        error, _ = load_fits_1d(files["error"])
        if flux.shape != error.shape:
            raise ValueError(
                f"Flux and error arrays have different lengths: {len(flux)} vs {len(error)}."
            )

        wave = wavelength_array(flux_header, len(flux))
        # Ensure all wavelength-dependent arrays are in increasing wavelength.
        order = np.argsort(wave)
        wave = wave[order]
        flux = flux[order]
        error = error[order]

        quasar_z = float(REDSHIFTS[key])
        dv_native = native_velocity_spacing(flux_header)
        snr, median_snr, masked_fraction = snr_calculations(flux, error)
        chi_full = comoving_coordinate(wave)
        pixel_spacing = mean_pixel_spacing(chi_full)

        ps_lomb = lya_power_spectrum_lomb(
            wave_obs=wave,
            flux=flux,
            error=error,
            quasar_z=quasar_z,
            dv_native=dv_native,
            window_cMpc=window_cMpc,
            segment_length=segment_length,
            k_bin_edges=K_BIN_EDGES,
            logk_noise_min=logk_noise_min,
        )

        object_name = str(flux_header.get("OBJECT", key))
        instrument = str(flux_header.get("INSTRUME", "Unknown"))
        spectrum_record = {
            "key": key,
            "object": object_name,
            "z": quasar_z,
            "instrument": instrument,
            "wave": wave,
            "flux": flux,
            "error": error,
            "snr": snr,
            "median_snr": median_snr,
            "masked_fraction": masked_fraction,
            "dv_native": dv_native,
            "chi": chi_full,
            "pixel_spacing": pixel_spacing,
            "window_cMpc": float(window_cMpc),
            "segment_length": float(segment_length),
            "ps_lomb": ps_lomb,
        }
        spectra.append(spectrum_record)

        if ps_lomb is None:
            n_forest = 0
            n_good = 0
            n_segments = 0
            noise_valid_segments = 0
        else:
            n_forest = ps_lomb["n_forest_pixels"]
            n_good = ps_lomb["n_good"]
            n_segments = len(ps_lomb["segment_ps"])
            noise_valid_segments = sum(seg["noise_valid"] for seg in ps_lomb["segment_ps"])

        summary_rows.append({
            "Object": object_name,
            "File key": key,
            "Quasar z": quasar_z,
            "Instrument": instrument,
            "Pixels": len(flux),
            "Wavelength min (Å)": float(np.min(wave)),
            "Wavelength max (Å)": float(np.max(wave)),
            "Median S/N": median_snr,
            "Unusable pixels (%)": masked_fraction,
            "Native dv (km/s)": dv_native,
            "Median pixel spacing (h⁻¹ cMpc)": pixel_spacing,
            "Forest pixels used": n_forest,
            "Good normalized pixels": n_good,
            "Accepted segments": n_segments,
            "Segments with noise estimate": noise_valid_segments,
        })

    except Exception as exc:
        load_errors.append({"File key": key, "Error": str(exc)})

    progress.progress(pair_index / len(pairs), text=f"Processed {pair_index}/{len(pairs)} spectra")
progress.empty()

if not spectra:
    st.error("No spectra could be processed. See the loading errors below.")
    if load_errors:
        st.dataframe(pd.DataFrame(load_errors), width="stretch")
    st.stop()


# -----------------------------------------------------------------------------
# SECTION 9: COMBINE SEGMENTS FROM ALL QUASARS
# -----------------------------------------------------------------------------

all_segments: list[dict[str, Any]] = []
for spec in spectra:
    ps_lomb = spec["ps_lomb"]
    if ps_lomb is None:
        continue
    for segment in ps_lomb["segment_ps"]:
        segment_record = dict(segment)
        segment_record["quasar"] = spec["object"]
        segment_record["quasar_key"] = spec["key"]
        all_segments.append(segment_record)

grouped_segments = group_segments_by_redshift(all_segments, REDSHIFT_BINS)

redshift_results: dict[tuple[float, float], dict[str, Any]] = {}
for zbin, segments in grouped_segments.items():
    if not segments:
        continue
    redshift_results[zbin] = {
        "final": average_segment_power_spectra(
            segments, power_key="pk_bin", error_key="pk_err", modes_key="n_modes"
        ),
        "raw": average_segment_power_spectra(
            segments, power_key="pk_raw_bin", error_key="pk_raw_err", modes_key="n_raw_modes"
        ),
        "noise": average_segment_power_spectra(
            segments, power_key="pk_noise_bin", error_key="pk_noise_err", modes_key="n_noise_modes"
        ),
        "segments": segments,
    }


# -----------------------------------------------------------------------------
# SECTION 10: DATASET AND COMBINATION SUMMARY
# -----------------------------------------------------------------------------

st.header("Dataset summary")
summary_df = pd.DataFrame(summary_rows)
st.dataframe(summary_df, width="stretch", hide_index=True)

if load_errors:
    with st.expander(f"Files not processed ({len(load_errors)})", expanded=False):
        st.dataframe(pd.DataFrame(load_errors), width="stretch", hide_index=True)

segment_summary_rows = []
for zbin, segments in grouped_segments.items():
    segment_summary_rows.append({
        "Absorber redshift bin": f"{zbin[0]:.1f} ≤ z < {zbin[1]:.1f}",
        "Segments": len(segments),
        "Distinct quasars": len({seg["quasar_key"] for seg in segments}),
        "Segments with noise estimate": sum(bool(seg["noise_valid"]) for seg in segments),
    })
st.subheader("Segments contributing to each redshift bin")
st.dataframe(pd.DataFrame(segment_summary_rows), width="stretch", hide_index=True)


# -----------------------------------------------------------------------------
# SECTION 11: FINAL ALL-QUASAR POWER SPECTRUM
# -----------------------------------------------------------------------------

st.header("Final combined power spectrum by absorber redshift")
st.caption(
    "Signed linear-power plot: all finite noise-subtracted measurements are included, "
    "including negative values. The y-axis is kP(k)/π in linear units."
)

fig_signed = go.Figure()
for (z_min, z_max), products in redshift_results.items():
    final = products["final"]
    add_signed_trace(
        fig_signed,
        final["k_bin"],
        final["pk_bin"],
        name=f"{z_min:.1f} ≤ z < {z_max:.1f}",
        error=final["pk_err"],
    )
fig_signed.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.7)
fig_signed.update_layout(
    title="Combined Lyα forest power spectrum (signed)",
    xaxis_title="log₁₀(k / (km s⁻¹)⁻¹)",
    yaxis_title="kP(k) / π (linear; signed)",
    hovermode="x unified",
    legend_title="Absorber redshift",
)
st.plotly_chart(fig_signed, width="stretch")

with st.expander("Conventional logarithmic presentation (positive bins only)", expanded=False):
    st.caption(
        "This is a presentation-only view. Non-positive binned powers are omitted "
        "because they cannot be represented on a logarithmic y-axis; they remain "
        "in the signed plot, result table, and CSV export above/below."
    )
    fig_log = go.Figure()
    for (z_min, z_max), products in redshift_results.items():
        final = products["final"]
        add_positive_log_trace(
            fig_log,
            final["k_bin"],
            final["pk_bin"],
            name=f"{z_min:.1f} ≤ z < {z_max:.1f}",
            error=final["pk_err"],
        )
    fig_log.update_layout(
        title="Combined Lyα forest power spectrum (positive bins only)",
        xaxis_title="log₁₀(k / (km s⁻¹)⁻¹)",
        yaxis_title="log₁₀[kP(k) / π]",
        hovermode="x unified",
        legend_title="Absorber redshift",
    )
    st.plotly_chart(fig_log, width="stretch")

with st.expander("Raw, noise, and noise-subtracted comparison", expanded=False):
    fig_stages = go.Figure()
    for (z_min, z_max), products in redshift_results.items():
        raw = products["raw"]
        noise = products["noise"]
        final = products["final"]
        suffix = f"({z_min:.1f}–{z_max:.1f})"
        add_signed_trace(fig_stages, raw["k_bin"], raw["pk_bin"], f"Raw {suffix}", mode="lines")
        add_signed_trace(fig_stages, noise["k_bin"], noise["pk_bin"], f"Noise {suffix}", mode="lines")
        add_signed_trace(fig_stages, final["k_bin"], final["pk_bin"], f"Corrected {suffix}", error=final["pk_err"])
    fig_stages.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.7)
    fig_stages.update_layout(
        title="Mean raw, estimated noise, and corrected power",
        xaxis_title="log₁₀(k / (km s⁻¹)⁻¹)",
        yaxis_title="kP(k) / π (linear; signed)",
        hovermode="x unified",
    )
    st.plotly_chart(fig_stages, width="stretch")


# -----------------------------------------------------------------------------
# SECTION 12: REDSHIFT-BIN SEGMENT CONTRIBUTIONS
# -----------------------------------------------------------------------------

with st.expander("Individual segment contributions to each redshift bin", expanded=False):
    for (z_min, z_max), products in redshift_results.items():
        segments = products["segments"]
        st.subheader(f"{z_min:.1f} ≤ z < {z_max:.1f}  |  {len(segments)} segments")

        fig_seg = go.Figure()
        for seg in segments:
            add_signed_trace(
                fig_seg,
                seg["k_bin"],
                seg["pk_bin"],
                name=f"{seg['quasar']} — segment {seg['segment_number']}",
                mode="lines",
                opacity=0.3,
            )
        final = products["final"]
        add_signed_trace(
            fig_seg,
            final["k_bin"],
            final["pk_bin"],
            name="Redshift-bin mean",
            error=final["pk_err"],
            mode="lines+markers",
        )
        fig_seg.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.7)
        fig_seg.update_layout(
            title=f"Signed segment power: {z_min:.1f} ≤ z < {z_max:.1f}",
            xaxis_title="log₁₀(k / (km s⁻¹)⁻¹)",
            yaxis_title="kP(k) / π (linear; signed)",
            hovermode="closest",
            legend_title="Quasar and segment",
            height=650,
        )
        st.plotly_chart(fig_seg, width="stretch")


# -----------------------------------------------------------------------------
# SECTION 13: INDIVIDUAL QUASAR INSPECTION
# -----------------------------------------------------------------------------

st.header("Individual quasar diagnostics")
selectable_spectra = [
    i for i, spec in enumerate(spectra) if spec["ps_lomb"] is not None
]
if not selectable_spectra:
    st.info("No quasar has enough usable Lyα forest pixels and accepted segments.")
else:
    label_to_index = {
        f"{spectra[i]['object']}  (z={spectra[i]['z']:.3f}; {spectra[i]['instrument']})": i
        for i in selectable_spectra
    }
    selected_label = st.selectbox("Select a quasar", list(label_to_index.keys()))
    spec = spectra[label_to_index[selected_label]]
    ps_lomb = spec["ps_lomb"]

    metric_cols = st.columns(5)
    metric_cols[0].metric("Quasar redshift", f"{spec['z']:.3f}")
    metric_cols[1].metric("Median S/N", f"{spec['median_snr']:.2f}" if np.isfinite(spec["median_snr"]) else "N/A")
    metric_cols[2].metric("Unusable pixels", f"{spec['masked_fraction']:.1f}%")
    metric_cols[3].metric("Accepted segments", str(len(ps_lomb["segment_ps"])))
    metric_cols[4].metric("Good forest pixels", str(ps_lomb["n_good"]))

    with st.expander("Forest flux, rolling mean, and flux contrast", expanded=False):
        fig_flux = go.Figure()
        fig_flux.add_trace(go.Scatter(
            x=ps_lomb["wave_obs"], y=ps_lomb["flux_forest"],
            mode="lines", name="Forest flux", line=dict(width=1),
        ))
        fig_flux.add_trace(go.Scatter(
            x=ps_lomb["wave_obs"], y=ps_lomb["smooth"],
            mode="lines", name=f"Rolling mean ({window_cMpc:.0f} h⁻¹ cMpc)",
            line=dict(width=2),
        ))
        fig_flux.update_layout(
            title=f"{spec['object']}: Lyα forest flux and rolling mean",
            xaxis_title="Observed wavelength (Å)",
            yaxis_title="Flux",
            hovermode="x unified",
        )
        st.plotly_chart(fig_flux, width="stretch")

        fig_delta = go.Figure()
        fig_delta.add_trace(go.Scatter(
            x=ps_lomb["wave_obs"], y=ps_lomb["deltaF"],
            mode="lines", name="δF", line=dict(width=1),
        ))
        fig_delta.add_hline(y=0, line_dash="dash", line_color="gray")
        fig_delta.update_layout(
            title=f"{spec['object']}: rolling-mean normalized flux contrast",
            xaxis_title="Observed wavelength (Å)",
            yaxis_title="δF",
            hovermode="x unified",
        )
        st.plotly_chart(fig_delta, width="stretch")

    with st.expander("Per-quasar signed and logarithmic power spectra", expanded=True):
        fig_qso_signed = go.Figure()
        add_signed_trace(
            fig_qso_signed,
            ps_lomb["k_bin"],
            ps_lomb["pk_bin"],
            "Mean of this quasar's segments",
            error=ps_lomb["pk_err"],
        )
        fig_qso_signed.add_hline(y=0, line_dash="dash", line_color="gray")
        fig_qso_signed.update_layout(
            title=f"{spec['object']}: signed segment-averaged power",
            xaxis_title="log₁₀(k / (km s⁻¹)⁻¹)",
            yaxis_title="kP(k) / π (linear; signed)",
        )
        st.plotly_chart(fig_qso_signed, width="stretch")

        fig_qso_log = go.Figure()
        add_positive_log_trace(
            fig_qso_log,
            ps_lomb["k_bin"],
            ps_lomb["pk_bin"],
            "Positive bins only",
            error=ps_lomb["pk_err"],
        )
        fig_qso_log.update_layout(
            title=f"{spec['object']}: positive bins on logarithmic scale",
            xaxis_title="log₁₀(k / (km s⁻¹)⁻¹)",
            yaxis_title="log₁₀[kP(k) / π]",
        )
        st.plotly_chart(fig_qso_log, width="stretch")

    with st.expander("Segment noise estimates and signed binned powers", expanded=False):
        qso_segment_rows = []
        for seg in ps_lomb["segment_ps"]:
            qso_segment_rows.append({
                "Segment": seg["segment_number"],
                "Absorber z": seg["z"],
                "Noise estimate valid": seg["noise_valid"],
                "P_noise": seg["P_noise"],
                "High-k modes used": seg["n_high_k"],
                "Negative corrected bins": int(np.sum(np.isfinite(seg["pk_bin"]) & (seg["pk_bin"] < 0))),
                "Finite corrected bins": int(np.sum(np.isfinite(seg["pk_bin"]))),
            })
        st.dataframe(pd.DataFrame(qso_segment_rows), width="stretch", hide_index=True)

        fig_qso_segments = go.Figure()
        for seg in ps_lomb["segment_ps"]:
            add_signed_trace(
                fig_qso_segments,
                seg["k_bin"],
                seg["pk_bin"],
                name=f"Segment {seg['segment_number']}",
                mode="lines+markers",
                opacity=0.7,
            )
        fig_qso_segments.add_hline(y=0, line_dash="dash", line_color="gray")
        fig_qso_segments.update_layout(
            title=f"{spec['object']}: signed power spectrum for each segment",
            xaxis_title="log₁₀(k / (km s⁻¹)⁻¹)",
            yaxis_title="kP(k) / π (linear; signed)",
            hovermode="closest",
        )
        st.plotly_chart(fig_qso_segments, width="stretch")


# -----------------------------------------------------------------------------
# SECTION 14: EXPORTS
# -----------------------------------------------------------------------------

st.header("Download results")

combined_df = make_results_table(redshift_results)
st.subheader("Combined signed power-spectrum table")
st.caption("P_corrected is in linear units and includes negative values.")
st.dataframe(combined_df, width="stretch", hide_index=True)

csv_bytes = combined_df.to_csv(index=False).encode("utf-8")
st.download_button(
    label="Download combined signed power spectrum (CSV)",
    data=csv_bytes,
    file_name="combined_lya_power_spectrum_signed.csv",
    mime="text/csv",
)

spectra_bytes = pickle.dumps(
    {
        "spectra": spectra,
        "all_segments": all_segments,
        "grouped_segments": grouped_segments,
        "redshift_results": redshift_results,
        "k_bin_centers": K_BIN_CENTERS,
        "k_bin_edges": K_BIN_EDGES,
        "settings": {
            "window_cMpc": window_cMpc,
            "segment_length": segment_length,
            "logk_noise_min": logk_noise_min,
            "redshift_bins": REDSHIFT_BINS,
        },
    },
    protocol=pickle.HIGHEST_PROTOCOL,
)
st.download_button(
    label="Download full analysis data (pickle)",
    data=spectra_bytes,
    file_name="lya_power_spectrum_full_analysis.pkl",
    mime="application/octet-stream",
)

st.caption(
    "Method note: the code preserves signed noise-subtracted values, but the "
    "constant high-k noise-floor model and unweighted Lomb–Scargle normalization "
    "are inherited methodological assumptions from the supplied scripts. "
    "Validate these assumptions before using the result as a final publication measurement."
)
