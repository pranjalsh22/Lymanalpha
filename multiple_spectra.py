#----section 1: imports config and dataset folder-------------

import os
import numpy as np
import pandas as pd
import streamlit as st
from astropy.io import fits
import plotly.graph_objects as go

st.set_page_config(
    page_title=".fit file plots",
    layout="wide")

SPEC_DIR = "spec"
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
    "J0915+4924_HIRES": 5.20,
}
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
def extract_lya_forest(wave_obs,flux,z,rest_min=1040,rest_max=1180):

    wave_rest = (wave_obs/ (1 + z))

    mask = ((wave_rest >= rest_min) & (wave_rest <= rest_max) & np.isfinite(flux) & np.isfinite(error)&(error > 0))

    return (wave_rest[mask],flux[mask])

#---section 2.8: flux_contrast--------------------------------

def flux_contrast(flux_forest):

    if len(flux_forest) < 10:
        return None, None

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

    L = N * dv

    Pk = (np.abs(fft_vals) ** 2) / L

    return Pk

#---- section 2.13:

def boera_quantity(k,pk):

    valid = (np.isfinite(k) & np.isfinite(pk) &(k > 0)&(pk > 0))

    logk = np.log10(k[valid])

    logkpk = np.log10((k[valid]*pk[valid])/np.pi)

    return (logk,logkpk)

#---- section 2.14: master power spectrum function

def lya_power_spectrum(wave_obs,flux,z):

    wave_rest, flux_forest = (extract_lya_forest(wave_obs,flux,z))

    if len(flux_forest) < 10:
        return None

    Fmean, deltaF = (flux_contrast(flux_forest))

    if deltaF is None:
        return None

    velocity, dv_forest = (velocity_grid(wave_rest))

    fft_vals, N = (compute_fft(deltaF))

    k = compute_k(N,dv_forest)

    pk = compute_power_spectrum(fft_vals,N,dv_forest)

    logk, logkpk = (boera_quantity(k,pk))
    (logk_bin,logpk_bin,logpk_err) = bin_power_spectrum(k,pk)
    
    return {

        "wave_rest": wave_rest,

        "flux_forest": flux_forest,

        "Fmean": Fmean,

        "deltaF": deltaF,

        "velocity": velocity,

        "dv_forest": dv_forest,

        "fft": fft_vals,

        "k": k,

        "pk": pk,

        "logk": logk,

        "logkpk": logkpk,
        "logk_bin": logk_bin,
        "logpk_bin": logpk_bin,
        "logpk_err": logpk_err,}

#------section 2.15: bining the power spectrum-------------
def bin_power_spectrum(k, pk, n_bins=20):
    #output arrays: mean of the bin values in log(k), mean k*P_/pi, standard deviation of log(k*p(k)/pi)
    #st.write('Taking the values where both k and pk are finite and positive.
    #Creating 20 bins in range of log(k).')
    
    valid = (np.isfinite(k) & np.isfinite(pk) & (k > 0) & (pk > 0))
    k = k[valid]
    pk = pk[valid]
    logk = np.log10(k)
    bins = np.linspace(logk.min(),logk.max(),n_bins + 1)

    k_bin = []
    pk_bin = []
    pk_err = []

    for i in range(n_bins):

        m = ((logk >= bins[i]) &(logk < bins[i + 1]))

        k_bin.append(np.mean(logk[m]))

        pk_values = (k[m]*pk[m] /np.pi)

        pk_bin.append(np.log10(np.mean(pk_values)))

        pk_err.append(np.std(np.log10(pk_values)))

    return (np.array(k_bin),np.array(pk_bin),np.array(pk_err))


#------ Section 3:Setting up the data------------------------------------ 

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

        ps = lya_power_spectrum(wave,flux,z)

        dv = velocity_spacing(header)

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

            "ps": ps})

        summary_rows.append({

            "Object": object_name,

            "z": z,

            "Instrument": instrument,

            "Pixels": len(flux),

            "Lambda Min": wave.min(),

            "Lambda Max": wave.max(),

            "Median S/N": median_snr,

            "Masked %": masked_fraction,

            "dv (km/s)": dv})

    except Exception as e:
        st.warning(f"{key}: {e}")

#-----section 4: Display---------------------------
        
#-----section 4.1

st.title("Quasar Spectra")

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

        ps = spec["ps"]

        #-----section 4.2.2: Basic Statistics:
        finite_pixels = int(np.sum(np.isfinite(flux)))

        masked_pixels = int(np.sum(~np.isfinite(flux)))

        c1, c2, c3, c4, c5 = st.columns(5)

        c1.metric("Median S/N",f"{spec['median_snr']:.6e}")

        c2.metric("Quality",quality_label(spec["median_snr"]))

        c3.metric("Masked %",f"{spec['masked_fraction']:.6e}")

        c4.metric("Velocity Spacing",f"{spec['dv']:.6e}")

        c5.metric("Redshift",f"{spec['z']:.6e}")

        st.write(f"Pixels: {len(flux)}")

        st.write(f"Finite Pixels: {finite_pixels}")

        st.write(f"Masked Pixels: {masked_pixels}")

        st.write(
            f"Wavelength Range: "
            f"{wave.min():.1f}"
            f" – "
            f"{wave.max():.1f} Å")

        #-----section 4.2.3: Lyα Forest Diagnostics
        if ps is not None:

            st.subheader("Lyα Forest Diagnostics")

            st.write(f"Forest Pixels: "
                f"{len(ps['deltaF'])}")

            st.write(f"Mean Flux <F>: "
                f"{ps['Fmean']:.6e}")

            st.write(f"Forest dv: "
                f"{ps['dv_forest']:.6e} km/s")

            st.write(
                f"k range: "
                f"{ps['k'][1]:.4e}"
                f" to "
                f"{ps['k'][-1]:.4e}")
        
        #-----section 4.2.4: Lyα Power Spectrum Plot
        #Part A
        if ps is not None:

            fig_ps = go.Figure()

            fig_ps.add_trace(

                go.Scatter(

                    x=ps["logk"],

                    y=ps["logkpk"],

                    mode="lines",

                    name="log(kP(k)/π)"))

            fig_ps.update_layout(

                title="Lyα Forest Power Spectrum",

                xaxis_title="log₁₀(k / km⁻¹ s)",

                yaxis_title="log₁₀(kP(k)/π)")

            st.plotly_chart(fig_ps,use_container_width=True)
        #Part B
        if ps is not None:

            fig_binned = go.Figure()

            fig_binned.add_trace(

                go.Scatter(

                    x=ps["logk_bin"],

                    y=ps["logpk_bin"],

                    mode="markers+lines",

                    error_y=dict(
                        type="data",
                        array=ps["logpk_err"],
                        visible=True
                    ),

                    name="Binned Power"
                )
            )

            fig_binned.update_layout(

                title=
                "Binned Lyα Forest Power Spectrum",

                xaxis_title=
                "log₁₀(k / km⁻¹ s)",

                yaxis_title=
                "log₁₀(kP(k)/π)"
            )

            st.plotly_chart(
                fig_binned,
                use_container_width=True)
        #-----section 4.2.5: Flux Contrast Plot

        if ps is not None:

            fig_forest = go.Figure()

            fig_forest.add_trace(

                go.Scatter(

                    x=ps["wave_rest"],

                    y=ps["deltaF"],

                    mode="lines",

                    name="δF"))

            fig_forest.update_layout(

                title=
                "Lyα Forest Flux Contrast",

                xaxis_title=
                "Rest Wavelength (Å)",

                yaxis_title="δF")

            st.plotly_chart(fig_forest,use_container_width=True)
            
        #-----section 4.2.6:Flux Spectrum

        mask = np.isfinite(flux)

        fig1 = go.Figure()

        fig1.add_trace(

            go.Scatter(

                x=wave[mask],

                y=flux[mask],

                mode="lines",

                name="Flux"))

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

            fig2.add_trace(

                go.Scatter(

                    x=wave[smask],

                    y=snr[smask],

                    mode="lines",

                    name="SNR"))

            fig2.update_layout(

                title=
                "Signal-to-Noise Ratio",

                xaxis_title=
                "Observed Wavelength (Å)",

                yaxis_title="S/N")

            st.plotly_chart(fig2,use_container_width=True)
        
        #-----section 4.2.8:Text Analysis

        st.write(

            f"This {spec['instrument']} spectrum "

            f"contains {len(flux):,} pixels. "

            f"The median S/N is "

            f"{spec['median_snr']:.2f}, "

            f"which corresponds to "

            f"{quality_label(spec['median_snr'])} "

            f"data quality. "

            f"The masked fraction is "

            f"{spec['masked_fraction']:.2f}% "

            f"and the velocity spacing is "

            f"{spec['dv']:.2f} km/s.")
            
        #-----section 4.2.9: FITS Header

        with st.expander("FITS Header"):
            st.json(dict(spec["header"]))
