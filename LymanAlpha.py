
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
        hdu_rows = []

        for i, hdu in enumerate(hdul):
            shape = None if hdu.data is None else str(np.shape(hdu.data))

            hdu_rows.append({
                "HDU": i,
                "Name": hdu.name,
                "Shape": shape
            })

            if data is None and hdu.data is not None:
                arr = np.array(hdu.data, dtype=np.float64)

                if arr.ndim == 2:
                    if arr.shape[0] == 1:
                        arr = arr[0]
                    elif arr.shape[1] == 1:
                        arr = arr[:, 0]

                data = np.squeeze(arr)
                header = hdu.header

    if data is None:
        raise ValueError("No spectrum found")

    return data, header, pd.DataFrame(hdu_rows)

def wavelength_array(header, n):
    return 10 ** (
        header["CRVAL1"]
        + np.arange(n) * header["CDELT1"]
    )

st.title("Lyα Forest")
#st.caption("Interactive walkthrough from FITS files to Lyα forest spectra")

flux_upload = st.file_uploader(
    "Upload Flux FITS",
    type=["fits"],
    key="flux"
)

error_upload = st.file_uploader(
    "Upload Error FITS",
    type=["fits"],
    key="error"
)

try:
    if flux_upload and error_upload:
        flux, header, hdu_df = load_fits(flux_upload)
        error, _, _ = load_fits(error_upload)
        st.success("Using uploaded files")
    else:
        flux, header, hdu_df = load_fits(DEFAULT_FLUX)
        error, _, _ = load_fits(DEFAULT_ERROR)
        st.info("Using bundled example files")
except Exception as e:
    st.error(str(e))
    st.stop()


if len(flux) != len(error):
    st.error(
        f"Flux length ({len(flux)}) "
        f"does not match "
        f"error length ({len(error)})"
    )
    st.stop()

n = len(flux)
wave = wavelength_array(header, n)

snr = np.full(n, np.nan)
good = np.isfinite(flux) & np.isfinite(error) & (error > 0)
snr[good] = flux[good] / error[good]

# --------------------------------------------------
# OBJECT INFORMATION
# --------------------------------------------------

st.header("Object Information")

object_name = header.get("OBJECT", "Unknown")
instrument = header.get("INSTRUME", "Unknown")
flux_units = header.get("BUNIT", "Not specified")

st.write("Object:", object_name)
st.write("Instrument:", instrument)
st.write("Flux Units:", flux_units)

header_z = None
for key in ["REDSHIFT", "Z", "EMZ", "QSO_Z"]:
    if key in header:
        try:
            header_z = float(header[key])
            break
        except:
            pass

known_redshifts = {
    "SDSS J0011+1446": 4.967,
    "J0011+1446": 4.967,
}

default_z = header_z if header_z is not None else known_redshifts.get(object_name, 5.0)

st.metric("Quasar Redshift", f"{default_z:.3f}")

st.subheader("Derived Quantities")

st.latex(r"\lambda_{\alpha}=1215.67(1+z)")
st.latex(r"\lambda_{\beta}=1025.72(1+z)")

lya_obs = 1215.67 * (1 + default_z)
lyb_obs = 1025.72 * (1 + default_z)

st.write(f"Observed Lyα wavelength: {lya_obs:.1f} Å")
st.write(f"Observed Lyβ wavelength: {lyb_obs:.1f} Å")

forest_start = 1040 * (1 + default_z)
forest_end = 1180 * (1 + default_z)

st.write(f"Expected Lyα forest region: {forest_start:.1f} Å – {forest_end:.1f} Å")


# --------------------------------------------------
# WORKFLOW EXPLANATION
# --------------------------------------------------

st.header("📖 How We Go From FITS Files to Spectra")

st.markdown("""
This app follows the same workflow used when analyzing HIRES/UVES quasar spectra.

### Step 1: Read the FITS files

The flux FITS file contains the measured quasar spectrum.

The error FITS file contains the estimated 1σ uncertainty for each pixel.

Each pixel corresponds to one wavelength bin.
""")

st.header("1️⃣ FITS Structure")
st.dataframe(hdu_df, width="stretch")

st.markdown("""
For this spectrum:

- NAXIS = 1 → one-dimensional spectrum
- NAXIS1 = number of pixels
- CRVAL1 = starting log wavelength
- CDELT1 = spacing in log wavelength
""")

# --------------------------------------------------

st.header("2️⃣ Constructing the Wavelength Array")

st.latex(r"\lambda_i = 10^{CRVAL1 + i\,CDELT1}")

st.markdown("""
The FITS file does not store wavelength explicitly for every pixel.

Instead, it stores:

- CRVAL1 = starting log wavelength
- CDELT1 = step size
- i = pixel number

Using the formula above we compute the wavelength corresponding to every pixel.
""")

st.write("CRVAL1 =", header["CRVAL1"])
st.write("CDELT1 =", header["CDELT1"])

# --------------------------------------------------

st.header("3️⃣ Constant Velocity Binning")

st.latex(r"\Delta v = c\ln(10)\,CDELT1")

dv = 299792.458 * np.log(10) * header["CDELT1"]

st.write(f"Velocity spacing = {dv:.3f} km/s")

st.markdown("""
HIRES spectra are stored on a logarithmic wavelength grid.

This means each pixel corresponds to approximately the same velocity interval.
""")

# --------------------------------------------------

st.header("4️⃣ Data Quality")

c1, c2, c3 = st.columns(3)

c1.metric(
    "Total Pixels",
    n
)

c2.metric(
    "Finite Flux Pixels",
    int(np.sum(np.isfinite(flux)))
)

c3.metric(
    "Masked Flux Pixels",
    int(np.sum(~np.isfinite(flux)))
)

first_valid = np.where(np.isfinite(flux))[0][0]

st.write("First valid pixel:", first_valid)

# --------------------------------------------------

preview = pd.DataFrame({
    "Pixel": np.arange(n),
    "Wavelength (Å)": wave,
    "Flux": flux,
    "Error": error,
    "SNR": snr
})

st.header("5️⃣ First Valid Data")

st.markdown("""
The beginning of the file contains masked pixels (NaNs).

Below we display the first valid measurements in the spectrum.
""")

valid_preview = preview[np.isfinite(preview["Flux"])].head(1000)

st.dataframe(
    valid_preview,
    width="stretch"
)

# --------------------------------------------------

st.header("6️⃣ Flux Spectrum")

st.markdown("""
This is the observed quasar spectrum.

Every dip corresponds to absorption by intervening material between us and the quasar.
""")

mask = np.isfinite(flux)

fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=wave[mask],
        y=flux[mask],
        mode="lines",
        name="Flux"
    )
)

st.plotly_chart(fig, width="stretch")

# --------------------------------------------------

st.header("7️⃣ Signal-to-Noise Ratio")

st.latex(r"S/N = \frac{F}{\sigma}")

st.markdown("""
Where:

- F = measured flux
- σ = 1σ uncertainty

Large S/N means more reliable measurements.
""")

fig2 = go.Figure()

snrmask = np.isfinite(snr)

fig2.add_trace(
    go.Scatter(
        x=wave[snrmask],
        y=snr[snrmask],
        mode="lines",
        name="SNR"
    )
)

st.plotly_chart(fig2, width="stretch")

# --------------------------------------------------

st.header("8️⃣ Lyα Forest")

z = st.number_input(
    "Quasar Redshift",
    value=float(default_z),
    step=0.01
)

st.latex(r"\lambda_{\alpha}=1215.67(1+z)")

lya = 1215.67 * (1 + z)

fig3 = go.Figure()

fig3.add_trace(
    go.Scatter(
        x=wave[mask],
        y=flux[mask],
        mode="lines",
        name="Flux"
    )
)

fig3.add_vline(x=lya)

st.plotly_chart(fig3, width="stretch")

st.markdown("""
Everything blueward of the Lyα emission line contains the Lyα forest.

This is the region used for flux power spectrum measurements.
""")

# --------------------------------------------------

st.header("9️⃣ Absorber Redshift")

st.latex(r"z_{abs}=\frac{\lambda_{obs}}{1215.67}-1")

zabs = wave / 1215.67 - 1

fig4 = go.Figure()

fig4.add_trace(
    go.Scatter(
        x=wave,
        y=zabs,
        mode="lines"
    )
)

st.plotly_chart(fig4, width="stretch")

# --------------------------------------------------

with st.expander("🔬 Full FITS Header"):
    st.json(dict(header))
