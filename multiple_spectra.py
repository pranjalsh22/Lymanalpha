
import os
import numpy as np
import pandas as pd
import streamlit as st
from astropy.io import fits
import plotly.graph_objects as go

st.set_page_config(
    page_title=".fit file plots",
    layout="wide"
)

SPEC_DIR = "spec"

def load_fits(source):

    with fits.open(source) as hdul: #HDU is Header data unit list. in this case there's only one HDU 

        for hdu in hdul:

            if hdu.data is not None:

                arr = np.array(
                    hdu.data,
                    dtype=np.float64
                )

                if arr.ndim == 2: #ndim is no. of dimensions

                    if arr.shape[0] == 1:
                        arr = arr[0]

                    elif arr.shape[1] == 1:
                        arr = arr[:, 0]

                return (
                    np.squeeze(arr),
                    hdu.header
                )

    raise ValueError(
        f"No spectrum found in {source}"
    )


# --------------------------------------------------
# WAVELENGTH ARRAY
# --------------------------------------------------

def wavelength_array(header, n):

    return 10 ** (
        header["CRVAL1"]
        + np.arange(n) * header["CDELT1"]
    )


def velocity_spacing(header):

    return (
        299792.458
        * np.log(10)
        * header["CDELT1"]
    )

def quality_label(snr):

    if snr > 20:
        return "Excellent(>20)"

    elif snr > 10:
        return "Good(>10)"

    elif snr > 5:
        return "Moderate(>5)"

    else:
        return "Poor(<5)"


def find_pairs(folder):

    pairs = {}

    if not os.path.isdir(folder):
        return {}

    for fname in os.listdir(folder):

        if not fname.endswith(".fits"):
            continue

        full = os.path.join(
            folder,
            fname
        )

        if "_flux" in fname:

            key = fname.replace(
                "_flux.fits",
                ""
            )

            pairs.setdefault(
                key,
                {}
            )

            pairs[key]["flux"] = full

        elif "_error" in fname:

            key = fname.replace(
                "_error.fits",
                ""
            )

            pairs.setdefault(
                key,
                {}
            )

            pairs[key]["error"] = full

    return {

        k: v

        for k, v in pairs.items()

        if (
            "flux" in v
            and
            "error" in v
        )
    }

# part 2

st.title(".fit file plots")

pairs = find_pairs(SPEC_DIR)

if len(pairs) == 0:

    st.error(
        f"No matched spectra found in {SPEC_DIR}"
    )

    st.stop()

spectra = []
summary_rows = []

for key, files in pairs.items():

    try:

        flux, header = load_fits(
            files["flux"]
        )

        error, _ = load_fits(
            files["error"]
        )

        if len(flux) != len(error):
            continue

        n = len(flux)

        wave = wavelength_array(
            header,
            n
        )

        snr = np.full(
            n,
            np.nan
        )

        good = (
            np.isfinite(flux)
            &
            np.isfinite(error)
            &
            (error > 0)
        )

        snr[good] = (
            flux[good]
            /
            error[good]
        )

        median_snr = float(
            np.nanmedian(snr)
        )

        masked_fraction = (
            np.sum(
                ~np.isfinite(flux)
            )
            /
            len(flux)
            * 100
        )

        object_name = header.get(
            "OBJECT",
            key
        )

        instrument = header.get(
            "INSTRUME",
            "Unknown"
        )

        dv = velocity_spacing(
            header
        )

        spectra.append({

            "object":
                object_name,

            "instrument":
                instrument,

            "wave":
                wave,

            "flux":
                flux,

            "error":
                error,

            "snr":
                snr,

            "header":
                header,

            "median_snr":
                median_snr,

            "masked_fraction":
                masked_fraction,

            "dv":
                dv
        })

        summary_rows.append({

            "Object":
                object_name,

            "Instrument":
                instrument,

            "Pixels":
                len(flux),

            "Lambda Min":
                wave.min(),

            "Lambda Max":
                wave.max(),

            "Median S/N":
                median_snr,

            "Masked %":
                masked_fraction,

            "dv (km/s)":
                dv
        })

    except Exception as e:

        st.warning(
            f"{key}: {e}"
        )

summary_df = pd.DataFrame(
    summary_rows
)

st.header(
    "Dataset Summary"
)

st.dataframe(
    summary_df,
    use_container_width=True
)

#st.header(
#    "S/N Ranking"
#)

#st.dataframe(

#    summary_df.sort_values(
#        "Median S/N",
#        ascending=False
#    ),
#
#    use_container_width=True
#)

#part 3

st.header("Flux plots")

for spec in spectra:

    with st.expander(

        f"{spec['object']} ({spec['instrument']})",

        expanded=False
    ):

        flux = spec["flux"]
        error = spec["error"]
        wave = spec["wave"]
        snr = spec["snr"]

        finite_pixels = int(
            np.sum(
                np.isfinite(flux)
            )
        )

        masked_pixels = int(
            np.sum(
                ~np.isfinite(flux)
            )
        )

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "Median S/N",
            f"{spec['median_snr']:.2f}"
        )

        c2.metric(
            "Quality",
            quality_label(
                spec["median_snr"]
            )
        )

        c3.metric(
            "Masked %",
            f"{spec['masked_fraction']:.2f}"
        )

        c4.metric(
            "Velocity Spacing",
            f"{spec['dv']:.2f}"
        )

        st.write(
            f"Pixels: {len(flux)}"
        )

        st.write(
            f"Finite Pixels: {finite_pixels}"
        )

        st.write(
            f"Masked Pixels: {masked_pixels}"
        )

        st.write(
            f"Wavelength Range: "
            f"{wave.min():.1f}"
            f"–"
            f"{wave.max():.1f} Å"
        )

        # -----------------------
        # FLUX
        # -----------------------

        mask = np.isfinite(flux)

        fig1 = go.Figure()

        fig1.add_trace(

            go.Scatter(

                x=wave[mask],

                y=flux[mask],

                mode="lines",

                name="Flux"
            )
        )

        fig1.update_layout(

            title="Flux Spectrum",

            xaxis_title=
                "Observed Wavelength (Å)",

            yaxis_title=
                "Flux"
        )

        st.plotly_chart(
            fig1,
            use_container_width=True
        )

        # -----------------------
        # ERROR
        # -----------------------

  #      emask = np.isfinite(error)

  #      fig2 = go.Figure()

  #      fig2.add_trace(
#
  #          go.Scatter(

  #              x=wave[emask],

   #             y=error[emask],

   #             mode="lines",

   #             name="Error"
    #        )
    #    )

     #   fig2.update_layout(

     #       title="Error Spectrum",

     #       xaxis_title=
     #           "Observed Wavelength (Å)",

      #      yaxis_title=
     #           "1σ Error"
      #  )

      #  st.plotly_chart(
     #       fig2,
      #      use_container_width=True
     #   )

        # -----------------------
        # SNR
        # -----------------------

        smask = np.isfinite(snr)

        fig3 = go.Figure()

        fig3.add_trace(

            go.Scatter(

                x=wave[smask],

                y=snr[smask],

                mode="lines",

                name="SNR"
            )
        )

        fig3.update_layout(

            title="Signal-to-Noise Ratio",

            xaxis_title=
                "Observed Wavelength (Å)",

            yaxis_title=
                "S/N"
        )

        st.plotly_chart(
            fig3,
            use_container_width=True
        )

        # -----------------------
        # ANALYSIS
        # -----------------------

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
            f"{spec['dv']:.2f} km/s."
        )

        with st.expander(
            "FITS Header"
        ):
            st.json(
                dict(spec["header"])
            )
