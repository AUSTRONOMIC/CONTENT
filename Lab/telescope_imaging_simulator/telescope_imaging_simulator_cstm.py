#!/usr/bin/env python3
"""
Telescope Imaging Simulator  v2.0
==================================
Physically accurate simulation of diffraction and exposure efficiency
for astrophotography telescopes with one-shot colour (OSC) sensors.

Three independently togglable filter layers:
  1. DIFFRACTION   Wavelength-integrated, chromatic Airy-disk PSF convolution.
                   Each RGB channel is convolved with its own spectrally
                   weighted effective PSF (QE x Bayer filter x Airy kernel).
  2. EXPOSURE      Per-channel relative signal gain based on f-ratio, exposure
                   time, and wavelength-dependent quantum efficiency.
  3. NOISE         Poisson photon shot noise + Gaussian read noise.

Scientific units
----------------
  - Aperture / focal length : mm
  - FOV                     : arcsec  (independent x and y)
  - Wavelength              : nm  (internally converted to m)
  - Pixel size              : µm  (for diagnostics and noise scaling)
  - QE                      : dimensionless fraction in [0, 1]
  - Electron counts         : e-

Physics references
------------------
  Diffraction:
    Airy (1835); Born & Wolf, Principles of Optics, Ch. 8;
    PSF for annular aperture: I(theta) = [A_outer(u) - A_inner(u)]^2 / (1-eps^2)^2
    where A(u) = 2*J1(u)/u,  u = pi*D*theta/lambda

  Exposure (extended source):
    Signal per pixel [e-] proportional to I_sky * Omega_pix * A_tel * t * QE
    Omega_pix = (plate_scale_rad)^2  ->  signal ∝ (D/f)^2 * t * QE  = t * QE / N^2

  Sensor noise (Hamamatsu / standard CCD model):
    SNR = (P * QE * t) / sqrt(P*QE*t + D*t + Nr^2)
    Shot noise = sqrt(signal_e);  read noise = Nr [e- RMS];
    dark noise = sqrt(dark_current * t)

Dependencies: numpy, scipy, Pillow

Usage
-----
  Edit the CONFIGURATION section below and run:
      python telescope_imaging_simulator_v2.py
"""

# =============================================================================
# STANDARD LIBRARY
# =============================================================================

import math
from pathlib import Path

# =============================================================================
# THIRD-PARTY
# =============================================================================

import numpy as np
from PIL import Image
from scipy.signal import fftconvolve
from scipy.special import j1


# =============================================================================
#  CONFIGURATION  -- Edit here
# =============================================================================
OTA=6
OBJECT = "SIRUS_A"
AP = [71, 91, 103, 150, 203.2,235]
FC = [490, 609, 700, 1800, 2032, 2350]
FOV = {"JUPITER":46.91, "NGC7293":1500, "SIRUS_A":97.47}

INPUT_IMAGE  = "SIRIUS_A_97P47AS.png" #JUPITER_46P91AS; NGC7293_1500AS; SIRIUS_A_97P47AS
OUTPUT_IMAGE = INPUT_IMAGE.split(".")[0]+f"_{OTA}.png"

# --- Telescope geometry ---
APERTURE_MM           = AP[OTA-1]    # primary aperture diameter              [mm]
FOCAL_LENGTH_MM       = FC[OTA-1]   # focal length                           [mm]
CENTRAL_OBSCURATION   = 0.0     # secondary/primary diameter ratio       [0, 1)
FOV_X_ARCSEC          = FOV[OBJECT]  # horizontal field of view               [arcsec]
FOV_Y_ARCSEC          = None    # vertical FOV; None -> infer from image aspect ratio
SEEING_FWHM_ARCSEC    = 0.0     # atmospheric seeing FWHM (0 = disabled) [arcsec]

# --- Sensor ---
PIXEL_SIZE_UM         = 2.90    # physical pixel pitch (for diagnostics and noise) [µm]

# Quantum-efficiency model for the TARGET telescope/sensor:
#   Built-in presets:  "ASI585MC"   (IMX585, BSI, peak QE ~91%)
#                      "ASI533MC"   (IMX533, BSI, peak QE ~85%)
#                      "ASI462MC"   (IMX462, BSI, peak QE ~80%)
#                      "generic_80" (representative 80%-peak sensor)
#                      "generic_60" (representative 60%-peak sensor)
#   Custom:            A list of (wavelength_nm, QE_fraction) tuples
#                      e.g. [(400, 0.55), (500, 0.82), (600, 0.78), (700, 0.60)]
QE_MODEL              = "ASI585MC"

# --- Exposure ---
EXPOSURE_TIME_S       = 60.0    # integration time per sub-frame         [s]
SYSTEM_THROUGHPUT     = 0.95    # scalar throughput (optics x atmosphere) [0, 1]

# --- Reference system (defines "image as input = this system's output") ---
REFERENCE_APERTURE_MM    = 71.0
REFERENCE_FOCAL_LENGTH_MM = 490.0
REFERENCE_EXPOSURE_TIME_S = 60.0
REFERENCE_QE_MODEL        = "ASI585MC"

# --- Noise layer parameters ---
FULL_WELL_ELECTRONS      = 40000  # sensor full-well depth              [e-]
                                  # maps pixel value 1.0 to this count
READ_NOISE_ELECTRONS     = 0.8    # read noise RMS                      [e-/px]
DARK_CURRENT_E_PER_S     = 0.002  # dark current at operating temp      [e-/px/s]

# --- Filter layer enable/disable flags ---
ENABLE_DIFFRACTION   = True    # toggle the diffraction PSF layer
ENABLE_EXPOSURE      = False    # toggle the exposure/QE scaling layer
ENABLE_NOISE         = False   # toggle the photon/read-noise layer

# --- Display stretch applied to the final image ---
#   "none"      no stretch, clip to [0, 1]
#   "normalize" scale so peak = 1
#   "asinh"     arcsinh stretch  (recommended for nebulae)
#   "log"       logarithmic stretch
DISPLAY_STRETCH_MODE  = "normalize"#"asinh"
DISPLAY_ASINH_BETA    = 0.5     # controls asinh softness  (larger = more aggressive)
DISPLAY_LOG_A         = 1000.0  # log stretch parameter:   y = log(1+A*x)/log(1+A)
GAMMA                 = 1.0     # output gamma correction  (1.0 = linear)

PRINT_DIAGNOSTICS     = True


# =============================================================================
#  SPECTRAL DATA  (internal constants, not intended for user editing)
# =============================================================================

# For each RGB channel we define a set of spectral sample wavelengths
# that span the nominal passband of the Bayer colour filter.
# These samples are used to:
#   (a) weight individual monochromatic Airy kernels -> effective PSF
#   (b) compute the channel-averaged effective QE for the exposure layer.
#
# Bayer passbands are approximate; exact edges are sensor-dependent.
_CHANNEL_SAMPLES_NM = {
    "R": [590, 610, 630, 650, 670, 690, 710],   # Red   ~580-720 nm
    "G": [500, 515, 530, 545, 560, 575, 590],   # Green ~490-600 nm
    "B": [390, 410, 430, 450, 470, 490, 510],   # Blue  ~380-520 nm
}

# Bayer colour-filter transmissions at the sample wavelengths above.
# Derived from representative RGGB silicon Bayer filter datasheets
# (Sony IMX series) and smoothly interpolated.
# Format: transmission fraction in [0, 1]
_BAYER_TRANSMISSION = {
    #      590   610   630   650   670   690   710  nm
    "R": [0.10, 0.50, 0.85, 0.97, 1.00, 0.99, 0.95],
    #      500   515   530   545   560   575   590  nm
    "G": [0.80, 0.95, 1.00, 0.99, 0.93, 0.75, 0.45],
    #      390   410   430   450   470   490   510  nm
    "B": [0.65, 0.90, 1.00, 0.99, 0.88, 0.60, 0.25],
}

# QE curves: lists of (wavelength_nm, QE_fraction) pairs.
# Derived from published ZWO / Sony spectral response graphs
# and cross-checked against the astrojolo.com QE database.

# IMX585 (back-illuminated STARVIS 2) -- ZWO ASI585MC
_QE_ASI585MC = [
    (370, 0.58), (380, 0.65), (400, 0.76), (420, 0.83),
    (450, 0.88), (480, 0.91), (510, 0.91), (540, 0.90),
    (570, 0.88), (600, 0.85), (630, 0.82), (660, 0.78),
    (690, 0.72), (720, 0.63), (750, 0.52), (780, 0.38),
    (800, 0.28), (850, 0.12),
]

# IMX533 (back-illuminated) -- ZWO ASI533MC Pro
_QE_ASI533MC = [
    (370, 0.44), (380, 0.52), (400, 0.63), (420, 0.72),
    (450, 0.80), (480, 0.84), (510, 0.85), (540, 0.84),
    (570, 0.82), (600, 0.79), (630, 0.75), (660, 0.70),
    (690, 0.63), (720, 0.53), (750, 0.41), (780, 0.28),
    (800, 0.18), (850, 0.07),
]

# IMX462 (back-illuminated STARVIS) -- ZWO ASI462MC
_QE_ASI462MC = [
    (370, 0.40), (380, 0.48), (400, 0.58), (420, 0.68),
    (450, 0.76), (480, 0.80), (510, 0.81), (540, 0.80),
    (570, 0.78), (600, 0.75), (630, 0.71), (660, 0.66),
    (690, 0.59), (720, 0.50), (750, 0.38), (780, 0.24),
    (800, 0.15), (850, 0.05),
]

# Generic representative sensors
_QE_GENERIC_80 = [
    (370, 0.35), (400, 0.55), (450, 0.73), (500, 0.80),
    (550, 0.80), (600, 0.77), (650, 0.72), (700, 0.62),
    (750, 0.44), (800, 0.22), (850, 0.07),
]
_QE_GENERIC_60 = [
    (370, 0.22), (400, 0.38), (450, 0.52), (500, 0.60),
    (550, 0.60), (600, 0.57), (650, 0.52), (700, 0.44),
    (750, 0.30), (800, 0.14), (850, 0.04),
]

_QE_PRESETS = {
    "ASI585MC":   _QE_ASI585MC,
    "ASI533MC":   _QE_ASI533MC,
    "ASI462MC":   _QE_ASI462MC,
    "generic_80": _QE_GENERIC_80,
    "generic_60": _QE_GENERIC_60,
}


# =============================================================================
#  UTILITIES
# =============================================================================

def arcsec_to_rad(x: float) -> float:
    """Convert arcseconds to radians."""
    return x * math.pi / (180.0 * 3600.0)


def rad_to_arcsec(x: float) -> float:
    """Convert radians to arcseconds."""
    return x * 180.0 * 3600.0 / math.pi


def resolve_path(path_str: str, base_dir: Path) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else (base_dir / p).resolve()


def load_image(path: Path):
    """
    Load an image as a float64 array with values in [0, 1].
    Returns (rgb_array [H, W, 3], alpha [H, W, 1] or None).
    """
    img = Image.open(path)
    if img.mode not in ("L", "RGB", "RGBA"):
        img = img.convert("RGBA" if img.mode in ("RGBA", "LA") else "RGB")
    arr = np.asarray(img).astype(np.float64) / 255.0
    alpha = None
    if arr.ndim == 3 and arr.shape[2] == 4:
        alpha = arr[..., 3:4].copy()
        arr   = arr[..., :3]
    if arr.ndim == 2:                          # grayscale -> stack to RGB
        arr = np.stack([arr, arr, arr], axis=2)
    return arr, alpha


def save_image(path: Path, arr: np.ndarray, alpha=None):
    arr = np.clip(arr, 0.0, 1.0)
    if alpha is not None:
        alpha = np.clip(alpha, 0.0, 1.0)
        rgba  = np.concatenate([arr, alpha], axis=2)
        out   = (rgba * 255.0 + 0.5).astype(np.uint8)
        Image.fromarray(out, mode="RGBA").save(path)
    else:
        out = (arr * 255.0 + 0.5).astype(np.uint8)
        Image.fromarray(out, mode="RGB").save(path)


# =============================================================================
#  QE INTERPOLATION
# =============================================================================

def build_qe_fn(model):
    """
    Build a callable qe(lambda_nm) -> float from a preset name or a
    user-supplied list of (nm, QE) control points.

    Parameters
    ----------
    model : str or list of (float, float)
        Preset name string or list of (wavelength_nm, QE_fraction) pairs.

    Returns
    -------
    callable
        qe_fn(lambda_nm) -> float, linearly interpolated, clamped to 0
        outside the defined range.
    """
    if isinstance(model, str):
        if model not in _QE_PRESETS:
            raise ValueError(
                f"Unknown QE model '{model}'.  "
                f"Available presets: {list(_QE_PRESETS.keys())}"
            )
        pts = _QE_PRESETS[model]
    else:
        pts = list(model)

    wl = np.array([p[0] for p in pts], dtype=np.float64)
    qe = np.array([p[1] for p in pts], dtype=np.float64)

    def qe_fn(lam_nm: float) -> float:
        return float(np.interp(lam_nm, wl, qe, left=0.0, right=0.0))

    return qe_fn


# =============================================================================
#  BAYER FILTER TRANSMISSION
# =============================================================================

def bayer_tx(channel: str, lam_nm: float) -> float:
    """
    Linear interpolation of the Bayer colour filter transmission for
    the given channel ('R', 'G', or 'B') at wavelength lam_nm.
    Returns 0 outside the defined sample range.
    """
    samples = _CHANNEL_SAMPLES_NM[channel]
    trans   = _BAYER_TRANSMISSION[channel]
    return float(np.interp(lam_nm, samples, trans, left=0.0, right=0.0))


# =============================================================================
#  PSF CONSTRUCTION
# =============================================================================

def _monochromatic_airy_psf(
    pixel_scale_rad: float,
    aperture_m: float,
    wavelength_m: float,
    obscuration: float,
    half_px: int,
) -> np.ndarray:
    """
    Compute a normalised Fraunhofer (Airy) PSF kernel for a circular
    or annular aperture.

    Theory
    ------
    For a full circular aperture (obscuration = 0):
        I(theta) = [2*J1(u) / u]^2
    For an annular aperture (obscuration = eps, inner/outer ratio):
        Amplitude = [2*J1(u)/u  -  eps^2 * 2*J1(eps*u)/(eps*u)] / (1 - eps^2)
        I(theta) = Amplitude^2
    where u = pi * D * theta / lambda  and  theta is the off-axis angle [rad].

    Note: At theta = 0 (u -> 0):
        2*J1(u)/u -> 1  (L'Hopital / Taylor expansion of J1)
        So the annular amplitude -> (1 - eps^2)/(1 - eps^2) = 1  (peak normalised).

    The kernel is then renormalised to sum to unity (flux conservation).

    Parameters
    ----------
    pixel_scale_rad : float
        Angular size of one image pixel [rad/px].
    aperture_m : float
        Telescope aperture (outer) diameter [m].
    wavelength_m : float
        Monochromatic wavelength [m].
    obscuration : float
        Central obstruction ratio (inner/outer diameter), in [0, 1).
    half_px : int
        Half-size of the kernel in pixels.  Full kernel: (2*half_px+1)^2.

    Returns
    -------
    np.ndarray, shape (2*half_px+1, 2*half_px+1)
        Normalised PSF kernel (sums to 1.0).
    """
    y, x   = np.mgrid[-half_px : half_px + 1, -half_px : half_px + 1]
    theta  = np.hypot(x, y) * pixel_scale_rad     # off-axis angle [rad]
    u      = math.pi * aperture_m * theta / wavelength_m

    eps    = float(obscuration)
    psf    = np.empty_like(u, dtype=np.float64)

    nz = u > 0.0
    z  = ~nz

    if eps == 0.0:
        # Unobscured circular aperture
        psf[z]  = 1.0
        psf[nz] = (2.0 * j1(u[nz]) / u[nz]) ** 2
    else:
        # Annular aperture
        psf[z]  = 1.0
        u_nz    = u[nz]
        amp_out = 2.0 * j1(u_nz) / u_nz
        amp_in  = eps ** 2 * 2.0 * j1(eps * u_nz) / (eps * u_nz)
        amp     = (amp_out - amp_in) / (1.0 - eps ** 2)
        psf[nz] = amp ** 2

    psf = np.clip(psf, 0.0, None)
    s   = psf.sum()
    if s > 0.0:
        psf /= s
    return psf


def _gaussian_psf(pixel_scale_rad: float, fwhm_arcsec: float, half_px: int) -> np.ndarray:
    """
    Gaussian PSF kernel representing atmospheric seeing.
    Returns a delta function (identity kernel) if fwhm_arcsec <= 0.
    """
    if fwhm_arcsec <= 0.0:
        k = np.zeros((2 * half_px + 1, 2 * half_px + 1), dtype=np.float64)
        k[half_px, half_px] = 1.0
        return k
    sigma_rad = arcsec_to_rad(fwhm_arcsec) / (2.0 * math.sqrt(2.0 * math.log(2.0)))
    y, x      = np.mgrid[-half_px : half_px + 1, -half_px : half_px + 1]
    r2        = (x ** 2 + y ** 2) * pixel_scale_rad ** 2
    g         = np.exp(-0.5 * r2 / sigma_rad ** 2)
    g        /= g.sum()
    return g


def psf_preflight_check(
    pixel_scale_rad: float,
    aperture_m: float,
    seeing_fwhm_arcsec: float,
    image_w_px: int,
    image_h_px: int,
    pixel_size_um: float,
    focal_length_mm: float,
) -> dict:
    """
    Run a pre-simulation geometry sanity check and return a report dict.

    Computes, for the green channel (550 nm representative wavelength):
      - Airy first-zero radius in arcsec and pixels
      - The ideal PSF kernel half-size the simulator would normally build
      - Whether clamping will be applied and how severe it is
      - How many Airy radii the image spans
      - The correct FOV for this telescope+sensor combination
      - A recommendation when the PSF is larger than the image

    Parameters
    ----------
    pixel_scale_rad : float
        Angular pixel scale [rad/px].
    aperture_m : float
        Telescope aperture [m].
    seeing_fwhm_arcsec : float
        Atmospheric seeing FWHM (0 = disabled).
    image_w_px, image_h_px : int
        Image dimensions in pixels.
    pixel_size_um : float
        Physical pixel pitch [µm], used to compute the physically correct FOV.
    focal_length_mm : float
        Telescope focal length [mm].

    Returns
    -------
    dict with keys: airy_zero_arcsec, airy_zero_px, fwhm_arcsec, fwhm_px,
                    ideal_half_px, clamped_half_px, image_airy_radii,
                    psf_fills_image, physically_correct_fov_arcsec,
                    warnings (list of str)
    """
    lam_m       = 550e-9   # green channel representative wavelength
    fov_arcsec  = rad_to_arcsec(pixel_scale_rad) * image_w_px

    airy_zero_rad    = 1.22 * lam_m / aperture_m
    airy_zero_arcsec = rad_to_arcsec(airy_zero_rad)
    airy_zero_px     = airy_zero_rad / pixel_scale_rad
    fwhm_arcsec      = rad_to_arcsec(1.028 * lam_m / aperture_m)
    fwhm_px          = fwhm_arcsec / rad_to_arcsec(pixel_scale_rad)

    seeing_sigma_px = 0.0
    if seeing_fwhm_arcsec > 0.0:
        seeing_sigma_px = (
            arcsec_to_rad(seeing_fwhm_arcsec)
            / (2.0 * math.sqrt(2.0 * math.log(2.0)))
            / pixel_scale_rad
        )

    ideal_half_px   = int(math.ceil(max(12.0 * airy_zero_px, 6.0 * seeing_sigma_px, 4.0)))
    max_allowed     = min(image_w_px, image_h_px) // 2 - 1
    clamped_half_px = min(ideal_half_px, max_allowed)
    image_airy_radii = (min(image_w_px, image_h_px) / 2.0) / max(airy_zero_px, 1e-9)

    # Encircled energy fraction within the clamped kernel radius
    # Approximated from the theoretical Airy pattern series:
    # EE(r_in_airy_units) ~ 1 - J0(pi*r')^2 - J1(pi*r')^2  but we simplify
    # to a lookup over known fractions: 84% within 1st zero, 91%/94%/96%...
    clamped_airy_radii = clamped_half_px / max(airy_zero_px, 1e-9)
    if clamped_airy_radii >= 12.0:
        captured_pct = 99.0
    elif clamped_airy_radii >= 5.0:
        captured_pct = 97.0
    elif clamped_airy_radii >= 2.0:
        captured_pct = 94.0
    elif clamped_airy_radii >= 1.0:
        captured_pct = 84.0
    elif clamped_airy_radii >= 0.5:
        captured_pct = 50.0
    else:
        captured_pct = clamped_airy_radii * 100.0  # rough linear below 0.5

    # Physically correct plate scale and FOV for this telescope + sensor
    correct_ps_arcsec  = 206265.0 * (pixel_size_um * 1e-6) / (focal_length_mm * 1e-3)
    correct_fov_arcsec = correct_ps_arcsec * image_w_px

    warnings = []
    psf_fills_image = airy_zero_px > (min(image_w_px, image_h_px) / 2.0)

    if psf_fills_image:
        ideal_kb = (2 * ideal_half_px + 1) ** 2 * 8 / 1e9
        warnings.append(
            f"PSF FIRST ZERO ({airy_zero_arcsec:.3f}\") is LARGER than the "
            f"image half-width ({fov_arcsec/2:.3f}\"). "
            f"The unclamped kernel would be {2*ideal_half_px+1}x{2*ideal_half_px+1} px "
            f"({ideal_kb:.1f} GB). Kernel CLAMPED to {2*clamped_half_px+1}x{2*clamped_half_px+1} px. "
            f"Only {captured_pct:.0f}% of PSF energy falls within the kernel. "
            f"The rest is distributed outside the image boundaries -- this is "
            f"physically correct but means the convolution result will be dim."
        )

    if abs(correct_ps_arcsec - rad_to_arcsec(pixel_scale_rad)) / correct_ps_arcsec > 0.10:
        warnings.append(
            f"FOV MISMATCH: The stated FOV implies a plate scale of "
            f"{rad_to_arcsec(pixel_scale_rad):.5f}\"/px, but the physically correct "
            f"plate scale for {focal_length_mm:.0f}mm focal length + {pixel_size_um:.1f}µm pixels "
            f"is {correct_ps_arcsec:.4f}\"/px. "
            f"Correct FOV for a {image_w_px}x{image_h_px} px image: "
            f"{correct_fov_arcsec:.1f}\" = {correct_fov_arcsec/60:.2f}'. "
            f"The current FOV is {fov_arcsec/correct_fov_arcsec:.0f}x smaller than physical reality."
        )

    return {
        "airy_zero_arcsec":             airy_zero_arcsec,
        "airy_zero_px":                 airy_zero_px,
        "fwhm_arcsec":                  fwhm_arcsec,
        "fwhm_px":                      fwhm_px,
        "ideal_half_px":                ideal_half_px,
        "clamped_half_px":              clamped_half_px,
        "was_clamped":                  clamped_half_px < ideal_half_px,
        "clamped_airy_radii":           clamped_airy_radii,
        "captured_pct":                 captured_pct,
        "image_airy_radii":             image_airy_radii,
        "psf_fills_image":              psf_fills_image,
        "physically_correct_ps_arcsec": correct_ps_arcsec,
        "physically_correct_fov_arcsec": correct_fov_arcsec,
        "warnings":                     warnings,
    }


def build_channel_effective_psf(
    channel: str,
    pixel_scale_rad: float,
    aperture_m: float,
    obscuration: float,
    seeing_fwhm_arcsec: float,
    qe_fn,
    max_kernel_half_px: int | None = None,
) -> tuple:
    """
    Build the spectrally-integrated effective PSF for one RGB colour channel.

    The effective PSF is a weighted sum of monochromatic Airy kernels:

        PSF_eff(channel) = SUM_lambda [ w(lambda) * Airy(lambda) ]
                         / SUM_lambda [ w(lambda) ]

    where the weight at each sample wavelength is:
        w(lambda) = QE(lambda) * Bayer_transmission(channel, lambda)

    This correctly models:
      - Chromatic scaling of the Airy disk (larger at longer lambda)
      - Spectral sensitivity of the sensor (QE curve)
      - Spectral bandpass of the Bayer colour filter

    If atmospheric seeing is enabled, the result is then convolved
    with a Gaussian kernel of the specified FWHM.

    The kernel is HARD-CLAMPED to max_kernel_half_px when the PSF is
    larger than the image.  This prevents memory exhaustion and is
    physically valid: the clipped kernel correctly represents the PSF
    within the image boundaries; energy outside is simply not modelled
    (which corresponds to PSF light falling outside the sensor).

    Parameters
    ----------
    channel : str
        'R', 'G', or 'B'.
    pixel_scale_rad : float
        Angular pixel scale [rad/px].
    aperture_m : float
        Telescope aperture [m].
    obscuration : float
        Central obstruction ratio.
    seeing_fwhm_arcsec : float
        Atmospheric seeing FWHM (0 = diffraction-limited).
    qe_fn : callable
        QE function from build_qe_fn().
    max_kernel_half_px : int or None
        Hard upper bound on kernel half-size.  None = no cap (original behaviour).

    Returns
    -------
    (psf_kernel, diagnostics_dict)
    """
    samples_nm = _CHANNEL_SAMPLES_NM[channel]

    # Kernel size: must be large enough to capture ~12 Airy zeros
    # of the shortest (smallest) wavelength in the channel, plus seeing.
    lam_min_m         = min(samples_nm) * 1e-9
    airy_zero_rad     = 1.22 * lam_min_m / aperture_m   # first zero [rad]
    airy_zero_px      = airy_zero_rad / pixel_scale_rad  # first zero [px]
    seeing_sigma_px   = 0.0
    if seeing_fwhm_arcsec > 0.0:
        seeing_sigma_px = (
            arcsec_to_rad(seeing_fwhm_arcsec)
            / (2.0 * math.sqrt(2.0 * math.log(2.0)))
            / pixel_scale_rad
        )
    half_px = max(4, int(math.ceil(max(12.0 * airy_zero_px,
                                       6.0 * seeing_sigma_px))))
    # Apply the hard cap if requested
    was_clamped = False
    if max_kernel_half_px is not None and half_px > max_kernel_half_px:
        half_px     = max(4, max_kernel_half_px)
        was_clamped = True

    accum       = np.zeros((2 * half_px + 1, 2 * half_px + 1), dtype=np.float64)
    weight_sum  = 0.0
    diag_weights = []
    diag_fwhms   = []

    for lam_nm in samples_nm:
        w = qe_fn(lam_nm) * bayer_tx(channel, lam_nm)
        if w <= 0.0:
            continue
        lam_m   = lam_nm * 1e-9
        kernel  = _monochromatic_airy_psf(
            pixel_scale_rad = pixel_scale_rad,
            aperture_m      = aperture_m,
            wavelength_m    = lam_m,
            obscuration     = obscuration,
            half_px         = half_px,
        )
        accum       += w * kernel
        weight_sum  += w

        # Diffraction-limited FWHM (Gaussian approximation: FWHM ~ 1.028 lambda/D)
        diag_fwhms.append(rad_to_arcsec(1.028 * lam_m / aperture_m))
        diag_weights.append(w)

    if weight_sum == 0.0:
        raise ValueError(
            f"Zero spectral weight for channel {channel}.  "
            "Check that the QE model has non-zero values in the channel passband."
        )

    # Normalised weighted-average PSF (flux-conserving)
    accum /= weight_sum

    # Convolve with seeing kernel if enabled
    if seeing_fwhm_arcsec > 0.0:
        g     = _gaussian_psf(pixel_scale_rad, seeing_fwhm_arcsec, half_px)
        accum = fftconvolve(accum, g, mode="same")
        accum = np.clip(accum, 0.0, None)
        s     = accum.sum()
        if s > 0.0:
            accum /= s

    w_arr = np.array(diag_weights)
    diag  = {
        "channel":               channel,
        "samples_nm":            samples_nm,
        "spectral_weights":      w_arr.tolist(),
        "mean_qe_x_bayer":       float(w_arr.mean()) if len(w_arr) else 0.0,
        "diffraction_fwhm_min":  min(diag_fwhms),
        "diffraction_fwhm_max":  max(diag_fwhms),
        "first_zero_arcsec":     rad_to_arcsec(1.22 * lam_min_m / aperture_m),
        "airy_zero_px":          airy_zero_px,
        "kernel_half_px":        half_px,
        "kernel_total_px":       2 * half_px + 1,
        "was_clamped":           was_clamped,
    }
    return accum, diag


# =============================================================================
#  FILTER 1: DIFFRACTION
# =============================================================================

def apply_diffraction_filter(
    image: np.ndarray,
    pixel_scale_rad: float,
    aperture_m: float,
    obscuration: float,
    seeing_fwhm_arcsec: float,
    qe_fn,
    max_kernel_half_px: int | None = None,
) -> tuple:
    """
    Convolve each RGB channel with its spectrally-integrated effective PSF.

    Physical basis
    --------------
    The diffraction PSF of a circular aperture scales with wavelength:
        Airy disk angular radius ~ 1.22 * lambda / D

    For a broad-band colour channel the effective PSF is therefore slightly
    larger and more Gaussian-like than a purely monochromatic kernel at the
    channel centre wavelength.  This is often called *chromatic smearing*.
    Weighting by QE(lambda) and Bayer filter transmission ensures that
    photons where the sensor is most sensitive dominate the PSF shape.

    Kernel size cap
    ---------------
    When max_kernel_half_px is provided and the natural PSF kernel would
    exceed it, the kernel is clamped.  This is physically correct when the
    PSF is larger than the image: the clamped kernel models the portion of
    the PSF that falls within the image field; light that would fall outside
    the sensor is simply not captured.  The kernel is still re-normalised to
    sum to 1, which means the output will be dimmer than reality (the missing
    energy went off-sensor).  This is correct behaviour for a simulation that
    asks "what does the sensor record?" -- the answer is: only the fraction
    of PSF energy that landed on the detector.

    Parameters
    ----------
    image : np.ndarray, shape (H, W, 3), float64, values in [0, 1]
    pixel_scale_rad : float
    aperture_m : float
    obscuration : float
    seeing_fwhm_arcsec : float
    qe_fn : callable
    max_kernel_half_px : int or None
        Hard upper limit on kernel half-size in pixels.
        Recommended: min(H, W) // 2 - 1.

    Returns
    -------
    (convolved_image [H, W, 3], list_of_per-channel_diagnostics)
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Diffraction filter requires a 3-channel (H, W, 3) RGB image.")

    out   = np.empty_like(image)
    diags = []

    for idx, ch in enumerate(("R", "G", "B")):
        psf, d = build_channel_effective_psf(
            channel             = ch,
            pixel_scale_rad     = pixel_scale_rad,
            aperture_m          = aperture_m,
            obscuration         = obscuration,
            seeing_fwhm_arcsec  = seeing_fwhm_arcsec,
            qe_fn               = qe_fn,
            max_kernel_half_px  = max_kernel_half_px,
        )
        convolved = fftconvolve(image[..., idx], psf, mode="same")
        out[..., idx] = np.clip(convolved, 0.0, None)
        diags.append(d)

    return out, diags


# =============================================================================
#  FILTER 2: EXPOSURE
# =============================================================================

def _channel_effective_qe(channel: str, qe_fn) -> float:
    """
    Compute the mean QE x Bayer-filter product over the channel's
    spectral sample points.  This is the effective quantum efficiency
    for signal calculation in that colour channel.
    """
    samples = _CHANNEL_SAMPLES_NM[channel]
    vals    = [qe_fn(lam) * bayer_tx(channel, lam) for lam in samples]
    return float(np.mean(vals))


def apply_exposure_filter(
    image: np.ndarray,
    aperture_mm: float,
    focal_length_mm: float,
    exposure_time_s: float,
    throughput: float,
    qe_fn,
    ref_aperture_mm: float,
    ref_focal_length_mm: float,
    ref_exposure_time_s: float,
    ref_qe_fn,
) -> tuple:
    """
    Scale each colour channel by its relative signal gain.

    Physical model (extended source)
    ---------------------------------
    For a spatially-resolved extended source with surface brightness I_s
    [photons / s / m^2 / sr], the signal collected in one image pixel is:

        S_e [e-] = I_s * Omega_pix * A_tel * t * QE_eff

    where:
        Omega_pix = pixel_solid_angle = (plate_scale_rad)^2 = (p_phys/f)^2 [sr]
        A_tel     = pi * (D/2)^2  [m^2]
        t         = exposure time [s]
        QE_eff    = effective quantum efficiency (QE x Bayer filter)

    Collecting area and solid angle can be combined:
        S_e ∝ (p_phys/f)^2 * (D^2/4) * t * QE_eff
            = (p_phys * D / f)^2 * t * QE_eff / 4
            = (p_phys / N)^2 * t * QE_eff / 4          (N = f/D)

    For fixed physical pixel size p_phys, the signal scales as:
        S_e ∝ t * QE_eff / N^2

    The per-channel gain relative to a reference configuration is:
        gain_c = (t / t_ref) * (N_ref / N)^2 * (QE_eff_c / QE_eff_ref_c) * eta

    where eta = system throughput (additional optics/atmosphere factor).

    Note on aperture independence
    ------------------------------
    For a fixed f-ratio (N constant), changing aperture D at the same f/N
    changes focal length f proportionally, which changes the plate scale
    and hence Omega_pix.  In this simulator the image resolution is fixed,
    so a change in plate scale is encoded in the user's FOV input, not
    automatically derived.  If comparing two telescopes at the same f/ratio
    and same physical pixel size, gain = 1 for extended sources.  Aperture
    advantage for extended sources only manifests as more pixels across the
    target (wider field or finer plate scale), not in per-pixel brightness.

    Parameters
    ----------
    image : np.ndarray (H, W, 3), linear float64

    Returns
    -------
    (scaled_image, diagnostics_dict)
    """
    N_target = focal_length_mm / aperture_mm
    N_ref    = ref_focal_length_mm / ref_aperture_mm

    out   = np.empty_like(image)
    gains = {}

    for idx, ch in enumerate(("R", "G", "B")):
        qe_c     = _channel_effective_qe(ch, qe_fn)
        qe_ref_c = _channel_effective_qe(ch, ref_qe_fn)
        qe_ratio = (qe_c / qe_ref_c) if qe_ref_c > 0.0 else 1.0

        gain = (
            throughput
            * (exposure_time_s / ref_exposure_time_s)
            * (N_ref / N_target) ** 2
            * qe_ratio
        )
        out[..., idx] = image[..., idx] * gain
        gains[ch] = {
            "gain":          round(gain, 6),
            "qe_eff_target": round(qe_c, 4),
            "qe_eff_ref":    round(qe_ref_c, 4),
            "qe_ratio":      round(qe_ratio, 4),
        }

    diag = {
        "f_number_target": N_target,
        "f_number_ref":    N_ref,
        "per_channel":     gains,
    }
    return out, diag


# =============================================================================
#  FILTER 3: NOISE
# =============================================================================

def apply_noise_filter(
    image: np.ndarray,
    full_well_e: float,
    read_noise_e: float,
    dark_current_e_per_s: float,
    exposure_time_s: float,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Add physically-motivated Poisson photon shot noise and Gaussian read noise.

    Noise model (Hamamatsu / standard CCD/CMOS model)
    ---------------------------------------------------
    The image pixel value v in [0, 1] is treated as a normalised electron
    count: signal_e = v * full_well_e.

    Total noise:
        Dark contribution : dark_e = dark_current * t          [e-]
        Shot noise        : N_shot = sqrt(signal_e + dark_e)   [e- RMS]
        Read noise        : N_read = read_noise_e              [e- RMS]
        Combined noise    : N_total = sqrt(N_shot^2 + N_read^2)

    Gaussian approximation to Poisson noise is valid when signal_e >> 1,
    which holds for any pixel above background in a real exposure.

    Parameters
    ----------
    image : np.ndarray (H, W, 3), float64
    full_well_e : float
        Full-well electron capacity; image pixel = 1.0 maps to this count.
    read_noise_e : float
        Read noise in electrons RMS per pixel.
    dark_current_e_per_s : float
        Dark current at operating sensor temperature [e-/px/s].
    exposure_time_s : float
        Integration time [s].
    rng : np.random.Generator, optional
        Seeded RNG for reproducibility.  Defaults to np.random.default_rng().

    Returns
    -------
    np.ndarray (H, W, 3), float64
        Noisy image, same units as input.
    """
    if rng is None:
        rng = np.random.default_rng()

    signal_e = np.clip(image, 0.0, None) * full_well_e
    dark_e   = dark_current_e_per_s * exposure_time_s

    # Variance = signal + dark (Poisson);  read noise is Gaussian
    shot_variance = np.maximum(signal_e + dark_e, 0.0)
    noise_e = (
        rng.normal(0.0, 1.0, signal_e.shape) * np.sqrt(shot_variance)
        + rng.normal(0.0, read_noise_e, signal_e.shape)
    )

    noisy_e = signal_e + dark_e + noise_e
    return noisy_e / full_well_e     # normalise back to [0, 1] reference scale


# =============================================================================
#  DISPLAY STRETCH
# =============================================================================

def stretch_for_display(
    image: np.ndarray,
    mode: str,
    gamma: float,
    asinh_beta: float = 0.5,
    log_a: float = 1000.0,
) -> np.ndarray:
    """
    Apply a non-linear display stretch to the linear signal image.

    Modes
    -----
    "none" / "clip"
        Linear, clip to [0, 1].
    "normalize"
        Divide by global peak value, then clip.
    "asinh"
        y = arcsinh(beta * x) / arcsinh(beta)
        Commonly used in astronomical image display (e.g. DS9, PixInsight).
        beta controls the softening: larger -> more aggressive stretch.
    "log"
        y = log(1 + A*x) / log(1 + A)
        A controls the contrast expansion of faint detail.

    An optional gamma correction (power-law) is applied after the stretch:
        y -> y^(1/gamma)
    """
    x = np.clip(image, 0.0, None)

    if mode in ("none", "clip"):
        y = np.clip(x, 0.0, 1.0)

    elif mode == "normalize":
        peak = float(x.max())
        y    = x / peak if peak > 0.0 else x.copy()

    elif mode == "asinh":
        b = max(float(asinh_beta), 1e-6)
        y = np.arcsinh(b * x) / np.arcsinh(b)

    elif mode == "log":
        a = max(float(log_a), 1.0)
        y = np.log1p(a * x) / math.log1p(a)

    else:
        raise ValueError(
            f"Unknown DISPLAY_STRETCH_MODE '{mode}'.  "
            "Valid options: 'none', 'clip', 'normalize', 'asinh', 'log'."
        )

    y = np.clip(y, 0.0, 1.0)
    if gamma != 1.0:
        y = np.power(y, 1.0 / gamma)
    return np.clip(y, 0.0, 1.0)


# =============================================================================
#  MAIN SIMULATION FUNCTION
# =============================================================================

def simulate(
    input_path: Path,
    output_path: Path,
    aperture_mm: float,
    focal_length_mm: float,
    fov_x_arcsec: float,
    fov_y_arcsec: float | None,
    central_obscuration: float,
    seeing_fwhm_arcsec: float,
    qe_model,
    exposure_time_s: float,
    system_throughput: float,
    ref_aperture_mm: float,
    ref_focal_length_mm: float,
    ref_exposure_time_s: float,
    ref_qe_model,
    full_well_e: float,
    read_noise_e: float,
    dark_current_e_per_s: float,
    enable_diffraction: bool,
    enable_exposure: bool,
    enable_noise: bool,
    display_mode: str,
    asinh_beta: float,
    log_a: float,
    gamma: float,
    pixel_size_um: float,
    print_diagnostics: bool,
) -> None:
    """
    Execute the full simulation pipeline and save the output image.
    """
    # --- Load image ---
    image, alpha = load_image(input_path)
    H, W = image.shape[:2]

    # --- Compute geometry ---
    if fov_y_arcsec is None:
        fov_y_arcsec = fov_x_arcsec * (H / W)

    pixel_scale_x_rad = arcsec_to_rad(fov_x_arcsec / W)
    pixel_scale_y_rad = arcsec_to_rad(fov_y_arcsec / H)
    # Geometric mean -> isotropic kernel for the PSF
    pixel_scale_rad   = math.sqrt(pixel_scale_x_rad * pixel_scale_y_rad)

    D_m = aperture_mm * 1e-3
    N   = focal_length_mm / aperture_mm

    # --- Build QE functions ---
    qe_fn     = build_qe_fn(qe_model)
    ref_qe_fn = build_qe_fn(ref_qe_model)

    # --- Pre-flight geometry check ---
    # This must run before the pipeline so we can cap the kernel size.
    preflight = psf_preflight_check(
        pixel_scale_rad    = pixel_scale_rad,
        aperture_m         = D_m,
        seeing_fwhm_arcsec = seeing_fwhm_arcsec,
        image_w_px         = W,
        image_h_px         = H,
        pixel_size_um      = pixel_size_um,
        focal_length_mm    = focal_length_mm,
    )
    # Print any warnings immediately, before spending time on computation
    if preflight["warnings"]:
        print()
        print("!" * 80)
        print("  PRE-FLIGHT GEOMETRY WARNINGS")
        print("!" * 80)
        for i, w in enumerate(preflight["warnings"], 1):
            # Word-wrap at 76 chars for readability
            words   = w.split()
            line    = f"  [{i}] "
            indent  = " " * 6
            for word in words:
                if len(line) + len(word) + 1 > 78:
                    print(line)
                    line = indent + word + " "
                else:
                    line += word + " "
            print(line)
        print()
        print("  RECOMMENDED CORRECT FOV for this telescope + sensor:")
        cps = preflight["physically_correct_ps_arcsec"]
        cfov = preflight["physically_correct_fov_arcsec"]
        print(f"    FOV_X_ARCSEC = {cfov:.2f}   "
              f"(plate scale {cps:.4f}\"/px,  "
              f"Airy FWHM = {preflight['fwhm_px']:.2f} px at correct scale)")
        print()
        print("  FOR A DETAILED AIRY-DISK VIEW (showing several rings):")
        # Recommend a FOV that makes the Airy zero span ~50 pixels
        target_airy_px = 50
        rec_ps  = preflight["airy_zero_arcsec"] / target_airy_px
        rec_fov = rec_ps * W
        print(f"    FOV_X_ARCSEC = {rec_fov:.2f}   "
              f"(plate scale {rec_ps:.5f}\"/px,  "
              f"Airy 1st-zero = {target_airy_px} px)")
        print("!" * 80)
        print()

    # Compute the kernel size cap: at most half the image in each direction.
    # This prevents memory exhaustion when the PSF is larger than the image.
    max_kernel_half_px = min(W, H) // 2 - 1

    # --- Run the pipeline ---
    result      = image.copy()
    diff_diags  = []
    exp_diag    = {}

    # ---- FILTER 1: DIFFRACTION ----
    if enable_diffraction:
        result, diff_diags = apply_diffraction_filter(
            image               = result,
            pixel_scale_rad     = pixel_scale_rad,
            aperture_m          = D_m,
            obscuration         = central_obscuration,
            seeing_fwhm_arcsec  = seeing_fwhm_arcsec,
            qe_fn               = qe_fn,
            max_kernel_half_px  = max_kernel_half_px,
        )
    # If disabled: identity (no PSF convolution)

    # ---- FILTER 2: EXPOSURE ----
    if enable_exposure:
        result, exp_diag = apply_exposure_filter(
            image               = result,
            aperture_mm         = aperture_mm,
            focal_length_mm     = focal_length_mm,
            exposure_time_s     = exposure_time_s,
            throughput          = system_throughput,
            qe_fn               = qe_fn,
            ref_aperture_mm     = ref_aperture_mm,
            ref_focal_length_mm = ref_focal_length_mm,
            ref_exposure_time_s = ref_exposure_time_s,
            ref_qe_fn           = ref_qe_fn,
        )
    # If disabled: identity (unity gain)

    # ---- FILTER 3: NOISE ----
    if enable_noise:
        result = apply_noise_filter(
            image                = result,
            full_well_e          = full_well_e,
            read_noise_e         = read_noise_e,
            dark_current_e_per_s = dark_current_e_per_s,
            exposure_time_s      = exposure_time_s,
        )
    # If disabled: no noise added

    # ---- Display stretch ----
    display = stretch_for_display(
        image      = result,
        mode       = display_mode,
        gamma      = gamma,
        asinh_beta = asinh_beta,
        log_a      = log_a,
    )

    # --- Save ---
    save_image(output_path, display, alpha=alpha)

    # --- Diagnostics ---
    if not print_diagnostics:
        return

    sep = "=" * 80
    print()
    print(sep)
    print(" TELESCOPE IMAGING SIMULATOR v2.0 -- DIAGNOSTICS")
    print(sep)
    print(f"  Input         : {input_path}")
    print(f"  Output        : {output_path}")
    print(f"  Image size    : {W} x {H} px")
    print()
    print("  GEOMETRY")
    print(f"    FOV               : {fov_x_arcsec:.4g} x {fov_y_arcsec:.4g} arcsec")
    print(f"    Plate scale       : {rad_to_arcsec(pixel_scale_x_rad):.4f} x "
          f"{rad_to_arcsec(pixel_scale_y_rad):.4f} arcsec/px")
    implied_px_x = D_m * 1e3 * pixel_scale_x_rad * 1e3  # mm -> µm
    implied_px_y = D_m * 1e3 * pixel_scale_y_rad * 1e3
    print(f"    Implied px pitch  : {implied_px_x:.2f} x {implied_px_y:.2f} µm "
          f"(from FOV / f_mm; f = {focal_length_mm:.1f} mm)")
    print(f"    Physical px size  : {pixel_size_um:.2f} µm (user-specified)")
    print()
    print("  TELESCOPE")
    print(f"    Aperture          : {aperture_mm:.4g} mm")
    print(f"    Focal length      : {focal_length_mm:.4g} mm")
    print(f"    Focal ratio       : f/{N:.3f}")
    print(f"    Central obs.      : {central_obscuration:.4g}")
    print(f"    Seeing FWHM       : {seeing_fwhm_arcsec:.4g} arcsec "
          f"({'enabled' if seeing_fwhm_arcsec > 0 else 'disabled'})")
    print()
    print("  SENSOR / EXPOSURE")
    print(f"    QE model          : {qe_model}")
    print(f"    Exposure time     : {exposure_time_s:.4g} s")
    print(f"    System throughput : {system_throughput:.4g}")
    print(f"    Full-well depth   : {full_well_e:.0f} e-")
    print(f"    Read noise        : {read_noise_e:.2f} e- RMS")
    print(f"    Dark current      : {dark_current_e_per_s:.4g} e-/px/s")
    print()
    print("  FILTER LAYERS")
    print(f"    DIFFRACTION  : {'ON' if enable_diffraction else 'OFF (identity)'}")
    print(f"    EXPOSURE     : {'ON' if enable_exposure else 'OFF (unity gain)'}")
    print(f"    NOISE        : {'ON' if enable_noise else 'OFF'}")
    print()

    if diff_diags:
        print("  DIFFRACTION FILTER -- per-channel PSF summary")
        print(f"    {'Ch':>2}  {'Kernel':>10}  {'Clamped?':>9}  {'1st-zero':>10}  "
              f"{'FWHM range':>22}  {'mean QE x Bayer':>16}")
        for d in diff_diags:
            lo, hi    = d["diffraction_fwhm_min"], d["diffraction_fwhm_max"]
            clamp_str = "YES  <<" if d["was_clamped"] else "no"
            print(
                f"    {d['channel']:>2}  "
                f"{d['kernel_total_px']:>8} px  "
                f"{clamp_str:>9}  "
                f"{d['first_zero_arcsec']:>8.4f}\"  "
                f"  {lo:.4f}\" -- {hi:.4f}\"  "
                f"  {d['mean_qe_x_bayer']:>12.4f}"
            )
        print()
        print("  PSF GEOMETRY (green channel, representative)")
        pf = preflight
        print(f"    Airy FWHM           : {pf['fwhm_arcsec']:.4f}\"  = {pf['fwhm_px']:.2f} px")
        print(f"    Airy 1st zero       : {pf['airy_zero_arcsec']:.4f}\"  = {pf['airy_zero_px']:.2f} px")
        print(f"    Image half-width    : {rad_to_arcsec(pixel_scale_rad)*W/2:.4f}\"  = {W//2} px")
        print(f"    Image spans         : {pf['image_airy_radii']:.2f} Airy radii")
        if pf['was_clamped']:
            print(f"    Kernel capped to    : {2*pf['clamped_half_px']+1} px  "
                  f"(covers {pf['clamped_airy_radii']:.2f} Airy radii, "
                  f"~{pf['captured_pct']:.0f}% of PSF energy)")
            print(f"    *** PSF extends BEYOND image boundaries ***")
            print(f"    *** Remaining ~{100-pf['captured_pct']:.0f}% of source flux falls off-sensor ***")

    if exp_diag:
        print("  EXPOSURE FILTER -- per-channel gain summary")
        print(f"    Target f/{exp_diag['f_number_target']:.3f}  |  "
              f"Reference f/{exp_diag['f_number_ref']:.3f}")
        print(f"    {'Ch':>2}  {'gain':>10}  {'QE_eff':>8}  {'QE_ref':>8}  {'ratio':>8}")
        for ch, g in exp_diag["per_channel"].items():
            print(
                f"    {ch:>2}  "
                f"{g['gain']:>10.5f}  "
                f"{g['qe_eff_target']:>8.4f}  "
                f"{g['qe_eff_ref']:>8.4f}  "
                f"{g['qe_ratio']:>8.4f}"
            )
        print()

    print(f"  DISPLAY: stretch={display_mode}  gamma={gamma:.3f}")
    print(sep)


# =============================================================================
#  ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    _base = Path(__file__).resolve().parent

    simulate(
        input_path             = resolve_path(INPUT_IMAGE,  _base),
        output_path            = resolve_path(OUTPUT_IMAGE, _base),
        aperture_mm            = APERTURE_MM,
        focal_length_mm        = FOCAL_LENGTH_MM,
        fov_x_arcsec           = FOV_X_ARCSEC,
        fov_y_arcsec           = FOV_Y_ARCSEC,
        central_obscuration    = CENTRAL_OBSCURATION,
        seeing_fwhm_arcsec     = SEEING_FWHM_ARCSEC,
        qe_model               = QE_MODEL,
        exposure_time_s        = EXPOSURE_TIME_S,
        system_throughput      = SYSTEM_THROUGHPUT,
        ref_aperture_mm        = REFERENCE_APERTURE_MM,
        ref_focal_length_mm    = REFERENCE_FOCAL_LENGTH_MM,
        ref_exposure_time_s    = REFERENCE_EXPOSURE_TIME_S,
        ref_qe_model           = REFERENCE_QE_MODEL,
        full_well_e            = FULL_WELL_ELECTRONS,
        read_noise_e           = READ_NOISE_ELECTRONS,
        dark_current_e_per_s   = DARK_CURRENT_E_PER_S,
        enable_diffraction     = ENABLE_DIFFRACTION,
        enable_exposure        = ENABLE_EXPOSURE,
        enable_noise           = ENABLE_NOISE,
        display_mode           = DISPLAY_STRETCH_MODE,
        asinh_beta             = DISPLAY_ASINH_BETA,
        log_a                  = DISPLAY_LOG_A,
        gamma                  = GAMMA,
        pixel_size_um          = PIXEL_SIZE_UM,
        print_diagnostics      = PRINT_DIAGNOSTICS,
    )
