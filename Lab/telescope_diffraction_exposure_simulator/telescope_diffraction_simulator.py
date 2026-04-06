#!/usr/bin/env python3
import math
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.signal import fftconvolve
from scipy.special import j1


INPUT_IMAGE = "JUPITER.png"
OUTPUT_IMAGE = "JUPITER_DIFF_6.png"

APERTURE_MM = 235 #71 91 103 150 203.2 235
FOCAL_LENGTH_MM = 2350 #490 609 700 1800 2032 2350

FOV_X_DEG = 0.01303
FOV_Y_DEG = None

CENTRAL_OBSCURATION = 0.0
SEEING_FWHM_ARCSEC = 0.0

PRINT_DIAGNOSTICS = True


def load_image(path: Path):
    img = Image.open(path)
    has_alpha = img.mode in ("RGBA", "LA")
    if img.mode not in ("L", "RGB", "RGBA"):
        img = img.convert("RGBA" if has_alpha else "RGB")
    arr = np.asarray(img).astype(np.float64) / 255.0

    alpha = None
    if arr.ndim == 3 and arr.shape[2] == 4:
        alpha = arr[..., 3:4].copy()
        arr = arr[..., :3]

    return arr, alpha


def save_image(path: Path, arr: np.ndarray, alpha=None):
    arr = np.clip(arr, 0.0, 1.0)
    if arr.ndim == 2:
        out = (arr * 255.0 + 0.5).astype(np.uint8)
        img = Image.fromarray(out, mode="L")
    else:
        if alpha is not None:
            alpha = np.clip(alpha, 0.0, 1.0)
            rgba = np.concatenate([arr, alpha], axis=2)
            out = (rgba * 255.0 + 0.5).astype(np.uint8)
            img = Image.fromarray(out, mode="RGBA")
        else:
            out = (arr * 255.0 + 0.5).astype(np.uint8)
            img = Image.fromarray(out, mode="RGB")
    img.save(path)


def arcsec_from_rad(x_rad: float) -> float:
    return x_rad * 180.0 / math.pi * 3600.0


def rad_from_arcsec(x_arcsec: float) -> float:
    return x_arcsec / 3600.0 * math.pi / 180.0


def annular_airy_psf_kernel(
    pixel_scale_x_rad: float,
    pixel_scale_y_rad: float,
    aperture_diameter_m: float,
    wavelength_m: float,
    central_obscuration_ratio: float = 0.0,
    half_size_px: int = 32,
) -> np.ndarray:
    if not (0.0 <= central_obscuration_ratio < 1.0):
        raise ValueError("CENTRAL_OBSCURATION must be in [0, 1).")

    y, x = np.mgrid[-half_size_px:half_size_px + 1, -half_size_px:half_size_px + 1]
    theta_rad = np.sqrt((x * pixel_scale_x_rad) ** 2 + (y * pixel_scale_y_rad) ** 2)
    rho = np.pi * aperture_diameter_m * theta_rad / wavelength_m
    eps = float(central_obscuration_ratio)

    psf = np.empty_like(theta_rad, dtype=np.float64)
    nz = rho != 0.0

    if eps == 0.0:
        psf[~nz] = 1.0
        airy = 2.0 * j1(rho[nz]) / rho[nz]
        psf[nz] = airy ** 2
    else:
        psf[~nz] = 1.0
        ann = (2.0 * j1(rho[nz]) - 2.0 * eps * j1(eps * rho[nz])) / (
            rho[nz] * (1.0 - eps ** 2)
        )
        psf[nz] = ann ** 2

    psf = np.clip(psf, 0.0, None)
    psf /= psf.sum()
    return psf


def gaussian_seeing_kernel(
    pixel_scale_x_rad: float,
    pixel_scale_y_rad: float,
    seeing_fwhm_arcsec: float,
    half_size_px: int,
) -> np.ndarray:
    if seeing_fwhm_arcsec <= 0.0:
        g = np.zeros((2 * half_size_px + 1, 2 * half_size_px + 1), dtype=np.float64)
        g[half_size_px, half_size_px] = 1.0
        return g

    sigma_rad = rad_from_arcsec(seeing_fwhm_arcsec) / (2.0 * math.sqrt(2.0 * math.log(2.0)))
    y, x = np.mgrid[-half_size_px:half_size_px + 1, -half_size_px:half_size_px + 1]
    rr2 = (x * pixel_scale_x_rad) ** 2 + (y * pixel_scale_y_rad) ** 2
    g = np.exp(-0.5 * rr2 / (sigma_rad ** 2))
    g /= g.sum()
    return g


def build_channel_kernel(
    image_width_px: int,
    image_height_px: int,
    fov_x_deg: float,
    fov_y_deg: float,
    aperture_diameter_m: float,
    focal_length_m: float,
    wavelength_m: float,
    central_obscuration_ratio: float = 0.0,
    seeing_fwhm_arcsec: float = 0.0,
):
    pixel_scale_x_rad = math.radians(fov_x_deg) / image_width_px
    pixel_scale_y_rad = math.radians(fov_y_deg) / image_height_px

    airy_first_zero_rad = 1.22 * wavelength_m / aperture_diameter_m
    airy_first_zero_px = airy_first_zero_rad / min(pixel_scale_x_rad, pixel_scale_y_rad)

    implied_pixel_pitch_x_m = focal_length_m * pixel_scale_x_rad
    implied_pixel_pitch_y_m = focal_length_m * pixel_scale_y_rad
    airy_first_zero_focal_m = 1.22 * wavelength_m * focal_length_m / aperture_diameter_m

    seeing_sigma_px = 0.0
    if seeing_fwhm_arcsec > 0.0:
        seeing_sigma_rad = rad_from_arcsec(seeing_fwhm_arcsec) / (2.0 * math.sqrt(2.0 * math.log(2.0)))
        seeing_sigma_px = seeing_sigma_rad / min(pixel_scale_x_rad, pixel_scale_y_rad)

    half_size_px = int(math.ceil(max(12.0 * airy_first_zero_px, 6.0 * seeing_sigma_px, 4.0)))

    diffraction_psf = annular_airy_psf_kernel(
        pixel_scale_x_rad=pixel_scale_x_rad,
        pixel_scale_y_rad=pixel_scale_y_rad,
        aperture_diameter_m=aperture_diameter_m,
        wavelength_m=wavelength_m,
        central_obscuration_ratio=central_obscuration_ratio,
        half_size_px=half_size_px,
    )

    if seeing_fwhm_arcsec > 0.0:
        seeing_psf = gaussian_seeing_kernel(
            pixel_scale_x_rad=pixel_scale_x_rad,
            pixel_scale_y_rad=pixel_scale_y_rad,
            seeing_fwhm_arcsec=seeing_fwhm_arcsec,
            half_size_px=half_size_px,
        )
        kernel = fftconvolve(diffraction_psf, seeing_psf, mode="same")
        kernel = np.clip(kernel, 0.0, None)
        kernel /= kernel.sum()
    else:
        kernel = diffraction_psf

    diagnostics = {
        "pixel_scale_x_arcsec_per_px": arcsec_from_rad(pixel_scale_x_rad),
        "pixel_scale_y_arcsec_per_px": arcsec_from_rad(pixel_scale_y_rad),
        "implied_pixel_pitch_x_um": implied_pixel_pitch_x_m * 1e6,
        "implied_pixel_pitch_y_um": implied_pixel_pitch_y_m * 1e6,
        "airy_first_zero_arcsec": arcsec_from_rad(airy_first_zero_rad),
        "airy_first_zero_px": airy_first_zero_px,
        "airy_first_zero_focal_um": airy_first_zero_focal_m * 1e6,
        "kernel_size_px": int(kernel.shape[0]),
    }
    return kernel, diagnostics


def convolve_image_with_psf(image: np.ndarray, kernels):
    if image.ndim == 2:
        return np.clip(fftconvolve(image, kernels[1], mode="same"), 0.0, 1.0)

    if image.ndim != 3 or image.shape[2] not in (1, 3):
        raise ValueError("Input image must be grayscale or RGB.")

    out = np.empty_like(image, dtype=np.float64)

    if image.shape[2] == 1:
        out[..., 0] = fftconvolve(image[..., 0], kernels[1], mode="same")
    else:
        out[..., 0] = fftconvolve(image[..., 0], kernels[0], mode="same")
        out[..., 1] = fftconvolve(image[..., 1], kernels[1], mode="same")
        out[..., 2] = fftconvolve(image[..., 2], kernels[2], mode="same")

    return np.clip(out, 0.0, 1.0)


def resolve_path(path_str: str, base_dir: Path) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else (base_dir / p).resolve()


def simulate_telescope_diffraction(
    input_path: Path,
    output_path: Path,
    aperture_diameter_mm: float,
    focal_length_mm: float,
    fov_x_deg: float,
    fov_y_deg: float | None = None,
    central_obscuration_ratio: float = 0.0,
    seeing_fwhm_arcsec: float = 0.0,
    print_diagnostics: bool = True,
):
    image, alpha = load_image(input_path)
    h, w = image.shape[:2]

    if fov_y_deg is None:
        fov_y_deg = fov_x_deg * (h / w)

    D_m = aperture_diameter_mm * 1e-3
    f_m = focal_length_mm * 1e-3

    wavelengths_m = [650e-9, 550e-9, 450e-9]
    kernels = []
    diagnostics = []

    for wl in wavelengths_m:
        k, d = build_channel_kernel(
            image_width_px=w,
            image_height_px=h,
            fov_x_deg=fov_x_deg,
            fov_y_deg=fov_y_deg,
            aperture_diameter_m=D_m,
            focal_length_m=f_m,
            wavelength_m=wl,
            central_obscuration_ratio=central_obscuration_ratio,
            seeing_fwhm_arcsec=seeing_fwhm_arcsec,
        )
        kernels.append(k)
        diagnostics.append(d)

    result = convolve_image_with_psf(image, kernels)
    save_image(output_path, result, alpha=alpha)

    if print_diagnostics:
        channel_names = ["R (650 nm)", "G (550 nm)", "B (450 nm)"]
        print("\nSimulation diagnostics")
        print("-" * 80)
        print(f"Script directory:          {Path(__file__).resolve().parent}")
        print(f"Input image:              {input_path}")
        print(f"Output image:             {output_path}")
        print(f"Image size:               {w} x {h} px")
        print(f"Aperture diameter:        {aperture_diameter_mm:.6g} mm")
        print(f"Focal length:             {focal_length_mm:.6g} mm")
        print(f"Focal ratio:              f/{focal_length_mm / aperture_diameter_mm:.3f}")
        print(f"Horizontal field:         {fov_x_deg:.6g} deg")
        print(f"Vertical field:           {fov_y_deg:.6g} deg")
        print(f"Central obscuration:      {central_obscuration_ratio:.6g}")
        print(f"Seeing FWHM (optional):   {seeing_fwhm_arcsec:.6g} arcsec")
        print("-" * 80)

        for name, d in zip(channel_names, diagnostics):
            print(name)
            print(f"  Pixel scale X:          {d['pixel_scale_x_arcsec_per_px']:.6g} arcsec/px")
            print(f"  Pixel scale Y:          {d['pixel_scale_y_arcsec_per_px']:.6g} arcsec/px")
            print(f"  Implied pixel pitch X:  {d['implied_pixel_pitch_x_um']:.6g} um")
            print(f"  Implied pixel pitch Y:  {d['implied_pixel_pitch_y_um']:.6g} um")
            print(f"  Airy first zero:        {d['airy_first_zero_arcsec']:.6g} arcsec")
            print(f"  Airy first zero:        {d['airy_first_zero_px']:.6g} px")
            print(f"  Airy first zero @ fp:   {d['airy_first_zero_focal_um']:.6g} um")
            print(f"  Kernel size:            {d['kernel_size_px']} px")
            print("-" * 80)


if __name__ == "__main__":
    script_dir = Path(__file__).resolve().parent
    input_path = resolve_path(INPUT_IMAGE, script_dir)
    output_path = resolve_path(OUTPUT_IMAGE, script_dir)

    simulate_telescope_diffraction(
        input_path=input_path,
        output_path=output_path,
        aperture_diameter_mm=APERTURE_MM,
        focal_length_mm=FOCAL_LENGTH_MM,
        fov_x_deg=FOV_X_DEG,
        fov_y_deg=FOV_Y_DEG,
        central_obscuration_ratio=CENTRAL_OBSCURATION,
        seeing_fwhm_arcsec=SEEING_FWHM_ARCSEC,
        print_diagnostics=PRINT_DIAGNOSTICS,
    )
