import numpy as np
from scipy.signal import fftconvolve
from PIL import Image


def load_image(path: str) -> np.ndarray:
    """
    Load an image as float32 in [0, 1].
    Returns:
        - grayscale image with shape (H, W), or
        - color image with shape (H, W, C)
    """
    img = Image.open(path)
    arr = np.asarray(img).astype(np.float32)

    if arr.dtype != np.float32:
        arr = arr.astype(np.float32)

    # Normalize common integer image formats to [0, 1]
    if arr.max() > 1.0:
        arr /= 255.0

    return arr


def save_image(path: str, arr: np.ndarray) -> None:
    """
    Save float image in [0, 1] to disk.
    """
    out = np.clip(arr, 0.0, 1.0)
    out = (out * 255.0).astype(np.uint8)
    Image.fromarray(out).save(path)


def arcsec_per_pixel_from_min_dimension(image_shape, min_dim_arcsec: float) -> float:
    """
    Compute plate scale from the angular coverage of the shorter image dimension.

    Parameters
    ----------
    image_shape : tuple
        (H, W) or (H, W, C)
    min_dim_arcsec : float
        Total angular coverage of the shorter image dimension in arcseconds.

    Returns
    -------
    float
        Arcseconds per pixel.
    """
    h, w = image_shape[:2]
    n_min = min(h, w)
    if min_dim_arcsec <= 0:
        raise ValueError("min_dim_arcsec must be positive.")
    return min_dim_arcsec / float(n_min)


def moffat_kernel_from_fwhm(
    fwhm_px: float,
    beta: float = 4.765,
    kernel_radius_factor: float = 4.0
) -> np.ndarray:
    """
    Create a normalized circular Moffat kernel from a target FWHM in pixels.

    Literature note:
    beta ≈ 4.765 is a good fit to the atmospheric turbulence PSF.
    """
    if fwhm_px <= 0:
        raise ValueError("fwhm_px must be positive.")
    if beta <= 1.0:
        raise ValueError("beta should usually be > 1 for a well-behaved PSF.")

    alpha = fwhm_px / (2.0 * np.sqrt(2.0 ** (1.0 / beta) - 1.0))

    radius = max(2, int(np.ceil(kernel_radius_factor * fwhm_px)))
    y, x = np.mgrid[-radius:radius + 1, -radius:radius + 1]
    r2 = x * x + y * y

    kernel = (1.0 + r2 / (alpha * alpha)) ** (-beta)
    kernel /= kernel.sum()
    return kernel.astype(np.float32)


def gaussian_kernel_from_fwhm(
    fwhm_px: float,
    kernel_radius_factor: float = 4.0
) -> np.ndarray:
    """
    Create a normalized Gaussian kernel from a target FWHM in pixels.
    """
    if fwhm_px <= 0:
        raise ValueError("fwhm_px must be positive.")

    sigma = fwhm_px / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    radius = max(2, int(np.ceil(kernel_radius_factor * fwhm_px)))
    y, x = np.mgrid[-radius:radius + 1, -radius:radius + 1]
    r2 = x * x + y * y

    kernel = np.exp(-0.5 * r2 / (sigma * sigma))
    kernel /= kernel.sum()
    return kernel.astype(np.float32)


def convolve_image_with_psf(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Convolve grayscale or color image with a PSF kernel using FFT convolution.
    """
    if image.ndim == 2:
        return fftconvolve(image, kernel, mode="same").astype(np.float32)

    if image.ndim == 3:
        channels = []
        for c in range(image.shape[2]):
            conv_c = fftconvolve(image[:, :, c], kernel, mode="same")
            channels.append(conv_c.astype(np.float32))
        return np.stack(channels, axis=2)

    raise ValueError("Image must be 2D or 3D.")


def apply_atmospheric_seeing(
    image: np.ndarray,
    min_dim_arcsec: float,
    seeing_fwhm_arcsec: float,
    model: str = "moffat",
    beta: float = 4.765
):
    """
    Apply long-exposure atmospheric seeing blur to an image.

    Parameters
    ----------
    image : np.ndarray
        Input image, grayscale or color, normalized to [0, 1].
    min_dim_arcsec : float
        Total angular coverage of the shorter image dimension, in arcseconds.
    seeing_fwhm_arcsec : float
        Target atmospheric seeing FWHM, in arcseconds.
    model : str
        "moffat" or "gaussian".
    beta : float
        Moffat beta parameter if model == "moffat".

    Returns
    -------
    blurred : np.ndarray
        Seeing-degraded image.
    info : dict
        Useful diagnostic quantities.
    """
    if seeing_fwhm_arcsec <= 0:
        raise ValueError("seeing_fwhm_arcsec must be positive.")

    scale_arcsec_per_px = arcsec_per_pixel_from_min_dimension(image.shape, min_dim_arcsec)
    fwhm_px = seeing_fwhm_arcsec / scale_arcsec_per_px

    if model.lower() == "moffat":
        kernel = moffat_kernel_from_fwhm(fwhm_px=fwhm_px, beta=beta)
    elif model.lower() == "gaussian":
        kernel = gaussian_kernel_from_fwhm(fwhm_px=fwhm_px)
    else:
        raise ValueError("model must be 'moffat' or 'gaussian'.")

    blurred = convolve_image_with_psf(image, kernel)

    info = {
        "arcsec_per_pixel": scale_arcsec_per_px,
        "seeing_fwhm_arcsec": seeing_fwhm_arcsec,
        "seeing_fwhm_px": fwhm_px,
        "model": model.lower(),
        "beta": beta if model.lower() == "moffat" else None,
        "kernel_shape": kernel.shape,
    }

    return blurred, info


if __name__ == "__main__":
    # Example
    input_path = "AIRY_DIFFRACTION.png"
    output_path = "AIRY_DIFFRACTION_seeing_blurred.png"

    # Example meaning:
    # shorter image dimension spans 1200 arcsec total
    # seeing FWHM is 2.0 arcsec
    min_dim_arcsec = 8
    seeing_fwhm_arcsec = 0.5

    img = load_image(input_path)

    blurred, info = apply_atmospheric_seeing(
        image=img,
        min_dim_arcsec=min_dim_arcsec,
        seeing_fwhm_arcsec=seeing_fwhm_arcsec,
        model="moffat",   # recommended default
        beta=4.765
    )

    print("Simulation info:")
    for k, v in info.items():
        print(f"{k}: {v}")

    save_image(output_path, blurred)
    print(f"Saved: {output_path}")