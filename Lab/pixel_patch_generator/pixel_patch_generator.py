from PIL import Image


def load_image(path: str) -> Image.Image:
    """
    Load an image and convert it to RGB.
    """
    return Image.open(path).convert("RGB")


def save_image(image: Image.Image, path: str) -> None:
    """
    Save an image and print its size for verification.
    """
    image.save(path)
    print(f"Saved: {path} | size={image.size}")


def create_low_resolution_image(
    image: Image.Image,
    lowres_width: int,
    lowres_height: int
) -> Image.Image:
    """
    Create the true low-resolution image.
    """
    if lowres_width <= 0 or lowres_height <= 0:
        raise ValueError("lowres_width and lowres_height must be positive.")

    return image.resize((lowres_width, lowres_height), Image.Resampling.LANCZOS)


def create_same_size_pixelated_image(
    image: Image.Image,
    lowres_width: int,
    lowres_height: int
) -> Image.Image:
    """
    Create a same-size pixelated image by:
    1. shrinking to low resolution
    2. enlarging back to the original image size with nearest-neighbour
    """
    original_size = image.size
    lowres_image = create_low_resolution_image(image, lowres_width, lowres_height)
    same_size_pixelated = lowres_image.resize(original_size, Image.Resampling.NEAREST)
    return same_size_pixelated


def main():
    # ------------------------------------------------------------
    # USER SETTINGS
    # ------------------------------------------------------------
    input_path = "AIRY_DIFFRACTION.png"

    # Output 1: true low-resolution image
    output_lowres_path = "AIRY_DIFFRACTION_lowres_6x6.png"

    # Output 2: same-size image, but visually pixelated
    output_same_size_path = "AIRY_DIFFRACTION_6by6.png"

    # Target low-resolution grid
    lowres_width = 6
    lowres_height = 6

    # Save control
    save_lowres_image = True
    save_same_size_pixelated_image = True
    # ------------------------------------------------------------

    image = load_image(input_path)
    original_width, original_height = image.size

    print(f"Input image: {input_path}")
    print(f"Original size: {image.size}")
    print(f"Requested low-resolution size: ({lowres_width}, {lowres_height})")

    lowres_image = create_low_resolution_image(
        image=image,
        lowres_width=lowres_width,
        lowres_height=lowres_height
    )

    same_size_pixelated_image = create_same_size_pixelated_image(
        image=image,
        lowres_width=lowres_width,
        lowres_height=lowres_height
    )

    print(f"Low-resolution output size: {lowres_image.size}")
    print(f"Same-size pixelated output size: {same_size_pixelated_image.size}")

    if same_size_pixelated_image.size != (original_width, original_height):
        raise RuntimeError(
            "Same-size pixelated image does not match the original input size.\n"
            f"Original size = {(original_width, original_height)}\n"
            f"Generated size = {same_size_pixelated_image.size}"
        )

    if save_lowres_image:
        save_image(lowres_image, output_lowres_path)

    if save_same_size_pixelated_image:
        save_image(same_size_pixelated_image, output_same_size_path)


if __name__ == "__main__":
    main()