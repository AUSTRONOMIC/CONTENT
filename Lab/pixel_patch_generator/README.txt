PIXEL PATCH GENERATOR
=====================

OVERVIEW
--------
This script generates pixelated versions of an input image in two possible forms:

1. a true low-resolution image, such as 6 x 6 pixels,
2. a same-size pixelated image that keeps the original image dimensions but replaces the content with enlarged low-resolution pixel patches.

This is useful when you want to:
- create a genuinely small low-resolution image,
- create a visual pixel-block version of the original image while preserving the original canvas size.

RECOMMENDED SCRIPT NAME
-----------------------
pixel_patch_generator.py

INPUT FILE
----------
The script currently uses:

AIRY_DIFFRACTION.png

Make sure this image file is placed in the same folder as the Python script, unless a full path is provided.

WHAT THE SCRIPT DOES
--------------------
The script performs the following steps:

1. loads the input image,
2. converts it to RGB,
3. creates a true low-resolution version using the requested grid size,
4. creates a same-size pixelated version by:
   - shrinking the image to low resolution,
   - enlarging it back to the original size using nearest-neighbour resampling,
5. saves one or both outputs depending on the save settings.

MAIN OUTPUT TYPES
-----------------
Output 1:
True low-resolution image

This output is actually small in size, for example:
6 x 6 pixels

Output 2:
Same-size pixelated image

This output has the same dimensions as the original input image, but visually appears pixelated because each low-resolution pixel is expanded into a larger patch.

REQUIRED PYTHON PACKAGE
-----------------------
This script requires Pillow.

Install it with:

pip install pillow

HOW TO USE THE SCRIPT
---------------------
Open the Python file and go to the USER SETTINGS section.

You will see something like this:

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

MEANING OF EACH SETTING
-----------------------
input_path
    The input image file name or full file path.

output_lowres_path
    The file name for the true low-resolution output image.

output_same_size_path
    The file name for the same-size pixelated output image.

lowres_width
    The width of the low-resolution grid.

lowres_height
    The height of the low-resolution grid.

save_lowres_image
    If True, the script saves the true low-resolution image.

save_same_size_pixelated_image
    If True, the script saves the same-size pixelated image.

EXAMPLE
-------
If the source image is 401 x 401 pixels and:

lowres_width = 6
lowres_height = 6

then the outputs will be:

1. low-resolution image:
   6 x 6 pixels

2. same-size pixelated image:
   401 x 401 pixels

The second image will look like a 6 x 6 image expanded over the full original canvas.

HOW TO RUN THE SCRIPT
---------------------
Run the script from the terminal or command prompt:

python pixel_patch_generator.py

If your Python installation uses python3, run:

python3 pixel_patch_generator.py

EXPECTED TERMINAL OUTPUT
------------------------
The script prints useful information such as:

- input image name,
- original image size,
- requested low-resolution size,
- actual low-resolution output size,
- actual same-size pixelated output size,
- saved filenames.

Example:

Input image: AIRY_DIFFRACTION.png
Original size: (401, 401)
Requested low-resolution size: (6, 6)
Low-resolution output size: (6, 6)
Same-size pixelated output size: (401, 401)
Saved: AIRY_DIFFRACTION_lowres_6x6.png | size=(6, 6)
Saved: AIRY_DIFFRACTION_6by6.png | size=(401, 401)

HOW TO SAVE ONLY ONE OUTPUT
---------------------------
To save only the low-resolution image:

save_lowres_image = True
save_same_size_pixelated_image = False

To save only the same-size pixelated image:

save_lowres_image = False
save_same_size_pixelated_image = True

To save both:

save_lowres_image = True
save_same_size_pixelated_image = True

HOW THE SAME-SIZE PIXELATED IMAGE IS CREATED
--------------------------------------------
The script does not directly paint square blocks manually.

Instead, it uses a two-stage process:

1. reduce the image to a low-resolution version,
2. enlarge that low-resolution image back to the original dimensions using nearest-neighbour resampling.

This preserves sharp pixel-block edges and keeps the original canvas size.

IMPORTANT NOTES
---------------
1. The true low-resolution image is physically small in pixel dimensions.
2. The same-size pixelated image keeps the original image dimensions.
3. If the two saved outputs appear visually similar in some viewers, check the printed image sizes in the terminal.
4. The script does not simulate atmospheric seeing. It only generates low-resolution and pixel-patch visual representations.

TROUBLESHOOTING
---------------
Problem:
The low-resolution image is not saved.

Cause:
save_lowres_image may be set to False.

Fix:
Set:
save_lowres_image = True

Also make sure output_lowres_path is defined.

Problem:
The same-size image is not saved.

Cause:
save_same_size_pixelated_image may be set to False.

Fix:
Set:
save_same_size_pixelated_image = True

Problem:
Input file not found.

Cause:
The input image is not in the same folder as the script, or the file name is wrong.

Fix:
Check the file name and path.

Problem:
Both outputs look similar on screen.

Cause:
Some image viewers scale small images for display.

Fix:
Check the printed output sizes in the terminal to confirm the actual saved dimensions.

RECOMMENDED USE
---------------
For most use cases:
- keep both outputs enabled initially,
- verify the dimensions printed in the terminal,
- then disable whichever output you do not need.

END
---
Suggested script name:
pixel_patch_generator.py