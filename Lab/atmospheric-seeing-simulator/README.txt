ATMOSPHERIC SEEING SIMULATOR
============================

OVERVIEW
--------
This package simulates long-exposure atmospheric seeing on an input image.

The code takes:
1. an input image,
2. the total arcsecond coverage of the shorter image dimension,
3. the atmospheric seeing FWHM in arcseconds,

and produces a seeing-degraded output image.

The current model applies the seeing effect as a point spread function (PSF) blur.
The recommended default model is the Moffat PSF.

MAIN FILE
---------
atmospheric_seeing_simulator.py

WHAT THE CODE DOES
------------------
The script performs the following steps:

1. Loads the input image.
2. Reads the total angular coverage of the shorter image dimension.
3. Computes the image scale in arcseconds per pixel.
4. Converts the atmospheric seeing FWHM from arcseconds to pixels.
5. Builds a PSF kernel:
   - Moffat PSF, or
   - Gaussian PSF
6. Convolves the image with the PSF kernel.
7. Saves the blurred output image.
8. Prints useful diagnostic information in the terminal.

REQUIRED PYTHON PACKAGES
------------------------
Before running the script, make sure the following packages are installed:

- numpy
- scipy
- pillow

You can install them with:

pip install numpy scipy pillow

HOW TO USE THE CODE DIRECTLY THROUGH THE PYTHON FILE
----------------------------------------------------
The simplest way to use this package is to edit the main block at the bottom of the file and then run the script directly.

Open:
atmospheric_seeing_simulator.py

Then go to the bottom section:

if __name__ == "__main__":

In that section, modify these variables:

input_path = "AIRY_DIFFRACTION.png"
output_path = "AIRY_DIFFRACTION_seeing_blurred.png"
min_dim_arcsec = 8
seeing_fwhm_arcsec = 0.5

WHAT THESE VARIABLES MEAN
-------------------------
1. input_path
   The name or path of the input image.

2. output_path
   The name or path of the output image that will be generated.

3. min_dim_arcsec
   The total angular span, in arcseconds, covered by the shorter dimension of the image.

   Example:
   If the image height is smaller than the width, then this value must represent
   the total arcsecond coverage of the image height.

4. seeing_fwhm_arcsec
   The target atmospheric seeing full width at half maximum, in arcseconds.

MODEL SELECTION
---------------
Inside the main block, the code currently uses:

model="moffat"
beta=4.765

Recommended:
- Use model="moffat" for a more realistic atmospheric seeing profile.
- Use model="gaussian" only if you want a simpler approximation.

If using Gaussian, the beta value is ignored.

EXAMPLE
-------
Suppose you have:
- an input image called AIRY_DIFFRACTION.png
- total shorter-dimension coverage of 8 arcseconds
- atmospheric seeing FWHM of 0.5 arcseconds

Then set:

input_path = "AIRY_DIFFRACTION.png"
output_path = "AIRY_DIFFRACTION_seeing_blurred.png"
min_dim_arcsec = 8
seeing_fwhm_arcsec = 0.5

and keep:

model="moffat"
beta=4.765

HOW TO RUN THE SCRIPT
---------------------
After saving your changes, run the file from the terminal or command prompt:

python atmospheric_seeing_simulator.py

If Python is installed under the command "python3", use:

python3 atmospheric_seeing_simulator.py

WHAT THE SCRIPT OUTPUTS
-----------------------
After running, the script will:

1. generate the blurred image file,
2. save it using the specified output name,
3. print diagnostic values such as:
   - arcseconds per pixel,
   - seeing FWHM in arcseconds,
   - seeing FWHM in pixels,
   - PSF model used,
   - kernel size.

EXAMPLE TERMINAL OUTPUT
-----------------------
Simulation info:
arcsec_per_pixel: ...
seeing_fwhm_arcsec: ...
seeing_fwhm_px: ...
model: moffat
beta: 4.765
kernel_shape: ...

Saved: AIRY_DIFFRACTION_seeing_blurred.png

IMPORTANT NOTES
---------------
1. The script assumes the input seeing value is the target blur FWHM to apply.
2. The script simulates long-exposure seeing as a spatial blur.
3. The script does not currently simulate:
   - short-exposure speckle,
   - time-varying turbulence,
   - anisoplanatic field distortion,
   - telescope tracking error,
   - diffraction blur unless already present in the input image.

RECOMMENDED WORKFLOW
--------------------
1. Prepare or choose an input image.
2. Determine the angular coverage of the shorter image dimension.
3. Choose a seeing FWHM in arcseconds.
4. Edit the variables in the main block.
5. Run the script.
6. Inspect the saved output image.

TROUBLESHOOTING
---------------
Problem:
ModuleNotFoundError for numpy, scipy, or PIL

Solution:
Install the missing package with pip.

Problem:
Input file not found

Solution:
Make sure the image file is in the same folder as the script, or provide the full path.

Problem:
Output image looks too blurred or too weakly blurred

Solution:
Check:
- min_dim_arcsec
- seeing_fwhm_arcsec

If min_dim_arcsec is too small or too large relative to the image size, the pixel scaling will be wrong.

PROJECT PURPOSE
---------------
This code is intended as a clear and reasonably physical simulator of long-exposure atmospheric seeing for image-based astronomy demonstrations, analysis, and educational work.

END
---
For best results, start with the Moffat model and modify only:
- input_path
- output_path
- min_dim_arcsec
- seeing_fwhm_arcsec

Then run the file directly.