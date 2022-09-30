## TIC4005 - Features Implemented

### Pipelining - Allows to select a few video filters and see the result of doing them in a particular sequence

Here is a list of the features implemented on the application, which will be further described in subsequent sections:

Pixelation Filters
Background Image: Overlays a background image on the video canvas.
Greyscale: converts all pixel RGB to colour grey.
Invert: converts all pixel RGB colour to its opposite hue in the colour spectrum.
Red: converts all pixel RGB to colour red.
Green: converts all pixel RGB to colour green.
Blue: converts all pixel RGB to colour blue.
Mirror: Flipping of pixel frames on the x-axis.
Flip: Flipping of pixel frames on the y-axis.
Sepia: Converts all pixel RGB to colour black-brown with a reddish tone.
Saturation: converts all pixel RGB to colour in terms of hue, saturation, and intensity.

Edge Detection Filters
Prewitt Operator: Uses the Prewitt kernel to convolve the image array.
Sobel Operator: Uses the Sobel kernel to convolve the image array.
Laplacian Operator: Uses the Laplacian kernel to convolve the image array.
Sharpen: Uses the Sharpen kernel to convolve the image array.
Gaussian Blur: Uses the Gaussian Kernel to convolve the image array.
Emboss: Uses the Emboss Kernel to convolve the image array.

Features
Video filters can be toggled between GPU and CPU mode.
Video filters can be pipelined and toggled in real-time.
Video filters can be reordered depending on the filter selection.
Configurable computation runs on the Graphical Kernel (configurable)
