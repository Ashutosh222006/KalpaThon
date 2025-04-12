# Image resized to 128x128 or desired input shape using Pillow/OpenCV.

# Enhanced for brightness or contrast if needed.

# Converted to NumPy array, normalized, and reshaped.


import cv2
import numpy as np

def enhance_image(image_path):
    # Load image
    image = cv2.imread(image_path)

    # Resize to standard size
    image = cv2.resize(image, (256, 256))

    # Denoising
    image = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)

    # Sharpening
    kernel = np.array([[0, -1, 0],
                       [-1, 5,-1],
                       [0, -1, 0]])
    image = cv2.filter2D(src=image, ddepth=-1, kernel=kernel)

    # Save enhanced image
    enhanced_path = "enhanced_image.jpg"
    cv2.imwrite(enhanced_path, image)
    
    return enhanced_path
