import cv2
import numpy as np

def enhance_fingerprint(image):
    # Step 1: Convert to grayscale if not already
    if len(image.shape) == 3 and image.shape[2] == 3:
        img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        img = image.copy()

    # Step 2: Denoise using Non-Local Means Denoising
    denoised = cv2.fastNlMeansDenoising(img, None, h=30)

    # Step 3: Contrast Enhancement using CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    contrast_enhanced = clahe.apply(denoised)

    # Step 4: Adaptive Thresholding (Binarization)
    binary = cv2.adaptiveThreshold(
        contrast_enhanced, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )

    # Step 5: Thinning (Skeletonization)
    skel = np.zeros(binary.shape, np.uint8)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    done = False

    while not done:
        eroded = cv2.erode(binary, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(binary, temp)
        skel = cv2.bitwise_or(skel, temp)
        binary = eroded.copy()

        if cv2.countNonZero(binary) == 0:
            done = True

    return skel