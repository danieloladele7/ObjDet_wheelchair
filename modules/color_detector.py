import cv2
import numpy as np
from collections import Counter

def classify_color_roi(image, roi):
    # Extract the specified region of interest (ROI) from the image
    x, y, w, h = roi
    roi_image = image[y:y+h, x:x+w]

    # Convert the ROI image from BGR to HSV color space
    hsv_roi = cv2.cvtColor(roi_image, cv2.COLOR_BGR2HSV)

    # Define color ranges for each category in HSV
    color_ranges = {
        'Green': ((40, 40, 40), (80, 255, 255)),
        'Blue': ((90, 40, 40), (150, 255, 255)),
        'Red': ((0, 40, 40), (10, 255, 255)),
        'Yellow': ((20, 40, 40), (40, 255, 255)),
        'White': ((0, 0, 200), (180, 40, 255)),
        'Black': ((0, 0, 0), (180, 40, 40)),
        'Purple': ((120, 40, 40), (160, 255, 255)),
        'Orange': ((10, 40, 40), (20, 255, 255)),
        'Brown': ((0, 40, 40), (20, 255, 255)),
    }

    # Classify the ROI based on color ranges
    color_counts = Counter()

    for color, (lower_bound, upper_bound) in color_ranges.items():
        mask = cv2.inRange(hsv_roi, np.array(lower_bound), np.array(upper_bound))
        color_counts[color] = cv2.countNonZero(mask)

    # Get the predicted color with the highest count
    predicted_color = color_counts.most_common(1)[0][0]

    return predicted_color
