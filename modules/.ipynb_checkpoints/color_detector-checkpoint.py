import cv2
import numpy as np
from sklearn.cluster import KMeans
from collections import Counter

def get_dominant_colors(image, k=3):
    pixels = image.reshape((-1, 3))
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(pixels)
    dominant_colors = kmeans.cluster_centers_.astype(int)
    return dominant_colors

def display_roi_and_colors(image_path, roi_coordinates, display=False):
    # Read the image
    image = cv2.imread(image_path)

    # Extract the ROI
    x, y, w, h = roi_coordinates
    roi = image[y:y+h, x:x+w]

    # Get and print the dominant colors in the selected part of the image
    dominant_colors = get_dominant_colors(roi)
    print(f"Dominant Colours: {dominant_colors}")

    # Sort dominant colors by frequency (popularity)
    sorted_colors = sorted(Counter(pixel_colors).items(), key=lambda x: x[1], reverse=True)
    
    # Print the top three predominant colors
    print("\nTop Three Predominant Colors:")
    for i in range(min(3, len(sorted_colors))):
        color, count = sorted_colors[i]
        print(f"Color: {color}, Count: {count}")

    # Display the ROI
    if display:
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.imshow('Image with ROI', image)
        cv2.imshow('Selected ROI', roi)
        cv2.waitKey(0)
        cv2.destroyAllWindows()