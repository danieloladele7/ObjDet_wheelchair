import cv2
import numpy as np
from modules import yolo, depth_pc, color_detector as co_dec, track
import time
import math

# Initialize Camera Intel Realsense, object detector, and DeepSORT tracker
get_dc = depth_pc.DepthCamera()
obj_det = yolo.ObjectDetection()
class_names = ["person", "cup"]

# Define font, colors, and initialize FPS calculation variables
font = cv2.FONT_HERSHEY_PLAIN
colors = np.random.uniform(0, 255, size=(100, 3))
start_time = time.time()
frame_count = 0
confidence_threshold = 0.4  # Adjust as needed

# Initialize the object tracker
tracker = track.ObjectTracker(max_disappeared=5, max_distance=20)

while True:
    # Get frame in real-time from RealSense camera
    ret, depth_frame, bgr_frame = get_dc.get_frame()
    obj_det.image_size = depth_frame.shape

    # Get the frame width and height
    frame_height, frame_width = bgr_frame.shape[:2]

    # Get object mask
    class_ids, confidences, boxes = obj_det.detect(bgr_frame, class_names)

    detections = []
    for i in range(len(class_ids)):
        if class_ids[i] < len(obj_det.classes):
            if confidences[i] >= confidence_threshold:
                top, left, w, h = boxes[i]
                cls_id, confidence = class_ids, confidences[i]

                # Calculate center point and depth
                center_x = top + (w // 2)
                center_y = left + (h // 2)
                distance = depth_frame[center_y, center_x] / 1000

                detections.append((top, left, w, h, center_x, center_y, distance))

    # Update the object tracker with the current detections
    objects = tracker.update(detections)
    
    # Loop over the tracked objects and draw bounding boxes around them
    for track_id, detection in objects.items():
        top, left, w, h, cx, cy, distance = detection

        # Map x and y values to the range [-3.5, 3.5]
        normalized_x = np.interp(cx, [0, frame_width], [-3.5, 3.5])
        normalized_y = np.interp(cy, [0, frame_height], [-3.5, 3.5])

        # display track ID
        cv2.putText(bgr_frame, f"ID: {track_id}", (top, left - 10), font, 1, (255, 10, 0), 2)
        
        # Put distance text at the center of the bounding box
        text_size, _ = cv2.getTextSize("{:.2f}m".format(distance), font, 1, 2)
        cv2.putText(bgr_frame, f"{distance:.2f}m", (cx - text_size[0] // 2, cy + text_size[1] // 2), font, 1, (0, 200, 0), 2)
        
        # Draw bounding boxes
        cv2.rectangle(bgr_frame, (top, left), (top + w, left + h), (0, 255, 0), 2)
        
        # put center point in box
        cv2.circle(bgr_frame, (cx, cy), 4, (0, 0, 255), thickness=-1)
        
        # Get and print predominant color in RGB format
        predominant_color = co_dec.classify_color_roi(bgr_frame, (top, left, w, h))
        # Print the top three predominant colors
        print(f"\nThePredominant Color for ID-{track_id} is: {predominant_color}")

    # Print out the x, y, and z coordinates of each point in the cloud
    print("current frame points:")
    for point in get_dc.get_point_cloud():
        print("x:", point[0], "y:", point[1], "z:", point[2])
    
    # Calculate FPS and display it on the frame
    frame_count += 1
    elapsed_time = time.time() - start_time
    fps = frame_count / elapsed_time
    cv2.putText(bgr_frame, "FPS: {:.2f}".format(fps), (20, 50), font, 1, (0, 255, 0), 2)

    # Make a copy of the points
    prev_detections = detections.copy()
    
    # Display the resulting frame
    cv2.imshow("Bgr frame", bgr_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

get_dc.release()
cv2.destroyAllWindows()
