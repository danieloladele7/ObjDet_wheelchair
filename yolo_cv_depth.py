import cv2
import numpy as np
from modules import yolo, depth
import pyrealsense2 as rs
import time


# Initialize Camera Intel Realsense, and object detector
get_dc = depth.DepthCamera()
obj_det = yolo.ObjectDetection()
class_names = ["person"]

# Define font, colors, and initialize FPS calculation variables
font = cv2.FONT_HERSHEY_PLAIN
colors = np.random.uniform(0, 255, size=(100, 3))
start_time = time.time()
frame_count = 0


while True:
	# Get frame in real time from Realsense camera
	ret, depth_frame, bgr_frame = get_dc.get_frame()
	obj_det.image_size = depth_frame.shape

	# Get object mask
	class_ids, confidences, boxes = obj_det.detect(bgr_frame, class_names)
	
	for i in range(len(class_ids)):
		if class_ids[i] < len(obj_det.classes):  # Checking if class ID is within range
			x, y, w, h = boxes[i]
			cls_id = class_ids[i]
			confidence = confidences[i]  # Using the same index for confidences
			#cv2.putText(frame, "confidence: " + confidence, (100, 50), font, 1, (0, 255, 0), 2)
			
			# Draw bbox
			cv2.rectangle(bgr_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
			print("person: " + str(i) + " is at " + str((x+w)/2))
			
			# Calculate center point and depth
			center_x = x + (w // 2)
			center_y = y + (h // 2)
			distance = depth_frame[center_y, center_x]
			
			print("person: " + str(i) + " is " + str(distance) + " away from camera")
			cv2.circle(bgr_frame, (center_x, center_y), 10, (0, 0, 255), thickness=-1)
			cv2.putText(bgr_frame, "{}mm".format(distance), (center_x, center_y - 20), cv2.FONT_HERSHEY_PLAIN, 2, (0, 200, 0), 2)
			
	# Calculate FPS and display it on the frame
	frame_count += 1
	elapsed_time = time.time() - start_time
	fps = frame_count / elapsed_time
	cv2.putText(bgr_frame, "FPS: {:.2f}".format(fps), (20, 50), font, 1, (0, 255, 0), 2)


	cv2.imshow("depth frame", depth_frame)
	cv2.imshow("Bgr frame", bgr_frame)

	# Check for the 'q' key press to break the loop and exit
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

rs.release()
cv2.destroyAllWindows()

