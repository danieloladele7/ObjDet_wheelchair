import cv2
import numpy as np


class ObjectDetection:
    def __init__(self, weights_path="data/yolov4-tiny.weights", cfg_path="data/yolov4-tiny.cfg", image_size=(416, 416)):
        print("Loading Object Detection")
        print("Running opencv dnn with YOLOv4")
        self.nmsThreshold = 0.6
        self.confThreshold = 0.5
        self.image_size = image_size

        # Load Network
        net = cv2.dnn.readNet(weights_path, cfg_path)

        # Enable GPU CUDA
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        # net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)
        self.model = cv2.dnn_DetectionModel(net)

        self.classes = []
        self.load_class_names()
        self.colors = np.random.uniform(0, 255, size=(len(self.classes), 3))

        self.model.setInputParams(size=self.image_size, scale=1/255)

    # returns a list of class names
    def load_class_names(self, classes_path="data/coco.names"):
        with open(classes_path, "r") as file_object:
            for class_name in file_object.readlines():
                class_name = class_name.strip()
                self.classes.append(class_name)
        return self.classes

    # returns a list of tuples (class, confidence, (x, y, w, h))
    def detect(self, frame, class_names=None):
        if class_names is None:
            return self.model.detect(frame, nmsThreshold=self.nmsThreshold, confThreshold=self.confThreshold)
        else:
            class_indices = [self.classes.index(class_name) for class_name in class_names]
            class_id, conf, boxes = self.model.detect(frame, nmsThreshold=self.nmsThreshold, confThreshold=self.confThreshold)

            filtered_detections = [
                (
                    class_id[i],
                    conf[i],
                    boxes[i]
                )
                for i in range(len(class_id))
                if class_id[i] in class_indices
            ]

            return (
                [d[0] for d in filtered_detections],  # class_id
                [d[1] for d in filtered_detections],  # confidences
                [d[2] for d in filtered_detections]   # boxes
            )
