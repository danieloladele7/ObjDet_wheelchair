import numpy as np

class ObjectTracker:
    def __init__(self, max_disappeared=5, max_distance=20):
        self.next_object_id = 0
        self.objects = {}
        self.disappeared = {}
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        self.removed = set()

    def register(self, detection):
        if len(self.removed) > 0:
            object_id = self.removed.pop()
            self.objects[object_id] = detection
            self.disappeared[object_id] = 0
        else:
            self.objects[self.next_object_id] = detection
            self.disappeared[self.next_object_id] = 0
            self.next_object_id += 1

    def deregister(self, object_id):
        self.removed.add(object_id)
        del self.objects[object_id]
        del self.disappeared[object_id]

    def update(self, detections):
        if len(detections) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return self.objects

        if len(self.objects) == 0:
            for i in range(len(detections)):
                self.register(detections[i])
        else:
            object_ids = list(self.objects.keys())
            object_centers = np.array([detection[4:6] for detection in detections])
            object_boxes = np.array([detection[:4] for detection in detections])

            previous_object_centers = np.array([self.objects[object_id][4:6] for object_id in object_ids])
            previous_object_boxes = np.array([self.objects[object_id][:4] for object_id in object_ids])

            distance_centers = np.linalg.norm(object_centers[:, None] - previous_object_centers[None, :], axis=-1)
            distance_boxes = np.linalg.norm(object_boxes[:, None] - previous_object_boxes[None, :], axis=-1)

            rows = distance_centers.min(axis=1).argsort()
            cols = distance_centers.argmin(axis=1)[rows]

            used_rows = set()
            used_cols = set()

            for (row, col) in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue

                if distance_centers[row, col] > self.max_distance:
                    continue

                object_id = object_ids[col]
                self.objects[object_id] = detections[row]
                self.disappeared[object_id] = 0

                used_rows.add(row)
                used_cols.add(col)

            unused_rows = set(range(0, distance_centers.shape[0])).difference(used_rows)
            unused_cols = set(range(0, distance_centers.shape[1])).difference(used_cols)

            if distance_centers.shape[0] >= distance_centers.shape[1]:
                for row in unused_rows:
                    self.register(detections[row])
            else:
                for col in unused_cols:
                    self.deregister(object_ids[col])

        return self.objects
