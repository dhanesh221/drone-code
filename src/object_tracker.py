import numpy as np
from scipy.spatial import distance as dist
from collections import OrderedDict

class ObjectTracker:
    def __init__(self, max_disappeared=10):
        # next_id is the counter for assigning unique IDs
        self.next_id = 0
        # objects stores {id: (centroid_x, centroid_y)}
        self.objects = OrderedDict()
        # disappeared stores {id: frame_count_missing}
        self.disappeared = OrderedDict()
        self.max_disappeared = max_disappeared

    def register(self, centroid):
        self.objects[self.next_id] = centroid
        self.disappeared[self.next_id] = 0
        self.next_id += 1

    def deregister(self, object_id):
        del self.objects[object_id]
        del self.disappeared[object_id]

    def update(self, input_centroids):
        """
        rects: list of [startX, startY, endX, endY]
        """
        # If no detections, increment disappeared count for all existing objects
        if len(input_centroids) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return self.objects

        # If we aren't tracking anything yet, register all centroids
        if len(self.objects) == 0:
            for i in range(0, len(input_centroids)):
                self.register(input_centroids[i])
        else:
            object_ids = list(self.objects.keys())
            object_centroids = list(self.objects.values())

            # Distance matrix between existing objects and new detections
            D = dist.cdist(np.array(object_centroids), input_centroids)

            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            used_rows = set()
            used_cols = set()

            for (row, col) in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue

                object_id = object_ids[row]
                self.objects[object_id] = input_centroids[col]
                self.disappeared[object_id] = 0

                used_rows.add(row)
                used_cols.add(col)

            # Handle disappearing or new objects
            unused_rows = set(range(0, D.shape[0])).difference(used_rows)
            unused_cols = set(range(0, D.shape[1])).difference(used_cols)

            # Existing object lost?
            for row in unused_rows:
                object_id = object_ids[row]
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)

            # New object appeared?
            for col in unused_cols:
                self.register(input_centroids[col])

        return self.objects

# --- Usage Example ---
tracker = ObjectTracker(max_disappeared=5)

# Frame 1: One object detected
frame1_boxes = [[30,30]]
objects = tracker.update(frame1_boxes) # Returns {0: [30, 30]}
print(objects)

# Frame 2: Object 0 moved, and a NEW object (1) appears
frame2_boxes = [[32, 32], [425, 425]]
objects = tracker.update(frame2_boxes) # Returns {0: [32, 32], 1: [425, 425]}
print(objects)
