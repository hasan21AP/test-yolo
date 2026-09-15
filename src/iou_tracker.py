class IoUTracker:

    def __init__(
        self,
        iou_threshold=0.3,
        max_missed=5
    ):
        self.iou_threshold = iou_threshold
        self.max_missed = max_missed

        self.tracks = {}
        self.next_id = 1

    def calculate_iou(self, box1, box2):

        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])

        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        intersection_width = max(0, x2 - x1)
        intersection_height = max(0, y2 - y1)

        intersection = (
            intersection_width
            * intersection_height
        )

        area1 = (
            (box1[2] - box1[0])
            * (box1[3] - box1[1])
        )

        area2 = (
            (box2[2] - box2[0])
            * (box2[3] - box2[1])
        )

        union = area1 + area2 - intersection

        if union == 0:
            return 0.0

        return intersection / union

    def update(self, detections):

        assigned_tracks = set()
        output = []

        for detection in detections:

            bbox = detection["bbox"]
            class_id = detection["class_id"]

            best_iou = 0.0
            best_track_id = None

            for track_id, track in self.tracks.items():

                if track_id in assigned_tracks:
                    continue

                if track["class_id"] != class_id:
                    continue

                iou = self.calculate_iou(
                    bbox,
                    track["bbox"]
                )

                if iou > best_iou:
                    best_iou = iou
                    best_track_id = track_id

            if (
                best_track_id is not None
                and best_iou >= self.iou_threshold
            ):

                track_id = best_track_id

                self.tracks[track_id]["bbox"] = bbox
                self.tracks[track_id]["missed"] = 0

            else:

                track_id = self.next_id
                self.next_id += 1

                self.tracks[track_id] = {
                    "bbox": bbox,
                    "class_id": class_id,
                    "missed": 0
                }

            assigned_tracks.add(track_id)

            detection["track_id"] = track_id

            output.append(detection)

        for track_id in list(self.tracks.keys()):

            if track_id not in assigned_tracks:

                self.tracks[track_id]["missed"] += 1

                if (
                    self.tracks[track_id]["missed"]
                    > self.max_missed
                ):
                    del self.tracks[track_id]

        return output