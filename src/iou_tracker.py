from src.kalman_filter import KalmanFilter


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

        if union <= 0:
            return 0.0

        return intersection / union


    def update(self, detections):

        predicted_tracks = {}

        # -------------------------------------------------
        # 1. Kalman Predict for existing tracks
        # -------------------------------------------------

        for track_id, track in self.tracks.items():

            predicted_x, predicted_y = (
                track["kalman"].predict()
            )

            width = track["width"]
            height = track["height"]

            predicted_box = [
                predicted_x - width / 2,
                predicted_y - height / 2,
                predicted_x + width / 2,
                predicted_y + height / 2
            ]

            predicted_tracks[track_id] = predicted_box


        assigned_tracks = set()
        output = []


        # -------------------------------------------------
        # 2. Match detections with predicted tracks
        # -------------------------------------------------

        for detection in detections:

            bbox = detection["bbox"]
            class_id = detection["class_id"]

            best_iou = 0.0
            best_track_id = None

            for track_id, predicted_box in predicted_tracks.items():

                if track_id in assigned_tracks:
                    continue

                track = self.tracks[track_id]

                if track["class_id"] != class_id:
                    continue

                iou = self.calculate_iou(
                    bbox,
                    predicted_box
                )

                if iou > best_iou:
                    best_iou = iou
                    best_track_id = track_id


            # ---------------------------------------------
            # Existing track
            # ---------------------------------------------

            if (
                best_track_id is not None
                and best_iou >= self.iou_threshold
            ):

                track_id = best_track_id

                x1, y1, x2, y2 = bbox

                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2

                width = x2 - x1
                height = y2 - y1

                kalman = self.tracks[
                    track_id
                ]["kalman"]

                estimated_x, estimated_y = (
                    kalman.update(
                        center_x,
                        center_y
                    )
                )

                self.tracks[track_id]["bbox"] = bbox
                self.tracks[track_id]["width"] = width
                self.tracks[track_id]["height"] = height
                self.tracks[track_id]["missed"] = 0


            # ---------------------------------------------
            # New track
            # ---------------------------------------------

            else:

                track_id = self.next_id
                self.next_id += 1

                x1, y1, x2, y2 = bbox

                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2

                width = x2 - x1
                height = y2 - y1

                kalman = KalmanFilter(
                    center_x,
                    center_y
                )

                estimated_x = center_x
                estimated_y = center_y

                self.tracks[track_id] = {
                    "bbox": bbox,
                    "class_id": class_id,
                    "width": width,
                    "height": height,
                    "missed": 0,
                    "kalman": kalman
                }


            assigned_tracks.add(track_id)

            vx, vy = self.tracks[
                track_id
            ]["kalman"].get_velocity()

            detection["track_id"] = track_id
            detection["predicted_x"] = estimated_x
            detection["predicted_y"] = estimated_y
            detection["velocity_x"] = vx
            detection["velocity_y"] = vy

            output.append(detection)


        # -------------------------------------------------
        # 3. Handle tracks not detected in this frame
        # -------------------------------------------------

        for track_id in list(self.tracks.keys()):

            if track_id not in assigned_tracks:

                self.tracks[track_id]["missed"] += 1

                if (
                    self.tracks[track_id]["missed"]
                    > self.max_missed
                ):
                    del self.tracks[track_id]


        return output