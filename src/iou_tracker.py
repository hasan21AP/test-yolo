from src.kalman_filter import KalmanFilter


class IoUTracker:

    def __init__(
        self,
        iou_threshold=0.10,
        max_missed=12,
        distance_threshold=2.0
    ):

        self.iou_threshold = iou_threshold
        self.max_missed = max_missed

        self.tracks = {}
        self.next_id = 1


    # =====================================================
    # DIoU
    # =====================================================

    def calculate_diou(
        self,
        box_a,
        box_b
    ):

        # Intersection
        x1 = max(
            box_a[0],
            box_b[0]
        )

        y1 = max(
            box_a[1],
            box_b[1]
        )

        x2 = min(
            box_a[2],
            box_b[2]
        )

        y2 = min(
            box_a[3],
            box_b[3]
        )


        intersection = (
            max(0, x2 - x1)
            * max(0, y2 - y1)
        )


        area_a = max(
            1.0,
            (
                box_a[2] - box_a[0]
            )
            * (
                box_a[3] - box_a[1]
            )
        )


        area_b = max(
            1.0,
            (
                box_b[2] - box_b[0]
            )
            * (
                box_b[3] - box_b[1]
            )
        )


        union = (
            area_a
            + area_b
            - intersection
        )


        iou = (
            intersection / union
            if union > 0
            else 0.0
        )


        # ---------------------------------------------
        # Smallest enclosing box
        # ---------------------------------------------

        enclosing_x1 = min(
            box_a[0],
            box_b[0]
        )

        enclosing_y1 = min(
            box_a[1],
            box_b[1]
        )

        enclosing_x2 = max(
            box_a[2],
            box_b[2]
        )

        enclosing_y2 = max(
            box_a[3],
            box_b[3]
        )


        enclosing_diagonal_squared = (
            (
                enclosing_x2
                - enclosing_x1
            ) ** 2
            +
            (
                enclosing_y2
                - enclosing_y1
            ) ** 2
        )


        # ---------------------------------------------
        # Center distance
        # ---------------------------------------------

        center_a_x = (
            box_a[0]
            + box_a[2]
        ) / 2.0

        center_a_y = (
            box_a[1]
            + box_a[3]
        ) / 2.0


        center_b_x = (
            box_b[0]
            + box_b[2]
        ) / 2.0

        center_b_y = (
            box_b[1]
            + box_b[3]
        ) / 2.0


        center_distance_squared = (
            (
                center_a_x
                - center_b_x
            ) ** 2
            +
            (
                center_a_y
                - center_b_y
            ) ** 2
        )


        if (
            enclosing_diagonal_squared
            <= 0
        ):
            return iou


        return (
            iou
            -
            (
                center_distance_squared
                /
                enclosing_diagonal_squared
            )
        )


    # =====================================================
    # Kalman only
    # =====================================================

    def predict_only(self):

        output = []


        for track_id, track in (
            self.tracks.items()
        ):

            predicted_box = (
                track["kalman"].predict()
            )


            track["bbox"] = (
                predicted_box
            )


            center_x, center_y = (
                track["kalman"]
                .get_position()
            )

            velocity_x, velocity_y = (
                track["kalman"]
                .get_velocity()
            )


            output.append(
                {
                    "bbox":
                        predicted_box,

                    "track_id":
                        track_id,

                    "class_id":
                        track["class_id"],

                    "class_name":
                        track["class_name"],

                    "confidence":
                        track["confidence"],

                    "predicted_x":
                        center_x,

                    "predicted_y":
                        center_y,

                    "velocity_x":
                        velocity_x,

                    "velocity_y":
                        velocity_y
                }
            )


        return output


    # =====================================================
    # Update using YOLO
    # =====================================================

    def update(
        self,
        detections
    ):

        output = []

        predicted_tracks = {}


        # -------------------------------------------------
        # 1. Predict every existing track
        # -------------------------------------------------

        for track_id, track in (
            self.tracks.items()
        ):

            predicted_box = (
                track["kalman"].predict()
            )

            predicted_tracks[
                track_id
            ] = predicted_box


        assigned_tracks = set()
        matched_detections = set()


        # -------------------------------------------------
        # 2. Match detections with tracks using DIoU
        # -------------------------------------------------

        for detection_index, detection in enumerate(
            detections
        ):

            detection_box = (
                detection["bbox"]
            )


            best_track_id = None
            best_diou = -1.0


            for track_id, predicted_box in (
                predicted_tracks.items()
            ):

                if (
                    track_id
                    in assigned_tracks
                ):
                    continue


                track = self.tracks[
                    track_id
                ]


                # Match same class only
                if (
                    track["class_id"]
                    != detection["class_id"]
                ):
                    continue


                diou = self.calculate_diou(
                    predicted_box,
                    detection_box
                )


                if diou > best_diou:

                    best_diou = diou
                    best_track_id = track_id


            # ---------------------------------------------
            # Match found
            # ---------------------------------------------

            if (
                best_track_id is not None
                and
                best_diou
                >= self.iou_threshold
            ):

                track_id = (
                    best_track_id
                )

                track = self.tracks[
                    track_id
                ]


                corrected_box = (
                    track["kalman"].update(
                        detection_box,
                        detection[
                            "confidence"
                        ]
                    )
                )


                track["bbox"] = (
                    corrected_box
                )

                track["class_id"] = (
                    detection["class_id"]
                )

                track["class_name"] = (
                    detection["class_name"]
                )

                track["confidence"] = (
                    detection["confidence"]
                )

                track["missed"] = 0


                assigned_tracks.add(
                    track_id
                )

                matched_detections.add(
                    detection_index
                )


                center_x, center_y = (
                    track["kalman"]
                    .get_position()
                )

                velocity_x, velocity_y = (
                    track["kalman"]
                    .get_velocity()
                )


                output.append(
                    {
                        "bbox":
                            corrected_box,

                        "track_id":
                            track_id,

                        "class_id":
                            track["class_id"],

                        "class_name":
                            track["class_name"],

                        "confidence":
                            track["confidence"],

                        "predicted_x":
                            center_x,

                        "predicted_y":
                            center_y,

                        "velocity_x":
                            velocity_x,

                        "velocity_y":
                            velocity_y
                    }
                )


        # -------------------------------------------------
        # 3. New detections → new tracks
        # -------------------------------------------------

        for detection_index, detection in enumerate(
            detections
        ):

            if (
                detection_index
                in matched_detections
            ):
                continue


            track_id = (
                self.next_id
            )

            self.next_id += 1


            detection_box = (
                detection["bbox"]
            )


            kalman = KalmanFilter(
                detection_box
            )


            self.tracks[
                track_id
            ] = {

                "bbox":
                    detection_box,

                "class_id":
                    detection["class_id"],

                "class_name":
                    detection["class_name"],

                "confidence":
                    detection["confidence"],

                "missed":
                    0,

                "kalman":
                    kalman
            }


            assigned_tracks.add(
                track_id
            )


            center_x, center_y = (
                kalman.get_position()
            )


            output.append(
                {
                    "bbox":
                        detection_box,

                    "track_id":
                        track_id,

                    "class_id":
                        detection["class_id"],

                    "class_name":
                        detection["class_name"],

                    "confidence":
                        detection["confidence"],

                    "predicted_x":
                        center_x,

                    "predicted_y":
                        center_y,

                    "velocity_x":
                        0.0,

                    "velocity_y":
                        0.0
                }
            )


        # -------------------------------------------------
        # 4. YOLO missed an existing track
        # -------------------------------------------------

        for track_id in list(
            self.tracks.keys()
        ):

            if (
                track_id
                in assigned_tracks
            ):
                continue


            track = self.tracks[
                track_id
            ]


            track["missed"] += 1


            if (
                track["missed"]
                > self.max_missed
            ):

                del self.tracks[
                    track_id
                ]

                continue


            predicted_box = (
                predicted_tracks[
                    track_id
                ]
            )


            track["bbox"] = (
                predicted_box
            )


            center_x, center_y = (
                track["kalman"]
                .get_position()
            )

            velocity_x, velocity_y = (
                track["kalman"]
                .get_velocity()
            )


            output.append(
                {
                    "bbox":
                        predicted_box,

                    "track_id":
                        track_id,

                    "class_id":
                        track["class_id"],

                    "class_name":
                        track["class_name"],

                    "confidence":
                        track["confidence"],

                    "predicted_x":
                        center_x,

                    "predicted_y":
                        center_y,

                    "velocity_x":
                        velocity_x,

                    "velocity_y":
                        velocity_y
                }
            )


        return output