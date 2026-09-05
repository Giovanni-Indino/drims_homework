

import math
from collections import Counter
from itertools import combinations

import cv2
import numpy as np
import rclpy

from cv_bridge import CvBridge
from easy_motion_msgs.srv import DiceIdentification
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import TransformStamped
from image_geometry import PinholeCameraModel
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo
from sensor_msgs.msg import CompressedImage
from sensor_msgs.msg import Image
from std_msgs.msg import Float32
from std_msgs.msg import Int32
from tf2_ros import Buffer
from tf2_ros import TransformBroadcaster
from tf2_ros import TransformListener
from tf_transformations import quaternion_multiply


def _rotate_vector(vector, quaternion):
    """
    Rotate a 3-vector by a quaternion (x, y, z, w).

    Same dependency-light pattern as dice_manipulation_node._rotate_vector
    -- used here to carry a detection's position through a TF rotation by
    hand (see DiceDetectorNode._pose_to_world()).
    """
    x, y, z = vector
    qx, qy, qz, qw = quaternion
    v_q = (x, y, z, 0.0)
    q_conj = (-qx, -qy, -qz, qw)
    rx, ry, rz, _ = quaternion_multiply(
        quaternion_multiply((qx, qy, qz, qw), v_q), q_conj
    )
    return (rx, ry, rz)


class DiceDetectorNode(Node):
    """Detects a yellow die from recorded camera images."""

    def __init__(self) -> None:
        super().__init__("dice_detector_node")

        self.declare_parameter(
            "image_topic",
            "/oak/rgb/image_raw/compressed",
        )
        self.declare_parameter(
            "camera_info_topic",
            "/oak/rgb/camera_info",
        )

        # Setup 1: 0.585
        # Setup 2: 0.625
        self.declare_parameter("board_z", 0.585)
        self.declare_parameter("show_debug_windows", False)

        self.declare_parameter("yellow_h_min", 18)
        self.declare_parameter("yellow_h_max", 35)
        self.declare_parameter("yellow_s_min", 100)
        self.declare_parameter("yellow_v_min", 80)

        self.declare_parameter(
            "minimum_dice_area",
            300.0,
        )
        self.declare_parameter(
            "pip_max_value",
            120,
        )

        # --- dice_identification / dice_manipulation_node integration ---
        # See dice_common.py's module docstring ("simulator today, a real
        # camera-based node tomorrow"): everything downstream only needs
        # this node to (a) serve /dice_identification with face_number +
        # a pose in `world_frame`, and (b) keep broadcasting `dice_tf_frame`
        # live via TF -- see _pose_to_world() and image_callback().
        self.declare_parameter("world_frame", "world")
        self.declare_parameter("dice_tf_frame", "dice_tf")
        self.declare_parameter("publish_dice_tf", True)

        self.image_topic = str(
            self.get_parameter("image_topic").value
        )
        self.camera_info_topic = str(
            self.get_parameter(
                "camera_info_topic"
            ).value
        )
        self.board_z = float(
            self.get_parameter("board_z").value
        )
        self.minimum_dice_area = float(
            self.get_parameter(
                "minimum_dice_area"
            ).value
        )
        self.pip_max_value = int(
            self.get_parameter(
                "pip_max_value"
            ).value
        )
        self.show_debug_windows = bool(
            self.get_parameter("show_debug_windows").value
        )
        self.world_frame = str(
            self.get_parameter("world_frame").value
        )
        self.dice_tf_frame = str(
            self.get_parameter("dice_tf_frame").value
        )
        self.publish_dice_tf = bool(
            self.get_parameter("publish_dice_tf").value
        )

        # Resolves camera_frame -> world_frame (see _pose_to_world()): a
        # fixed hand-eye-calibrated extrinsic (drims_description's
        # camera_calibration_cellN.launch.py, "table_top -> <camera base>"),
        # so a plain lookup_transform() each frame is enough -- it is
        # already cached/static from tf2's point of view, no need to
        # re-derive it by hand or block waiting for it here.
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        # Publishes dice_tf_frame live -- dice_manipulation_node's
        # grasp_orientation() reads it once (via the /dice_identification
        # response's own pose, not a separate TF lookup) to decide which
        # way the jaws should face; pick_dice() then builds both the
        # approach and the grasp pose from that *same* single reading, in
        # world_frame, rather than re-resolving this TF at each of the two
        # separate moves (see dice_manipulation_node's module docstring,
        # "Why position/orientation come from captured or just-commanded
        # values, never a fresh TF lookup mid-sequence" -- frame-to-frame
        # jitter in this broadcast was otherwise showing up as a visible
        # extra rotation while descending onto the die). This TF still
        # matters for RViz/debugging and for anything else that looks up
        # dice_tf_frame live, just not for pick_dice()'s own two moves.
        self.tf_broadcaster = TransformBroadcaster(self)
        self._tf_warned = False

        self.bridge = CvBridge()
        self.frame_count = 0

        self.camera_model = PinholeCameraModel()
        self.camera_model_ready = False
        self.camera_frame = ""

        self.latest_face = None
        self.latest_pose = None
        self.angled_face_history = []
        self.raw_face_number = 0
        self.stable_face_number = 0
        self.pip_position_history = []
        self.pending_face_number = 0
        self.pending_face_frames = 0
        self.invalid_face_frames = 0
        self.ensemble_pending_face = 0
        self.ensemble_pending_frames = 0
        self.ensemble_invalid_frames = 0
        self.ensemble_stable_face = 0
        self.latest_model_results = []
        self.latest_selected_model = "none"
        self.latest_ensemble_score = 0.0
        self.latest_ensemble_margin = 0.0
        self.latest_white_recovery_used = False
        self.latest_shape_confidence = 0.0
        self.white_face_history = []
        self.latest_white_face_confidence = 0.0
        self.bright_face_history = []
        self.top_face_network = None
        self.top_face_probability = 0.55
        self.previous_face_polygon = None
        self.previous_selected_surface_mask = None
        self.previous_die_centre = None
        self.previous_die_orientation = None
        self.face_polygon_missing_frames = 0

        self.image_subscription = (
            self.create_subscription(
                CompressedImage,
                self.image_topic,
                self.image_callback,
                qos_profile_sensor_data,
            )
        )

        self.camera_info_subscription = (
            self.create_subscription(
                CameraInfo,
                self.camera_info_topic,
                self.camera_info_callback,
                qos_profile_sensor_data,
            )
        )

        self.face_publisher = self.create_publisher(
            Int32,
            "/dice_detection/face_number",
            10,
        )
        self.confidence_publisher = self.create_publisher(
            Float32,
            "/dice_detection/confidence",
            10,
        )

        self.pose_publisher = self.create_publisher(
            PoseStamped,
            "/dice_detection/pose",
            10,
        )

        self.debug_publisher = self.create_publisher(
            Image,
            "/dice_detection/debug_image",
            qos_profile_sensor_data,
        )

        self.mask_publisher = self.create_publisher(
            Image,
            "/dice_detection/yellow_mask",
            qos_profile_sensor_data,
        )

        self.identification_service = (
            self.create_service(
                DiceIdentification,
                "/dice_identification",
                self.identification_callback,
            )
        )

        self.get_logger().info(
            f"Listening on {self.image_topic}"
        )
        self.get_logger().info(
            f"Board z value: {self.board_z:.3f} m"
        )

    def show_window(
        self,
        name: str,
        image: np.ndarray,
    ) -> None:
        """Shows only the main result unless diagnostics are requested."""

        if (
            name in ("Dice detection", "Shape mask analysis")
            or self.show_debug_windows
        ):
            cv2.imshow(name, image)

    def show_shape_mask_analysis(
        self,
        image: np.ndarray,
        die_mask: np.ndarray,
        dark_mask: np.ndarray,
        white_mask: np.ndarray,
        candidate_mask: np.ndarray,
        lightness: np.ndarray,
    ) -> None:
        """Displays shape masks, edges and corners without changing detection."""

        def colour_panel(mask: np.ndarray, colour) -> np.ndarray:
            panel = np.zeros((*mask.shape, 3), dtype=np.uint8)
            panel[mask > 0] = colour
            return panel

        def add_label(panel: np.ndarray, label: str) -> np.ndarray:
            cv2.rectangle(panel, (0, 0), (panel.shape[1], 28), (0, 0, 0), -1)
            cv2.putText(
                panel,
                label,
                (8, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            return panel

        masked_lightness = cv2.bitwise_and(
            lightness,
            lightness,
            mask=die_mask,
        )
        blurred_lightness = cv2.GaussianBlur(masked_lightness, (5, 5), 0)
        edge_mask = cv2.Canny(blurred_lightness, 35, 100)
        edge_mask = cv2.bitwise_and(edge_mask, die_mask)

        corner_display = cv2.cvtColor(masked_lightness, cv2.COLOR_GRAY2BGR)
        corners = cv2.goodFeaturesToTrack(
            blurred_lightness,
            maxCorners=30,
            qualityLevel=0.025,
            minDistance=12,
            mask=die_mask,
            blockSize=5,
            useHarrisDetector=False,
        )

        if corners is not None:
            for corner in corners:
                corner_x, corner_y = np.round(corner[0]).astype(int)
                cv2.circle(
                    corner_display,
                    (corner_x, corner_y),
                    5,
                    (0, 255, 255),
                    2,
                )

        combined = image.copy()
        combined[die_mask == 0] = (
            0.20 * combined[die_mask == 0]
        ).astype(np.uint8)
        combined[dark_mask > 0] = (0, 0, 255)
        combined[white_mask > 0] = (255, 0, 255)
        combined[edge_mask > 0] = (255, 255, 0)

        if corners is not None:
            for corner in corners:
                corner_x, corner_y = np.round(corner[0]).astype(int)
                cv2.circle(combined, (corner_x, corner_y), 5, (0, 255, 255), 2)

        panels = [
            add_label(colour_panel(die_mask, (0, 180, 0)), "Yellow die mask"),
            add_label(colour_panel(dark_mask, (0, 0, 255)), "Dark candidates"),
            add_label(colour_panel(white_mask, (255, 0, 255)), "Bright candidates"),
            add_label(colour_panel(candidate_mask, (255, 255, 255)), "Combined candidates"),
            add_label(colour_panel(edge_mask, (255, 255, 0)), "Edges"),
            add_label(corner_display, "Corners"),
        ]

        panel_height = 260
        panel_width = 360
        panels = [
            cv2.resize(panel, (panel_width, panel_height))
            for panel in panels
        ]
        grid = np.vstack(
            [
                np.hstack(panels[:3]),
                np.hstack(panels[3:]),
            ]
        )

        overlay_height = 320
        overlay_width = int(
            round(combined.shape[1] * overlay_height / combined.shape[0])
        )
        combined = cv2.resize(combined, (overlay_width, overlay_height))
        add_label(
            combined,
            "Overlay: red=dark, magenta=bright, cyan=edges, yellow=corners",
        )

        if combined.shape[1] < grid.shape[1]:
            padding = np.zeros(
                (combined.shape[0], grid.shape[1] - combined.shape[1], 3),
                dtype=np.uint8,
            )
            combined = np.hstack((combined, padding))
        elif combined.shape[1] > grid.shape[1]:
            combined = combined[:, :grid.shape[1]]

        analysis = np.vstack((grid, combined))
        self.show_window("Shape mask analysis", analysis)

    def draw_model_summary(
        self,
        image: np.ndarray,
    ) -> None:
        """Draws detector votes and the selected result on the main image."""

        if self.latest_model_results:
            lines = [
                (
                    f"{result['method']}: face {result['face']}  "
                    f"conf {result['confidence']:.2f}"
                )
                for result in self.latest_model_results
            ]
        else:
            lines = ["models: no valid proposal"]

        lines.extend(
            [
                f"selected: {self.latest_selected_model}",
                (
                    f"ensemble: raw {self.raw_face_number}  "
                    f"stable {self.stable_face_number}"
                ),
                (
                    f"score {self.latest_ensemble_score:.2f}  "
                    f"margin {self.latest_ensemble_margin:.2f}"
                ),
            ]
        )

        line_height = 24
        panel_width = min(350, max(250, image.shape[1] - 20))
        panel_height = 18 + line_height * len(lines)
        left = max(10, image.shape[1] - panel_width - 10)
        top = 10
        right = min(image.shape[1] - 1, left + panel_width)
        bottom = min(image.shape[0] - 1, top + panel_height)
        overlay = image.copy()

        cv2.rectangle(
            overlay,
            (left, top),
            (right, bottom),
            (20, 20, 20),
            -1,
        )
        cv2.addWeighted(overlay, 0.68, image, 0.32, 0.0, image)

        for index, line in enumerate(lines):
            colour = (230, 230, 230)

            if line.startswith("selected:"):
                colour = (0, 255, 255)
            elif line.startswith("ensemble:"):
                colour = (255, 0, 255)

            cv2.putText(
                image,
                line,
                (left + 10, top + 20 + index * line_height),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                colour,
                1,
                cv2.LINE_AA,
            )

    def camera_info_callback(
        self,
        message: CameraInfo,
    ) -> None:
        """Updates the calibrated ROS pinhole camera model."""

        self.camera_model.fromCameraInfo(message)
        self.camera_model_ready = True

        self.camera_frame = (
            message.header.frame_id
        )

    def create_yellow_mask(
        self,
        image: np.ndarray,
    ) -> np.ndarray:
        """Finds yellow pixels using HSV filtering."""

        hsv_image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2HSV,
        )

        lower_yellow = np.array(
            [
                int(
                    self.get_parameter(
                        "yellow_h_min"
                    ).value
                ),
                int(
                    self.get_parameter(
                        "yellow_s_min"
                    ).value
                ),
                int(
                    self.get_parameter(
                        "yellow_v_min"
                    ).value
                ),
            ],
            dtype=np.uint8,
        )

        upper_yellow = np.array(
            [
                int(
                    self.get_parameter(
                        "yellow_h_max"
                    ).value
                ),
                255,
                255,
            ],
            dtype=np.uint8,
        )

        yellow_mask = cv2.inRange(
            hsv_image,
            lower_yellow,
            upper_yellow,
        )

        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (5, 5),
        )

        yellow_mask = cv2.morphologyEx(
            yellow_mask,
            cv2.MORPH_OPEN,
            kernel,
        )

        yellow_mask = cv2.morphologyEx(
            yellow_mask,
            cv2.MORPH_CLOSE,
            kernel,
        )

        return yellow_mask

    def find_die_rectangle(
        self,
        yellow_mask: np.ndarray,
    ):
        """Finds a square-like yellow contour."""

        contours, _hierarchy = cv2.findContours(
            yellow_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        image_height, image_width = (
            yellow_mask.shape[:2]
        )

        candidates = []

        for contour in contours:
            contour_area = cv2.contourArea(
                contour
            )

            if contour_area < self.minimum_dice_area:
                continue

            x, y, width, height = (
                cv2.boundingRect(contour)
            )

            margin = 5

            touches_border = (
                x <= margin
                or y <= margin
                or x + width
                >= image_width - margin
                or y + height
                >= image_height - margin
            )

            if touches_border:
                continue

            rectangle = cv2.minAreaRect(
                contour
            )

            centre, size, _angle = rectangle
            rectangle_width, rectangle_height = size

            if (
                rectangle_width <= 0.0
                or rectangle_height <= 0.0
            ):
                continue

            aspect_ratio = max(
                rectangle_width,
                rectangle_height,
            ) / min(
                rectangle_width,
                rectangle_height,
            )

            if aspect_ratio > 1.8:
                continue

            rectangle_area = (
                rectangle_width
                * rectangle_height
            )

            fill_ratio = (
                contour_area
                / rectangle_area
            )

            if fill_ratio < 0.45:
                continue

            box = cv2.boxPoints(rectangle)
            box = np.intp(box)

            candidates.append(
                (
                    contour_area,
                    contour,
                    rectangle,
                    box,
                    centre,
                )
            )

        if not candidates:
            return None

        return max(
            candidates,
            key=lambda candidate: candidate[0],
        )

    @staticmethod
    def get_orientation_degrees(
        rectangle,
    ) -> float:
        """Returns planar orientation from the box."""

        _centre, size, angle = rectangle
        width, height = size

        if width < height:
            angle += 90.0

        while angle >= 90.0:
            angle -= 180.0

        while angle < -90.0:
            angle += 180.0

        return angle

    def calculate_position(
        self,
        pixel_x: float,
        pixel_y: float,
    ):
        """Projects a pixel onto the board plane."""

        if not self.camera_model_ready:
            return None

        # Detection runs on image_raw, so first rectify the detected
        # pixel using the distortion data from CameraInfo.
        rectified_pixel = self.camera_model.rectifyPoint(
            (float(pixel_x), float(pixel_y))
        )
        ray_x, ray_y, ray_z = (
            self.camera_model.projectPixelTo3dRay(
                rectified_pixel
            )
        )

        if abs(ray_z) < 1.0e-9:
            self.get_logger().warning(
                "Camera ray is parallel to the board plane"
            )
            return None

        # The board is represented by z = board_z in the optical
        # camera frame. Scale the camera ray until it reaches that plane.
        ray_scale = self.board_z / float(ray_z)
        position_x = float(ray_x) * ray_scale
        position_y = float(ray_y) * ray_scale

        return (
            position_x,
            position_y,
            self.board_z,
        )

    def count_pips_threshold(
        self,
        image: np.ndarray,
        rectangle,
    ):
        """Counts dark and glare-brightened pips."""

        grayscale = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY,
        )

        hsv_image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2HSV,
        )

        saturation = hsv_image[:, :, 1]
        brightness = hsv_image[:, :, 2]

        # Detect normal black pips.
        dark_mask = cv2.inRange(
            grayscale,
            0,
            self.pip_max_value,
        )

        # Detect pips that appear white due to glare.
        bright_mask = cv2.inRange(
            brightness,
            210,
            255,
        )

        low_saturation_mask = cv2.inRange(
            saturation,
            0,
            70,
        )

        white_pip_mask = cv2.bitwise_and(
            bright_mask,
            low_saturation_mask,
        )

        # Normal views use the reliable black-pip mask.
        candidate_mask = dark_mask

        centre, size, angle = rectangle
        width, height = size

        # A smaller rectangle rejects many side-face dots.
        inner_rectangle = (
            centre,
            (
                width * 0.70,
                height * 0.70,
            ),
            angle,
        )

        inner_box = cv2.boxPoints(
            inner_rectangle
        )
        inner_box = np.intp(inner_box)

        inside_die_mask = np.zeros_like(
            candidate_mask
        )

        cv2.fillConvexPoly(
            inside_die_mask,
            inner_box,
            255,
        )

        pip_mask = cv2.bitwise_and(
            candidate_mask,
            inside_die_mask,
        )

        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (3, 3),
        )

        pip_mask = cv2.morphologyEx(
            pip_mask,
            cv2.MORPH_OPEN,
            kernel,
        )

        contours, _hierarchy = cv2.findContours(
            pip_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        rectangle_area = max(
            width * height,
            1.0,
        )

        minimum_area = max(
            8.0,
            rectangle_area * 0.0005,
        )
        maximum_area = (
            rectangle_area * 0.04
        )

        pip_centres = []

        for contour in contours:
            area = cv2.contourArea(contour)

            if not (
                minimum_area
                <= area
                <= maximum_area
            ):
                continue

            perimeter = cv2.arcLength(
                contour,
                True,
            )

            if perimeter <= 0.0:
                continue

            circularity = (
                4.0
                * math.pi
                * area
                / (perimeter * perimeter)
            )

            if circularity < 0.35:
                continue

            moments = cv2.moments(contour)

            if moments["m00"] == 0.0:
                continue

            pip_x = int(
                moments["m10"]
                / moments["m00"]
            )
            pip_y = int(
                moments["m01"]
                / moments["m00"]
            )

            pip_centres.append(
                (pip_x, pip_y)
            )

        face_number = len(pip_centres)

        return (
            face_number,
            pip_centres,
            pip_mask,
        )

    def count_pips_hough(
        self,
        image: np.ndarray,
        rectangle,
    ):
        """Counts pips from circular edges, regardless of colour."""

        grayscale = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY,
        )

        centre, size, angle = rectangle
        width, height = size

        # Exclude the outer bevel and visible side surfaces.
        inner_rectangle = (
            centre,
            (
                width * 0.76,
                height * 0.76,
            ),
            angle,
        )

        inner_box = cv2.boxPoints(
            inner_rectangle
        )
        inner_box = np.intp(inner_box)

        x, y, roi_width, roi_height = (
            cv2.boundingRect(inner_box)
        )

        image_height, image_width = (
            grayscale.shape[:2]
        )

        x = max(0, x)
        y = max(0, y)
        roi_width = min(
            roi_width,
            image_width - x,
        )
        roi_height = min(
            roi_height,
            image_height - y,
        )

        pip_mask = np.zeros_like(
            grayscale
        )

        if roi_width <= 0 or roi_height <= 0:
            return 0, [], pip_mask

        grayscale_roi = grayscale[
            y:y + roi_height,
            x:x + roi_width,
        ]

        blurred_roi = cv2.GaussianBlur(
            grayscale_roi,
            (5, 5),
            1.2,
        )

        minimum_dimension = max(
            1.0,
            min(width, height),
        )

        minimum_radius = max(
            2,
            int(minimum_dimension * 0.04),
        )
        maximum_radius = max(
            minimum_radius + 1,
            int(minimum_dimension * 0.16),
        )
        minimum_distance = max(
            6,
            int(minimum_dimension * 0.16),
        )

        circles = cv2.HoughCircles(
            blurred_roi,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=minimum_distance,
            param1=60,
            param2=9,
            minRadius=minimum_radius,
            maxRadius=maximum_radius,
        )

        pip_centres = []

        if circles is not None:
            detected_circles = np.round(
                circles[0]
            ).astype(int)

            inner_box_float = inner_box.astype(
                np.float32
            )

            for local_x, local_y, radius in (
                detected_circles
            ):
                pip_x = int(local_x + x)
                pip_y = int(local_y + y)

                inside_top_face = (
                    cv2.pointPolygonTest(
                        inner_box_float,
                        (
                            float(pip_x),
                            float(pip_y),
                        ),
                        False,
                    )
                    >= 0
                )

                if not inside_top_face:
                    continue

                duplicate = False

                for existing in pip_centres:
                    distance = math.hypot(
                        pip_x - existing[0],
                        pip_y - existing[1],
                    )

                    if distance < minimum_distance:
                        duplicate = True
                        break

                if duplicate:
                    continue

                pip_centres.append(
                    (pip_x, pip_y)
                )

                cv2.circle(
                    pip_mask,
                    (pip_x, pip_y),
                    int(radius),
                    255,
                    -1,
                )

        face_number = len(pip_centres)

        return (
            face_number,
            pip_centres,
            pip_mask,
        )

    @staticmethod
    def order_box_points(
        points: np.ndarray,
    ) -> np.ndarray:
        """Orders corners as top-left, top-right, bottom-right, bottom-left."""

        ordered = np.zeros(
            (4, 2),
            dtype=np.float32,
        )

        coordinate_sum = points.sum(axis=1)
        coordinate_difference = np.diff(
            points,
            axis=1,
        ).reshape(-1)

        ordered[0] = points[
            np.argmin(coordinate_sum)
        ]
        ordered[2] = points[
            np.argmax(coordinate_sum)
        ]
        ordered[1] = points[
            np.argmin(coordinate_difference)
        ]
        ordered[3] = points[
            np.argmax(coordinate_difference)
        ]

        return ordered

    def extract_rotated_top_face(
        self,
        image: np.ndarray,
        rectangle,
    ):
        """Rectifies the die and isolates its inner top surface."""

        source_box = cv2.boxPoints(
            rectangle
        ).astype(np.float32)
        source_box = self.order_box_points(
            source_box
        )

        output_size = 220

        destination_box = np.array(
            [
                [0.0, 0.0],
                [output_size - 1.0, 0.0],
                [
                    output_size - 1.0,
                    output_size - 1.0,
                ],
                [0.0, output_size - 1.0],
            ],
            dtype=np.float32,
        )

        transform = cv2.getPerspectiveTransform(
            source_box,
            destination_box,
        )
        inverse_transform = cv2.getPerspectiveTransform(
            destination_box,
            source_box,
        )

        rectified_image = cv2.warpPerspective(
            image,
            transform,
            (output_size, output_size),
        )

        grayscale = cv2.cvtColor(
            rectified_image,
            cv2.COLOR_BGR2GRAY,
        )
        blurred = cv2.GaussianBlur(
            grayscale,
            (5, 5),
            1.0,
        )

        edges = cv2.Canny(
            blurred,
            35,
            100,
        )

        edges = cv2.morphologyEx(
            edges,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(
                cv2.MORPH_RECT,
                (5, 5),
            ),
            iterations=2,
        )

        contours, _hierarchy = cv2.findContours(
            edges,
            cv2.RETR_LIST,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        image_area = float(
            output_size * output_size
        )
        image_centre = np.array(
            [output_size / 2.0, output_size / 2.0]
        )

        top_face_candidates = []

        for contour in contours:
            perimeter = cv2.arcLength(
                contour,
                True,
            )

            if perimeter <= 0.0:
                continue

            approximation = cv2.approxPolyDP(
                contour,
                0.04 * perimeter,
                True,
            )

            if len(approximation) != 4:
                continue

            if not cv2.isContourConvex(
                approximation
            ):
                continue

            area = cv2.contourArea(
                approximation
            )
            area_ratio = area / image_area

            if not 0.30 <= area_ratio <= 0.92:
                continue

            moments = cv2.moments(
                approximation
            )

            if moments["m00"] == 0.0:
                continue

            candidate_centre = np.array(
                [
                    moments["m10"] / moments["m00"],
                    moments["m01"] / moments["m00"],
                ]
            )

            centre_distance = np.linalg.norm(
                candidate_centre - image_centre
            )

            if centre_distance > output_size * 0.28:
                continue

            top_face_candidates.append(
                (
                    area,
                    approximation,
                )
            )

        top_face_mask = np.zeros_like(
            grayscale
        )

        if top_face_candidates:
            _area, top_face_contour = max(
                top_face_candidates,
                key=lambda candidate: candidate[0],
            )

            cv2.fillConvexPoly(
                top_face_mask,
                top_face_contour,
                255,
            )
        else:
            # Safe fallback when glare breaks the detected boundary.
            inset = int(output_size * 0.15)
            cv2.rectangle(
                top_face_mask,
                (inset, inset),
                (
                    output_size - inset,
                    output_size - inset,
                ),
                255,
                -1,
            )

        top_face_mask = cv2.erode(
            top_face_mask,
            cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (7, 7),
            ),
        )

        return (
            rectified_image,
            top_face_mask,
            inverse_transform,
        )

    def count_rotated_face_pips(
        self,
        image: np.ndarray,
        rectangle,
    ):
        """Counts adaptively segmented black and white top-face pips."""

        (
            rectified_image,
            top_face_mask,
            inverse_transform,
        ) = self.extract_rotated_top_face(
            image,
            rectangle,
        )

        grayscale = cv2.cvtColor(
            rectified_image,
            cv2.COLOR_BGR2GRAY,
        )

        hsv_image = cv2.cvtColor(
            rectified_image,
            cv2.COLOR_BGR2HSV,
        )
        lab_image = cv2.cvtColor(
            rectified_image,
            cv2.COLOR_BGR2LAB,
        )

        saturation = hsv_image[:, :, 1]
        brightness = hsv_image[:, :, 2]
        lightness = lab_image[:, :, 0]
        lab_yellow_blue = lab_image[:, :, 2]

        valid_pixels = top_face_mask > 0

        pip_mask = np.zeros(
            image.shape[:2],
            dtype=np.uint8,
        )

        if not np.any(valid_pixels):
            return 0, [], pip_mask

        median_saturation = float(
            np.median(saturation[valid_pixels])
        )
        median_brightness = float(
            np.median(brightness[valid_pixels])
        )
        median_lightness = float(
            np.median(lightness[valid_pixels])
        )
        median_yellow_blue = float(
            np.median(
                lab_yellow_blue[valid_pixels]
            )
        )

        white_saturation_limit = max(
            30,
            int(median_saturation - 25.0),
        )
        white_brightness_limit = max(
            170,
            int(median_brightness - 30.0),
        )
        white_lightness_limit = max(
            160,
            int(median_lightness - 30.0),
        )
        white_yellow_blue_limit = int(
            median_yellow_blue - 10.0
        )

        # White pixels are bright and less yellow than their background.
        hsv_white = (
            valid_pixels
            & (brightness >= white_brightness_limit)
            & (saturation <= white_saturation_limit)
        )
        lab_white = (
            valid_pixels
            & (lightness >= white_lightness_limit)
            & (
                lab_yellow_blue
                <= white_yellow_blue_limit
            )
        )

        white_pip_mask = np.zeros_like(
            grayscale
        )
        white_pip_mask[
            hsv_white | lab_white
        ] = 255

        # Black pips remain well separated by grayscale intensity.
        black_pip_mask = cv2.inRange(
            grayscale,
            0,
            self.pip_max_value,
        )
        black_pip_mask = cv2.bitwise_and(
            black_pip_mask,
            top_face_mask,
        )

        candidate_mask = cv2.bitwise_or(
            black_pip_mask,
            white_pip_mask,
        )
        candidate_mask = cv2.bitwise_and(
            candidate_mask,
            top_face_mask,
        )

        self.show_window(
            "Rectified top face",
            rectified_image,
        )
        self.show_window(
            "White pip mask",
            white_pip_mask,
        )
        self.show_window(
            "Black pip mask",
            black_pip_mask,
        )

        candidate_mask = cv2.morphologyEx(
            candidate_mask,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (3, 3),
            ),
        )

        contours, _hierarchy = cv2.findContours(
            candidate_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        top_face_area = max(
            float(cv2.countNonZero(top_face_mask)),
            1.0,
        )
        minimum_area = max(
            18.0,
            top_face_area * 0.0005,
        )
        maximum_area = (
            top_face_area * 0.035
        )

        candidates = []

        for contour in contours:
            area = cv2.contourArea(contour)

            if not minimum_area <= area <= maximum_area:
                continue

            perimeter = cv2.arcLength(
                contour,
                True,
            )

            if perimeter <= 0.0:
                continue

            circularity = (
                4.0
                * math.pi
                * area
                / (perimeter * perimeter)
            )

            if circularity < 0.38:
                continue

            x, y, width, height = cv2.boundingRect(
                contour
            )

            if width <= 0 or height <= 0:
                continue

            aspect_ratio = max(
                width,
                height,
            ) / min(
                width,
                height,
            )

            if aspect_ratio > 1.8:
                continue

            moments = cv2.moments(contour)

            if moments["m00"] == 0.0:
                continue

            centre_x = float(
                moments["m10"] / moments["m00"]
            )
            centre_y = float(
                moments["m01"] / moments["m00"]
            )

            candidates.append(
                (
                    circularity,
                    area,
                    centre_x,
                    centre_y,
                )
            )

        # Prefer strong circular candidates.
        candidates.sort(
            key=lambda candidate: (
                candidate[0],
                candidate[1],
            ),
            reverse=True,
        )

        # Merge several contours produced by one physical pip.
        merged_candidates = []
        merge_distance = 22.0

        for candidate in candidates:
            candidate_x = candidate[2]
            candidate_y = candidate[3]

            duplicate = False

            for existing in merged_candidates:
                distance = math.hypot(
                    candidate_x - existing[2],
                    candidate_y - existing[3],
                )

                if distance < merge_distance:
                    duplicate = True
                    break

            if not duplicate:
                merged_candidates.append(candidate)

        candidates = merged_candidates[:6]

        pip_centres = []

        for (
            _circularity,
            _area,
            centre_x,
            centre_y,
        ) in candidates:
            rectified_point = np.array(
                [[[centre_x, centre_y]]],
                dtype=np.float32,
            )

            original_point = cv2.perspectiveTransform(
                rectified_point,
                inverse_transform,
            )[0, 0]

            pip_x = int(round(original_point[0]))
            pip_y = int(round(original_point[1]))

            pip_centres.append(
                (pip_x, pip_y)
            )

            cv2.circle(
                pip_mask,
                (pip_x, pip_y),
                6,
                255,
                -1,
            )

        return (
            len(pip_centres),
            pip_centres,
            pip_mask,
        )

    def count_verified_rotated_face_pips(
        self,
        image: np.ndarray,
        rectangle,
    ):
        """Verifies angled-face circle candidates using local colour and layout."""

        (
            rectified_image,
            top_face_mask,
            inverse_transform,
        ) = self.extract_rotated_top_face(
            image,
            rectangle,
        )

        grayscale = cv2.cvtColor(
            rectified_image,
            cv2.COLOR_BGR2GRAY,
        )
        hsv_image = cv2.cvtColor(
            rectified_image,
            cv2.COLOR_BGR2HSV,
        )
        lab_image = cv2.cvtColor(
            rectified_image,
            cv2.COLOR_BGR2LAB,
        )

        saturation = hsv_image[:, :, 1]
        brightness = hsv_image[:, :, 2]
        lightness = lab_image[:, :, 0]
        lab_yellow_blue = lab_image[:, :, 2]

        blurred = cv2.GaussianBlur(
            grayscale,
            (5, 5),
            1.2,
        )

        top_face_points = cv2.findNonZero(
            top_face_mask
        )

        pip_mask = np.zeros(
            image.shape[:2],
            dtype=np.uint8,
        )

        if top_face_points is None:
            return 0, [], pip_mask

        face_x, face_y, face_width, face_height = (
            cv2.boundingRect(top_face_points)
        )

        minimum_dimension = float(
            max(1, min(face_width, face_height))
        )

        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=max(
                8,
                int(minimum_dimension * 0.08),
            ),
            param1=60,
            param2=8,
            minRadius=max(
                3,
                int(minimum_dimension * 0.025),
            ),
            maxRadius=max(
                7,
                int(minimum_dimension * 0.14),
            ),
        )

        self.show_window(
            "Rectified top face",
            rectified_image,
        )

        if circles is None:
            return 0, [], pip_mask

        yellow_h_min = int(
            self.get_parameter(
                "yellow_h_min"
            ).value
        )
        yellow_h_max = int(
            self.get_parameter(
                "yellow_h_max"
            ).value
        )

        yellow_mask = cv2.inRange(
            hsv_image,
            np.array(
                [yellow_h_min, 60, 60],
                dtype=np.uint8,
            ),
            np.array(
                [yellow_h_max, 255, 255],
                dtype=np.uint8,
            ),
        )
        yellow_mask = cv2.bitwise_and(
            yellow_mask,
            top_face_mask,
        )

        verified_candidates = []

        detected_circles = np.round(
            circles[0]
        ).astype(int)

        for centre_x, centre_y, radius in (
            detected_circles
        ):
            if not (
                0 <= centre_x < grayscale.shape[1]
                and 0 <= centre_y < grayscale.shape[0]
            ):
                continue

            if top_face_mask[
                centre_y,
                centre_x,
            ] == 0:
                continue

            centre_radius = max(
                2,
                int(round(radius * 0.70)),
            )
            ring_inner_radius = max(
                centre_radius + 1,
                int(round(radius * 1.25)),
            )
            ring_outer_radius = max(
                ring_inner_radius + 2,
                int(round(radius * 2.10)),
            )

            centre_mask = np.zeros_like(
                grayscale
            )
            cv2.circle(
                centre_mask,
                (centre_x, centre_y),
                centre_radius,
                255,
                -1,
            )
            centre_mask = cv2.bitwise_and(
                centre_mask,
                top_face_mask,
            )

            ring_mask = np.zeros_like(
                grayscale
            )
            cv2.circle(
                ring_mask,
                (centre_x, centre_y),
                ring_outer_radius,
                255,
                -1,
            )
            cv2.circle(
                ring_mask,
                (centre_x, centre_y),
                ring_inner_radius,
                0,
                -1,
            )

            complete_ring_area = cv2.countNonZero(
                ring_mask
            )

            if complete_ring_area == 0:
                continue

            ring_mask = cv2.bitwise_and(
                ring_mask,
                top_face_mask,
            )

            valid_ring_area = cv2.countNonZero(
                ring_mask
            )

            ring_coverage = (
                valid_ring_area
                / complete_ring_area
            )

            if ring_coverage < 0.55:
                continue

            yellow_ring = cv2.bitwise_and(
                yellow_mask,
                ring_mask,
            )

            yellow_fraction = (
                cv2.countNonZero(yellow_ring)
                / max(valid_ring_area, 1)
            )

            if yellow_fraction < 0.45:
                continue

            centre_indices = centre_mask > 0
            ring_indices = ring_mask > 0

            if (
                not np.any(centre_indices)
                or not np.any(ring_indices)
            ):
                continue

            centre_saturation = float(
                np.median(
                    saturation[centre_indices]
                )
            )
            ring_saturation = float(
                np.median(
                    saturation[ring_indices]
                )
            )
            centre_brightness = float(
                np.median(
                    brightness[centre_indices]
                )
            )
            ring_brightness = float(
                np.median(
                    brightness[ring_indices]
                )
            )
            centre_lightness = float(
                np.median(
                    lightness[centre_indices]
                )
            )
            ring_lightness = float(
                np.median(
                    lightness[ring_indices]
                )
            )
            centre_yellow_blue = float(
                np.median(
                    lab_yellow_blue[
                        centre_indices
                    ]
                )
            )
            ring_yellow_blue = float(
                np.median(
                    lab_yellow_blue[
                        ring_indices
                    ]
                )
            )

            saturation_contrast = (
                ring_saturation
                - centre_saturation
            )
            yellow_contrast = (
                ring_yellow_blue
                - centre_yellow_blue
            )
            brightness_contrast = (
                ring_brightness
                - centre_brightness
            )
            lightness_contrast = (
                centre_lightness
                - ring_lightness
            )

            is_white_pip = (
                centre_brightness
                >= ring_brightness - 40.0
                and (
                    saturation_contrast >= 20.0
                    or yellow_contrast >= 8.0
                    or lightness_contrast >= 12.0
                )
            )
            is_black_pip = (
                brightness_contrast >= 35.0
            )

            if not (is_white_pip or is_black_pip):
                continue

            if is_black_pip:
                verification_score = (
                    brightness_contrast
                    + 20.0 * yellow_fraction
                )
                pip_kind = "black"
            else:
                verification_score = (
                    max(
                        saturation_contrast,
                        yellow_contrast * 2.0,
                        lightness_contrast,
                    )
                    + 20.0 * yellow_fraction
                )
                pip_kind = "white"

            verified_candidates.append(
                {
                    "x": float(centre_x),
                    "y": float(centre_y),
                    "radius": int(radius),
                    "score": float(
                        verification_score
                    ),
                    "kind": pip_kind,
                }
            )

        verified_candidates.sort(
            key=lambda candidate: candidate["score"],
            reverse=True,
        )

        # Keep one result for each physical pip.
        merged_candidates = []
        merge_distance = max(
            14.0,
            minimum_dimension * 0.12,
        )

        for candidate in verified_candidates:
            duplicate = False

            for existing in merged_candidates:
                distance = math.hypot(
                    candidate["x"] - existing["x"],
                    candidate["y"] - existing["y"],
                )

                if distance < merge_distance:
                    duplicate = True
                    break

            if not duplicate:
                merged_candidates.append(candidate)

        # Assign candidates to the nearest cell of a 3-by-3 dice grid.
        cell_candidates = {}

        for candidate in merged_candidates:
            normalised_x = (
                candidate["x"] - face_x
            ) / max(face_width - 1, 1)
            normalised_y = (
                candidate["y"] - face_y
            ) / max(face_height - 1, 1)

            grid_x = int(round(normalised_x * 2.0))
            grid_y = int(round(normalised_y * 2.0))

            if not (
                0 <= grid_x <= 2
                and 0 <= grid_y <= 2
            ):
                continue

            target_x = grid_x / 2.0
            target_y = grid_y / 2.0

            if (
                abs(normalised_x - target_x) > 0.28
                or abs(normalised_y - target_y) > 0.28
            ):
                continue

            cell = (grid_x, grid_y)
            existing = cell_candidates.get(cell)

            if (
                existing is None
                or candidate["score"]
                > existing["score"]
            ):
                cell_candidates[cell] = candidate

        centre_cell = (1, 1)
        top_left = (0, 0)
        top_right = (2, 0)
        bottom_left = (0, 2)
        bottom_right = (2, 2)

        valid_patterns = {
            1: [
                {centre_cell},
            ],
            2: [
                {top_left, bottom_right},
                {top_right, bottom_left},
            ],
            3: [
                {
                    top_left,
                    centre_cell,
                    bottom_right,
                },
                {
                    top_right,
                    centre_cell,
                    bottom_left,
                },
            ],
            4: [
                {
                    top_left,
                    top_right,
                    bottom_left,
                    bottom_right,
                },
            ],
            5: [
                {
                    top_left,
                    top_right,
                    centre_cell,
                    bottom_left,
                    bottom_right,
                },
            ],
            6: [
                {
                    (0, 0),
                    (0, 1),
                    (0, 2),
                    (2, 0),
                    (2, 1),
                    (2, 2),
                },
                {
                    (0, 0),
                    (1, 0),
                    (2, 0),
                    (0, 2),
                    (1, 2),
                    (2, 2),
                },
            ],
        }

        detected_cells = set(
            cell_candidates.keys()
        )
        face_number = 0
        selected_cells = set()

        for candidate_face, patterns in (
            valid_patterns.items()
        ):
            for pattern in patterns:
                if detected_cells == pattern:
                    face_number = candidate_face
                    selected_cells = pattern
                    break

            if face_number != 0:
                break

        if face_number == 0:
            selected_cells = detected_cells

        pip_centres = []

        for cell in selected_cells:
            candidate = cell_candidates[cell]

            rectified_point = np.array(
                [
                    [
                        [
                            candidate["x"],
                            candidate["y"],
                        ]
                    ]
                ],
                dtype=np.float32,
            )

            original_point = cv2.perspectiveTransform(
                rectified_point,
                inverse_transform,
            )[0, 0]

            pip_x = int(round(original_point[0]))
            pip_y = int(round(original_point[1]))

            pip_centres.append(
                (pip_x, pip_y)
            )

            cv2.circle(
                pip_mask,
                (pip_x, pip_y),
                6,
                255,
                -1,
            )

        return (
            face_number,
            pip_centres,
            pip_mask,
        )

    def count_merged_rotated_face_pips(
        self,
        image: np.ndarray,
        rectangle,
    ):
        """Counts angled-face pips by merging overlapping circles."""

        (
            rectified_image,
            top_face_mask,
            inverse_transform,
        ) = self.extract_rotated_top_face(
            image,
            rectangle,
        )

        grayscale = cv2.cvtColor(
            rectified_image,
            cv2.COLOR_BGR2GRAY,
        )
        blurred = cv2.GaussianBlur(
            grayscale,
            (5, 5),
            1.2,
        )
        edge_image = cv2.Canny(
            blurred,
            40,
            110,
        )

        top_face_points = cv2.findNonZero(
            top_face_mask
        )
        pip_mask = np.zeros(
            image.shape[:2],
            dtype=np.uint8,
        )

        if top_face_points is None:
            return 0, [], pip_mask

        _x, _y, face_width, face_height = (
            cv2.boundingRect(top_face_points)
        )
        minimum_dimension = float(
            max(1, min(face_width, face_height))
        )

        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=max(
                6,
                int(minimum_dimension * 0.05),
            ),
            param1=60,
            param2=8,
            minRadius=max(
                3,
                int(minimum_dimension * 0.025),
            ),
            maxRadius=max(
                7,
                int(minimum_dimension * 0.14),
            ),
        )

        self.show_window(
            "Rectified top face",
            rectified_image,
        )
        self.show_window(
            "Top face edges",
            edge_image,
        )

        filtered_circles = []

        if circles is not None:
            detected_circles = np.round(
                circles[0]
            ).astype(int)

            for centre_x, centre_y, radius in (
                detected_circles
            ):
                if not (
                    0 <= centre_x < grayscale.shape[1]
                    and 0 <= centre_y < grayscale.shape[0]
                ):
                    continue

                if top_face_mask[
                    centre_y,
                    centre_x,
                ] == 0:
                    continue

                circle_mask = np.zeros_like(
                    grayscale
                )
                cv2.circle(
                    circle_mask,
                    (centre_x, centre_y),
                    int(radius),
                    255,
                    -1,
                )

                circle_area = cv2.countNonZero(
                    circle_mask
                )
                inside_face = cv2.bitwise_and(
                    circle_mask,
                    top_face_mask,
                )
                inside_fraction = (
                    cv2.countNonZero(inside_face)
                    / max(circle_area, 1)
                )

                if inside_fraction < 0.80:
                    continue

                edge_ring = np.zeros_like(
                    grayscale
                )
                outer_radius = int(radius + 2)
                inner_radius = max(
                    1,
                    int(radius - 2),
                )
                cv2.circle(
                    edge_ring,
                    (centre_x, centre_y),
                    outer_radius,
                    255,
                    -1,
                )
                cv2.circle(
                    edge_ring,
                    (centre_x, centre_y),
                    inner_radius,
                    0,
                    -1,
                )

                ring_area = cv2.countNonZero(
                    edge_ring
                )
                supported_edges = cv2.bitwise_and(
                    edge_image,
                    edge_ring,
                )
                edge_support = (
                    cv2.countNonZero(supported_edges)
                    / max(ring_area, 1)
                )

                if edge_support < 0.06:
                    continue

                filtered_circles.append(
                    {
                        "x": float(centre_x),
                        "y": float(centre_y),
                        "radius": float(radius),
                        "score": float(edge_support),
                    }
                )

        # Build groups of overlapping detections.
        groups = []

        for candidate in filtered_circles:
            matching_group = None

            for group in groups:
                overlaps_group = False

                for member in group:
                    centre_distance = math.hypot(
                        candidate["x"] - member["x"],
                        candidate["y"] - member["y"],
                    )
                    overlap_limit = max(
                        0.65
                        * (
                            candidate["radius"]
                            + member["radius"]
                        ),
                        minimum_dimension * 0.10,
                    )

                    if centre_distance < overlap_limit:
                        overlaps_group = True
                        break

                if overlaps_group:
                    matching_group = group
                    break

            if matching_group is None:
                groups.append([candidate])
            else:
                matching_group.append(candidate)

        merged_circles = []

        for group in groups:
            total_weight = sum(
                max(member["score"], 0.001)
                for member in group
            )
            merged_x = sum(
                member["x"]
                * max(member["score"], 0.001)
                for member in group
            ) / total_weight
            merged_y = sum(
                member["y"]
                * max(member["score"], 0.001)
                for member in group
            ) / total_weight
            merged_radius = sum(
                member["radius"]
                * max(member["score"], 0.001)
                for member in group
            ) / total_weight
            merged_score = max(
                member["score"]
                for member in group
            )

            merged_circles.append(
                {
                    "x": merged_x,
                    "y": merged_y,
                    "radius": merged_radius,
                    "score": merged_score,
                }
            )

        # Prefer circles with radii similar to the median pip radius.
        if merged_circles:
            median_radius = float(
                np.median(
                    [
                        circle["radius"]
                        for circle in merged_circles
                    ]
                )
            )
            merged_circles = [
                circle
                for circle in merged_circles
                if (
                    0.55 * median_radius
                    <= circle["radius"]
                    <= 1.60 * median_radius
                )
            ]

        merged_circles.sort(
            key=lambda circle: circle["score"],
            reverse=True,
        )

        raw_face_number = len(merged_circles)

        if not 1 <= raw_face_number <= 6:
            raw_face_number = 0

        # Stabilise the count over ten camera frames.
        self.angled_face_history.append(
            raw_face_number
        )
        self.angled_face_history = (
            self.angled_face_history[-10:]
        )

        valid_history = [
            value
            for value in self.angled_face_history
            if 1 <= value <= 6
        ]

        stable_face_number = 0

        if valid_history:
            most_common_face, occurrences = (
                Counter(valid_history).most_common(1)[0]
            )

            required_occurrences = max(
                2,
                int(
                    math.ceil(
                        len(self.angled_face_history)
                        * 0.60
                    )
                ),
            )

            if occurrences >= required_occurrences:
                stable_face_number = (
                    most_common_face
                )

        # Draw current merged candidates even while voting stabilises.
        pip_centres = []

        for circle in merged_circles[:6]:
            rectified_point = np.array(
                [
                    [
                        [circle["x"], circle["y"]]
                    ]
                ],
                dtype=np.float32,
            )
            original_point = cv2.perspectiveTransform(
                rectified_point,
                inverse_transform,
            )[0, 0]

            pip_x = int(round(original_point[0]))
            pip_y = int(round(original_point[1]))
            pip_centres.append(
                (pip_x, pip_y)
            )
            cv2.circle(
                pip_mask,
                (pip_x, pip_y),
                6,
                255,
                -1,
            )

        return (
            stable_face_number,
            pip_centres,
            pip_mask,
        )

    def extract_learned_top_surface(
        self,
        image: np.ndarray,
        complete_die_mask: np.ndarray,
    ):
        """Segments and rectifies the physical upward face with ONNX."""

        if self.top_face_network is None:
            return None

        die_points = cv2.findNonZero(complete_die_mask)

        if die_points is None:
            return None

        die_x, die_y, die_width, die_height = cv2.boundingRect(
            die_points
        )
        padding = max(
            4,
            int(round(0.18 * max(die_width, die_height))),
        )
        crop_x0 = max(0, die_x - padding)
        crop_y0 = max(0, die_y - padding)
        crop_x1 = min(
            image.shape[1],
            die_x + die_width + padding,
        )
        crop_y1 = min(
            image.shape[0],
            die_y + die_height + padding,
        )
        die_crop = image[crop_y0:crop_y1, crop_x0:crop_x1]

        if die_crop.size == 0:
            return None

        model_size = 128
        model_input = cv2.resize(
            die_crop,
            (model_size, model_size),
            interpolation=cv2.INTER_AREA,
        )
        blob = cv2.dnn.blobFromImage(
            model_input,
            scalefactor=1.0 / 127.5,
            size=(model_size, model_size),
            mean=(127.5, 127.5, 127.5),
            swapRB=True,
            crop=False,
        )
        self.top_face_network.setInput(blob)
        network_output = self.top_face_network.forward()
        probability = np.squeeze(network_output).astype(np.float32)

        if probability.shape != (model_size, model_size):
            probability = cv2.resize(
                probability,
                (model_size, model_size),
                interpolation=cv2.INTER_LINEAR,
            )

        if float(np.min(probability)) < 0.0 or float(
            np.max(probability)
        ) > 1.0:
            probability = 1.0 / (1.0 + np.exp(-probability))

        crop_probability = cv2.resize(
            probability,
            (die_crop.shape[1], die_crop.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )
        crop_mask = np.zeros(
            die_crop.shape[:2],
            dtype=np.uint8,
        )
        crop_mask[
            crop_probability >= self.top_face_probability
        ] = 255
        crop_mask = cv2.morphologyEx(
            crop_mask,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (3, 3),
            ),
        )
        crop_mask = cv2.morphologyEx(
            crop_mask,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (7, 7),
            ),
            iterations=2,
        )

        predicted_mask = np.zeros(
            image.shape[:2],
            dtype=np.uint8,
        )
        predicted_mask[crop_y0:crop_y1, crop_x0:crop_x1] = crop_mask
        predicted_mask = cv2.bitwise_and(
            predicted_mask,
            complete_die_mask,
        )
        contours, _hierarchy = cv2.findContours(
            predicted_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        if not contours:
            return None

        top_contour = max(contours, key=cv2.contourArea)
        top_area = cv2.contourArea(top_contour)
        die_contours, _hierarchy = cv2.findContours(
            complete_die_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        if not die_contours:
            return None

        die_area = max(
            cv2.contourArea(max(die_contours, key=cv2.contourArea)),
            1.0,
        )
        top_area_ratio = top_area / die_area

        if not 0.12 <= top_area_ratio <= 0.94:
            return None

        contour_mask = np.zeros_like(predicted_mask)
        cv2.drawContours(
            contour_mask,
            [top_contour],
            -1,
            255,
            thickness=-1,
        )
        local_contour_mask = contour_mask[
            crop_y0:crop_y1,
            crop_x0:crop_x1,
        ]
        selected_probabilities = crop_probability[
            local_contour_mask > 0
        ]

        if selected_probabilities.size == 0:
            return None

        segmentation_confidence = float(
            np.mean(selected_probabilities)
        )

        if segmentation_confidence < max(
            0.58,
            self.top_face_probability,
        ):
            return None

        top_hull = cv2.convexHull(top_contour)
        source_box = None
        perimeter = cv2.arcLength(top_hull, True)

        for factor in (0.02, 0.03, 0.04, 0.05, 0.07, 0.09):
            polygon = cv2.approxPolyDP(
                top_hull,
                factor * perimeter,
                True,
            )

            if len(polygon) == 4 and cv2.isContourConvex(polygon):
                if cv2.contourArea(polygon) >= 0.72 * top_area:
                    source_box = polygon.reshape(4, 2).astype(np.float32)
                    break

        if source_box is None:
            source_box = cv2.boxPoints(
                cv2.minAreaRect(top_hull)
            ).astype(np.float32)

        source_box = self.order_box_points(source_box)

        if self.previous_face_polygon is not None:
            corner_change = np.linalg.norm(
                source_box - self.previous_face_polygon,
                axis=1,
            )

            if float(np.max(corner_change)) <= max(
                14.0,
                0.28 * min(die_width, die_height),
            ):
                source_box = (
                    0.75 * self.previous_face_polygon
                    + 0.25 * source_box
                ).astype(np.float32)
            else:
                self.angled_face_history.clear()
                self.pip_position_history.clear()
                self.pending_face_number = 0
                self.pending_face_frames = 0

        self.previous_face_polygon = source_box.copy()
        self.previous_selected_surface_mask = contour_mask.copy()
        self.face_polygon_missing_frames = 0

        output_size = 220
        destination_box = np.array(
            [
                [0.0, 0.0],
                [output_size - 1.0, 0.0],
                [output_size - 1.0, output_size - 1.0],
                [0.0, output_size - 1.0],
            ],
            dtype=np.float32,
        )
        transform = cv2.getPerspectiveTransform(
            source_box,
            destination_box,
        )
        inverse_transform = cv2.getPerspectiveTransform(
            destination_box,
            source_box,
        )
        rectified_surface = cv2.warpPerspective(
            image,
            transform,
            (output_size, output_size),
        )
        rectified_mask = cv2.warpPerspective(
            contour_mask,
            transform,
            (output_size, output_size),
            flags=cv2.INTER_NEAREST,
        )
        erosion_size = max(
            7,
            int(round(output_size * 0.055)),
        )

        if erosion_size % 2 == 0:
            erosion_size += 1

        rectified_mask = cv2.erode(
            rectified_mask,
            cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (erosion_size, erosion_size),
            ),
            iterations=1,
        )
        probability_display = np.clip(
            crop_probability * 255.0,
            0.0,
            255.0,
        ).astype(np.uint8)
        self.show_window("Top face probability", probability_display)
        self.show_window("Learned top face mask", contour_mask)
        self.show_window("Rectified top face", rectified_surface)
        self.show_window("Rectified valid face mask", rectified_mask)

        return (
            rectified_surface,
            rectified_mask,
            inverse_transform,
            top_area_ratio,
            contour_mask,
            source_box,
        )

    def extract_adaptive_top_surface(
        self,
        image: np.ndarray,
        rectangle,
    ):
        """Finds the bright visible face and rectifies only that surface."""

        complete_box = cv2.boxPoints(
            rectangle
        ).astype(np.float32)

        complete_box_integer = np.intp(
            complete_box
        )
        box_mask = np.zeros(
            image.shape[:2],
            dtype=np.uint8,
        )
        cv2.fillConvexPoly(
            box_mask,
            complete_box_integer,
            255,
        )

        hsv_image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2HSV,
        )
        lab_image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2LAB,
        )
        lightness = lab_image[:, :, 0]

        yellow_h_min = int(
            self.get_parameter(
                "yellow_h_min"
            ).value
        )
        yellow_h_max = int(
            self.get_parameter(
                "yellow_h_max"
            ).value
        )

        relaxed_yellow_mask = cv2.inRange(
            hsv_image,
            np.array(
                [yellow_h_min, 45, 45],
                dtype=np.uint8,
            ),
            np.array(
                [yellow_h_max, 255, 255],
                dtype=np.uint8,
            ),
        )
        relaxed_yellow_mask = cv2.bitwise_and(
            relaxed_yellow_mask,
            box_mask,
        )

        # Fill black and white pip holes to recover the complete die body.
        complete_die_mask = cv2.morphologyEx(
            relaxed_yellow_mask,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (13, 13),
            ),
            iterations=2,
        )

        die_contours, _hierarchy = cv2.findContours(
            complete_die_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        if not die_contours:
            return None

        complete_die_contour = max(
            die_contours,
            key=cv2.contourArea,
        )
        complete_die_hull = cv2.convexHull(
            complete_die_contour
        )
        complete_die_mask[:] = 0
        cv2.fillConvexPoly(
            complete_die_mask,
            complete_die_hull,
            255,
        )

        die_x, die_y, die_width, die_height = cv2.boundingRect(
            complete_die_hull
        )
        minimum_die_dimension = float(
            max(1, min(die_width, die_height))
        )

        current_die_centre = np.array(
            rectangle[0],
            dtype=np.float32,
        )
        current_die_orientation = self.get_orientation_degrees(
            rectangle
        )
        scene_changed = False

        if self.previous_die_centre is not None:
            centre_change = float(
                np.linalg.norm(
                    current_die_centre
                    - self.previous_die_centre
                )
            )
            angle_change = abs(
                current_die_orientation
                - self.previous_die_orientation
            )
            angle_change = min(
                angle_change,
                180.0 - angle_change,
            )
            scene_changed = (
                centre_change
                > max(12.0, minimum_die_dimension * 0.25)
                or angle_change > 15.0
            )

        if scene_changed:
            self.previous_face_polygon = None
            self.previous_selected_surface_mask = None
            self.face_polygon_missing_frames = 0
            self.angled_face_history.clear()
            self.pip_position_history.clear()
            self.pending_face_number = 0
            self.pending_face_frames = 0
            self.invalid_face_frames = 0
            self.raw_face_number = 0
            self.stable_face_number = 0

        self.previous_die_centre = current_die_centre
        self.previous_die_orientation = current_die_orientation

        # For angled views the learned mask is authoritative. A failed or
        # low confidence segmentation returns unknown instead of falling
        # back to the complete die and admitting pips from side faces.
        if self.top_face_network is not None:
            return self.extract_learned_top_surface(
                image,
                complete_die_mask,
            )

        # Work on a cropped image enlarged four times. The die is only
        # about 60 pixels wide in some recordings, which is too small
        # for reliable surface boundary and morphology operations.
        roi_padding = 5
        roi_x0 = max(0, die_x - roi_padding)
        roi_y0 = max(0, die_y - roi_padding)
        roi_x1 = min(
            image.shape[1],
            die_x + die_width + roi_padding,
        )
        roi_y1 = min(
            image.shape[0],
            die_y + die_height + roi_padding,
        )
        die_crop = image[roi_y0:roi_y1, roi_x0:roi_x1]
        die_mask_crop = complete_die_mask[
            roi_y0:roi_y1,
            roi_x0:roi_x1,
        ]
        processing_scale = 4.0
        large_die = cv2.resize(
            die_crop,
            None,
            fx=processing_scale,
            fy=processing_scale,
            interpolation=cv2.INTER_CUBIC,
        )
        large_die_mask = cv2.resize(
            die_mask_crop,
            (
                large_die.shape[1],
                large_die.shape[0],
            ),
            interpolation=cv2.INTER_NEAREST,
        )

        large_lab = cv2.cvtColor(
            large_die,
            cv2.COLOR_BGR2LAB,
        )
        large_lightness = large_lab[:, :, 0]
        clahe = cv2.createCLAHE(
            clipLimit=2.0,
            tileGridSize=(4, 4),
        )
        enhanced_lightness = clahe.apply(
            large_lightness
        )
        large_die_pixels = enhanced_lightness[
            large_die_mask > 0
        ]

        if large_die_pixels.size == 0:
            return None

        otsu_threshold, _unused = cv2.threshold(
            large_die_pixels.reshape(-1, 1),
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        )
        large_bright_surface = np.zeros_like(
            large_die_mask
        )
        large_bright_surface[
            (large_die_mask > 0)
            & (enhanced_lightness >= otsu_threshold)
        ] = 255

        # The kernels are scaled with the enlarged crop. Closing fills
        # pip holes before the target face contour is measured.
        large_bright_surface = cv2.morphologyEx(
            large_bright_surface,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (17, 17),
            ),
            iterations=2,
        )
        large_bright_surface = cv2.morphologyEx(
            large_bright_surface,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (7, 7),
            ),
        )

        bright_surface_crop = cv2.resize(
            large_bright_surface,
            (die_crop.shape[1], die_crop.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
        bright_surface_mask = np.zeros_like(
            complete_die_mask
        )
        bright_surface_mask[
            roi_y0:roi_y1,
            roi_x0:roi_x1,
        ] = bright_surface_crop
        bright_surface_mask = cv2.bitwise_and(
            bright_surface_mask,
            complete_die_mask,
        )

        # Build a multichannel LAB gradient. Grayscale Canny misses a
        # boundary when adjacent yellow faces have similar brightness but
        # different colour. Scharr combines changes in L, a and b.
        filtered_lab = cv2.bilateralFilter(
            large_lab,
            7,
            30,
            30,
        )
        gradient_squared = np.zeros(
            large_die_mask.shape,
            dtype=np.float32,
        )

        for lab_channel in cv2.split(filtered_lab):
            channel_float = lab_channel.astype(np.float32)
            gradient_x = cv2.Scharr(
                channel_float,
                cv2.CV_32F,
                1,
                0,
            )
            gradient_y = cv2.Scharr(
                channel_float,
                cv2.CV_32F,
                0,
                1,
            )
            gradient_squared += (
                gradient_x * gradient_x
                + gradient_y * gradient_y
            )

        colour_gradient = np.sqrt(gradient_squared)
        die_gradient_values = colour_gradient[
            large_die_mask > 0
        ]

        if die_gradient_values.size == 0:
            return None

        gradient_scale = max(
            float(np.percentile(die_gradient_values, 98.0)),
            1.0,
        )
        colour_gradient_u8 = np.clip(
            colour_gradient * (255.0 / gradient_scale),
            0.0,
            255.0,
        ).astype(np.uint8)
        colour_gradient_u8 = cv2.GaussianBlur(
            colour_gradient_u8,
            (5, 5),
            0,
        )

        _gradient_threshold, large_edges = cv2.threshold(
            colour_gradient_u8,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        )

        # Remove only the narrow outside silhouette. Eroding the whole
        # mask previously removed parts of the desired internal line.
        large_outer_boundary = cv2.morphologyEx(
            large_die_mask,
            cv2.MORPH_GRADIENT,
            cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (5, 5),
            ),
        )
        large_internal_edges = cv2.bitwise_and(
            large_edges,
            large_die_mask,
        )
        large_internal_edges[
            large_outer_boundary > 0
        ] = 0

        # Suppress compact high contrast regions before line fitting. Pip
        # boundaries are compact. A true face separator is long and is not
        # removed by this operation.
        local_lightness = cv2.medianBlur(
            large_lightness,
            21,
        )
        local_difference = cv2.absdiff(
            large_lightness,
            local_lightness,
        )
        _unused, compact_contrast = cv2.threshold(
            local_difference,
            16,
            255,
            cv2.THRESH_BINARY,
        )
        compact_contrast = cv2.bitwise_and(
            compact_contrast,
            large_die_mask,
        )
        compact_contours, _hierarchy = cv2.findContours(
            compact_contrast,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        pip_edge_exclusion = np.zeros_like(
            large_die_mask
        )
        large_minimum_dimension = (
            minimum_die_dimension * processing_scale
        )

        for compact_contour in compact_contours:
            compact_area = cv2.contourArea(compact_contour)
            compact_x, compact_y, compact_width, compact_height = (
                cv2.boundingRect(compact_contour)
            )
            compact_maximum = max(compact_width, compact_height)
            compact_minimum = max(1, min(compact_width, compact_height))
            compact_aspect = compact_maximum / compact_minimum

            if (
                4.0 <= compact_area
                <= 0.045 * large_minimum_dimension ** 2
                and compact_maximum
                <= 0.30 * large_minimum_dimension
                and compact_aspect <= 2.0
            ):
                cv2.drawContours(
                    pip_edge_exclusion,
                    [compact_contour],
                    -1,
                    255,
                    thickness=-1,
                )

        pip_edge_exclusion = cv2.dilate(
            pip_edge_exclusion,
            cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (9, 9),
            ),
            iterations=1,
        )
        large_internal_edges[
            pip_edge_exclusion > 0
        ] = 0

        internal_edge_crop = cv2.resize(
            large_internal_edges,
            (die_crop.shape[1], die_crop.shape[0]),
            interpolation=cv2.INTER_AREA,
        )
        _unused, internal_edge_crop = cv2.threshold(
            internal_edge_crop,
            24,
            255,
            cv2.THRESH_BINARY,
        )
        internal_edges = np.zeros_like(
            complete_die_mask
        )
        internal_edges[
            roi_y0:roi_y1,
            roi_x0:roi_x1,
        ] = internal_edge_crop
        internal_edges = cv2.bitwise_and(
            internal_edges,
            complete_die_mask,
        )
        surface_edges = internal_edges

        line_debug_image = image.copy()
        best_edge_face = None
        best_edge_line = None
        best_edge_score = -1.0

        # LSD proposes straight segments from the colour gradient. Every
        # proposal is validated below using the actual LAB colours on both
        # sides, so a pip arc cannot become a face boundary by itself.
        lsd_input = colour_gradient_u8.copy()
        lsd_input[large_die_mask == 0] = 0
        lsd_input[pip_edge_exclusion > 0] = 0
        line_detector = cv2.createLineSegmentDetector(
            cv2.LSD_REFINE_STD
        )
        lsd_result = line_detector.detect(lsd_input)
        lsd_lines = lsd_result[0]
        line_candidates = []

        if lsd_lines is not None:
            minimum_large_length = (
                minimum_die_dimension
                * processing_scale
                * 0.32
            )

            for large_line in lsd_lines[:, 0]:
                large_x1, large_y1, large_x2, large_y2 = (
                    float(value) for value in large_line
                )
                large_length = math.hypot(
                    large_x2 - large_x1,
                    large_y2 - large_y1,
                )

                if large_length < minimum_large_length:
                    continue

                large_middle_x = int(
                    round(0.5 * (large_x1 + large_x2))
                )
                large_middle_y = int(
                    round(0.5 * (large_y1 + large_y2))
                )

                if not (
                    0 <= large_middle_x < large_die_mask.shape[1]
                    and 0 <= large_middle_y < large_die_mask.shape[0]
                ):
                    continue

                if (
                    large_die_mask[
                        large_middle_y,
                        large_middle_x,
                    ] == 0
                    or large_outer_boundary[
                        large_middle_y,
                        large_middle_x,
                    ] > 0
                    or pip_edge_exclusion[
                        large_middle_y,
                        large_middle_x,
                    ] > 0
                ):
                    continue

                mapped_line = (
                    int(round(large_x1 / processing_scale + roi_x0)),
                    int(round(large_y1 / processing_scale + roi_y0)),
                    int(round(large_x2 / processing_scale + roi_x0)),
                    int(round(large_y2 / processing_scale + roi_y0)),
                )
                line_candidates.append(mapped_line)
                cv2.line(
                    line_debug_image,
                    mapped_line[:2],
                    mapped_line[2:],
                    (0, 165, 255),
                    1,
                    cv2.LINE_AA,
                )

        # Keep the probabilistic Hough transform only as a fallback for
        # frames where LSD cannot produce a usable segment.
        if not line_candidates:
            hough_lines = cv2.HoughLinesP(
                internal_edges,
                1,
                np.pi / 180.0,
                threshold=max(
                    6,
                    int(minimum_die_dimension * 0.12),
                ),
                minLineLength=max(
                    10,
                    int(minimum_die_dimension * 0.25),
                ),
                maxLineGap=max(
                    5,
                    int(minimum_die_dimension * 0.14),
                ),
            )

            if hough_lines is not None:
                line_candidates.extend(
                    tuple(int(value) for value in line)
                    for line in hough_lines[:, 0]
                )

        detected_lines = None

        if line_candidates:
            detected_lines = np.asarray(
                line_candidates,
                dtype=np.int32,
            ).reshape(-1, 1, 4)

        if detected_lines is not None:
            coordinate_y, coordinate_x = np.indices(
                complete_die_mask.shape
            )
            distance_from_silhouette = cv2.distanceTransform(
                complete_die_mask,
                cv2.DIST_L2,
                3,
            )

            for detected_line in detected_lines[:, 0]:
                line_x1, line_y1, line_x2, line_y2 = (
                    int(value) for value in detected_line
                )
                delta_x = float(line_x2 - line_x1)
                delta_y = float(line_y2 - line_y1)
                line_length = math.hypot(delta_x, delta_y)

                if line_length < minimum_die_dimension * 0.32:
                    continue

                middle_x = 0.5 * (line_x1 + line_x2)
                middle_y = 0.5 * (line_y1 + line_y2)

                if not (
                    0 <= int(round(middle_x)) < image.shape[1]
                    and 0 <= int(round(middle_y)) < image.shape[0]
                ):
                    continue

                # A real face boundary passes through the interior.
                if complete_die_mask[
                    int(round(middle_y)),
                    int(round(middle_x)),
                ] == 0:
                    continue

                normal_x = -delta_y / line_length
                normal_y = delta_x / line_length
                sampling_offset = max(
                    3.0,
                    minimum_die_dimension * 0.09,
                )
                sampling_thickness = max(
                    2,
                    int(minimum_die_dimension * 0.07),
                )

                side_masks = []
                side_lab_means = []
                side_variations = []

                for direction in (-1.0, 1.0):
                    offset_x = normal_x * sampling_offset * direction
                    offset_y = normal_y * sampling_offset * direction
                    sample_mask = np.zeros_like(
                        complete_die_mask
                    )
                    cv2.line(
                        sample_mask,
                        (
                            int(round(line_x1 + offset_x)),
                            int(round(line_y1 + offset_y)),
                        ),
                        (
                            int(round(line_x2 + offset_x)),
                            int(round(line_y2 + offset_y)),
                        ),
                        255,
                        sampling_thickness,
                    )
                    sample_mask = cv2.bitwise_and(
                        sample_mask,
                        complete_die_mask,
                    )

                    if cv2.countNonZero(sample_mask) < 8:
                        side_lab_means = []
                        break

                    side_masks.append(sample_mask)
                    lab_pixels = lab_image[
                        sample_mask > 0
                    ].astype(np.float32)
                    lab_mean = np.mean(
                        lab_pixels,
                        axis=0,
                    )
                    side_lab_means.append(lab_mean)
                    side_variations.append(
                        float(
                            np.mean(
                                np.linalg.norm(
                                    lab_pixels - lab_mean,
                                    axis=1,
                                )
                            )
                        )
                    )

                if len(side_lab_means) != 2:
                    continue

                colour_difference = float(
                    np.linalg.norm(
                        side_lab_means[0]
                        - side_lab_means[1]
                    )
                )
                brightness_difference = abs(
                    float(side_lab_means[0][0])
                    - float(side_lab_means[1][0])
                )

                # Require a sustained colour transition with reasonably
                # uniform surface colour along each side. Circular pips and
                # highlights normally produce high along-line variation.
                if colour_difference < 7.0:
                    continue

                if min(side_variations) > 20.0:
                    continue

                line_support_mask = np.zeros_like(
                    complete_die_mask
                )
                cv2.line(
                    line_support_mask,
                    (line_x1, line_y1),
                    (line_x2, line_y2),
                    255,
                    max(2, int(minimum_die_dimension * 0.05)),
                )
                line_support_area = cv2.countNonZero(
                    line_support_mask
                )
                supported_line = cv2.bitwise_and(
                    line_support_mask,
                    internal_edges,
                )
                gradient_support = (
                    cv2.countNonZero(supported_line)
                    / max(line_support_area, 1)
                )

                if gradient_support < 0.18:
                    continue

                endpoint_distances = []

                for endpoint_x, endpoint_y in (
                    (line_x1, line_y1),
                    (line_x2, line_y2),
                ):
                    endpoint_x = int(
                        np.clip(endpoint_x, 0, image.shape[1] - 1)
                    )
                    endpoint_y = int(
                        np.clip(endpoint_y, 0, image.shape[0] - 1)
                    )
                    endpoint_distances.append(
                        float(
                            distance_from_silhouette[
                                endpoint_y,
                                endpoint_x,
                            ]
                        )
                    )

                near_boundary_endpoints = sum(
                    distance
                    <= 0.20 * minimum_die_dimension
                    for distance in endpoint_distances
                )

                if near_boundary_endpoints == 0:
                    continue

                signed_distance = (
                    (coordinate_x - middle_x) * normal_x
                    + (coordinate_y - middle_y) * normal_y
                )
                brighter_direction = (
                    -1.0
                    if side_lab_means[0][0]
                    > side_lab_means[1][0]
                    else 1.0
                )
                half_plane_mask = np.zeros_like(
                    complete_die_mask
                )
                half_plane_mask[
                    signed_distance * brighter_direction >= 0.0
                ] = 255
                edge_face_mask = cv2.bitwise_and(
                    complete_die_mask,
                    half_plane_mask,
                )
                edge_face_mask = cv2.morphologyEx(
                    edge_face_mask,
                    cv2.MORPH_CLOSE,
                    cv2.getStructuringElement(
                        cv2.MORPH_ELLIPSE,
                        (7, 7),
                    ),
                )

                edge_face_contours, _hierarchy = cv2.findContours(
                    edge_face_mask,
                    cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_SIMPLE,
                )

                if not edge_face_contours:
                    continue

                edge_face_contour = max(
                    edge_face_contours,
                    key=cv2.contourArea,
                )
                edge_face_area = cv2.contourArea(
                    edge_face_contour
                )
                complete_area = max(
                    cv2.contourArea(complete_die_hull),
                    1.0,
                )
                edge_area_ratio = edge_face_area / complete_area

                if not 0.22 <= edge_area_ratio <= 0.92:
                    continue

                edge_rectangle = cv2.minAreaRect(
                    edge_face_contour
                )
                _edge_centre, edge_size, _edge_angle = (
                    edge_rectangle
                )
                edge_width, edge_height = edge_size

                if edge_width <= 0.0 or edge_height <= 0.0:
                    continue

                edge_aspect_ratio = max(
                    edge_width,
                    edge_height,
                ) / min(
                    edge_width,
                    edge_height,
                )

                if edge_aspect_ratio > 2.1:
                    continue

                edge_score = (
                    1.2 * line_length / minimum_die_dimension
                    + colour_difference / 30.0
                    + brightness_difference / 60.0
                    + 1.5 * gradient_support
                    + 0.5 * near_boundary_endpoints
                    + 1.0 / edge_aspect_ratio
                    - 0.03 * min(side_variations)
                )

                if edge_score > best_edge_score:
                    best_edge_score = edge_score
                    best_edge_face = (
                        edge_face_contour,
                        edge_rectangle,
                        edge_area_ratio,
                        edge_face_mask.copy(),
                    )
                    best_edge_line = (
                        line_x1,
                        line_y1,
                        line_x2,
                        line_y2,
                    )

        surface_contours, _hierarchy = cv2.findContours(
            bright_surface_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        complete_die_area = max(
            cv2.contourArea(complete_die_hull),
            1.0,
        )
        candidates = []

        for contour in surface_contours:
            area = cv2.contourArea(contour)
            area_ratio = area / complete_die_area

            if area_ratio < 0.28:
                continue

            candidate_rectangle = cv2.minAreaRect(
                contour
            )
            _centre, size, _angle = (
                candidate_rectangle
            )
            width, height = size

            if width <= 0.0 or height <= 0.0:
                continue

            aspect_ratio = max(
                width,
                height,
            ) / min(
                width,
                height,
            )

            if aspect_ratio > 1.9:
                continue

            contour_mask = np.zeros_like(
                complete_die_mask
            )
            cv2.drawContours(
                contour_mask,
                [contour],
                -1,
                255,
                -1,
            )
            mean_lightness = cv2.mean(
                lightness,
                mask=contour_mask,
            )[0]

            squareness_score = 1.0 / aspect_ratio
            selection_score = (
                2.0 * area_ratio
                + squareness_score
                + mean_lightness / 255.0
            )

            candidates.append(
                (
                    selection_score,
                    contour,
                    candidate_rectangle,
                    area_ratio,
                    contour_mask.copy(),
                )
            )

        if best_edge_face is not None:
            extraction_method = "EDGE FACE"
            (
                top_contour,
                top_rectangle,
                top_area_ratio,
                selected_surface_mask,
            ) = best_edge_face
            cv2.line(
                line_debug_image,
                best_edge_line[:2],
                best_edge_line[2:],
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
        elif candidates:
            extraction_method = "BRIGHTNESS FALLBACK"
            (
                _score,
                top_contour,
                top_rectangle,
                top_area_ratio,
                selected_surface_mask,
            ) = max(
                candidates,
                key=lambda candidate: candidate[0],
            )
        else:
            extraction_method = "COMPLETE DIE FALLBACK"
            # If the die has uniform lighting, its complete visible area is top.
            top_contour = complete_die_hull
            top_rectangle = cv2.minAreaRect(
                top_contour
            )
            top_area_ratio = 1.0
            selected_surface_mask = complete_die_mask.copy()

        top_hull = cv2.convexHull(
            top_contour
        )
        top_surface_mask = np.zeros_like(
            complete_die_mask
        )
        cv2.fillConvexPoly(
            top_surface_mask,
            top_hull,
            255,
        )

        # Prefer a perspective aware four corner polygon. Fall back to
        # the existing rotated rectangle when the rounded contour does
        # not provide four reliable corners.
        source_box = None
        top_perimeter = cv2.arcLength(
            top_hull,
            True,
        )

        for approximation_factor in (
            0.02,
            0.03,
            0.04,
            0.05,
            0.06,
            0.08,
        ):
            approximated_polygon = cv2.approxPolyDP(
                top_hull,
                approximation_factor * top_perimeter,
                True,
            )

            if (
                len(approximated_polygon) == 4
                and cv2.isContourConvex(approximated_polygon)
            ):
                polygon_area = cv2.contourArea(
                    approximated_polygon
                )

                if polygon_area >= 0.75 * cv2.contourArea(
                    top_hull
                ):
                    source_box = approximated_polygon.reshape(
                        4,
                        2,
                    ).astype(np.float32)
                    break

        if source_box is None:
            source_box = cv2.boxPoints(
                top_rectangle
            ).astype(np.float32)

        source_box = self.order_box_points(
            source_box
        )

        # Smooth a valid edge polygon across frames. If the current
        # frame loses the weak boundary briefly, reuse the last valid
        # polygon instead of allowing the face region to jump.
        if best_edge_face is not None:
            if self.previous_face_polygon is not None:
                corner_changes = np.linalg.norm(
                    source_box - self.previous_face_polygon,
                    axis=1,
                )

                if float(np.max(corner_changes)) <= max(
                    16.0,
                    minimum_die_dimension * 0.35,
                ):
                    source_box = (
                        0.75 * self.previous_face_polygon
                        + 0.25 * source_box
                    ).astype(np.float32)

            self.previous_face_polygon = source_box.copy()
            self.face_polygon_missing_frames = 0
        elif (
            self.previous_face_polygon is not None
            and self.previous_selected_surface_mask is not None
            and self.face_polygon_missing_frames < 5
        ):
            source_box = self.previous_face_polygon.copy()
            selected_surface_mask = (
                self.previous_selected_surface_mask.copy()
            )
            self.face_polygon_missing_frames += 1
            extraction_method = "TRACKED EDGE FACE"
        else:
            self.previous_face_polygon = None
            self.previous_selected_surface_mask = None
            self.face_polygon_missing_frames = 0

        # The polygon controls perspective geometry. The actual segmented
        # surface controls which pixels can contain pips. Intersecting the
        # two prevents the filled convex polygon from adding a side face
        # back into the valid region.
        polygon_mask = np.zeros_like(
            complete_die_mask
        )
        cv2.fillConvexPoly(
            polygon_mask,
            np.intp(source_box),
            255,
        )
        top_surface_mask = cv2.bitwise_and(
            polygon_mask,
            selected_surface_mask,
        )

        # If the selected physical surface changes sharply, old pip votes
        # must not be carried into the new region.
        if self.previous_selected_surface_mask is not None:
            previous_mask = cv2.bitwise_and(
                polygon_mask,
                self.previous_selected_surface_mask,
            )
            intersection = cv2.countNonZero(
                cv2.bitwise_and(
                    top_surface_mask,
                    previous_mask,
                )
            )
            union = cv2.countNonZero(
                cv2.bitwise_or(
                    top_surface_mask,
                    previous_mask,
                )
            )
            surface_overlap = intersection / max(union, 1)

            if surface_overlap < 0.45:
                self.angled_face_history.clear()
                self.pip_position_history.clear()
                self.pending_face_number = 0
                self.pending_face_frames = 0
                self.raw_face_number = 0
                self.stable_face_number = 0

        self.previous_selected_surface_mask = (
            top_surface_mask.copy()
        )

        cv2.polylines(
            line_debug_image,
            [np.intp(source_box)],
            True,
            (255, 0, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            line_debug_image,
            extraction_method,
            (20, 225),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 0, 0),
            2,
            cv2.LINE_AA,
        )

        output_size = 220
        destination_box = np.array(
            [
                [0.0, 0.0],
                [output_size - 1.0, 0.0],
                [
                    output_size - 1.0,
                    output_size - 1.0,
                ],
                [0.0, output_size - 1.0],
            ],
            dtype=np.float32,
        )

        transform = cv2.getPerspectiveTransform(
            source_box,
            destination_box,
        )
        inverse_transform = cv2.getPerspectiveTransform(
            destination_box,
            source_box,
        )

        rectified_surface = cv2.warpPerspective(
            image,
            transform,
            (output_size, output_size),
        )
        rectified_mask = cv2.warpPerspective(
            top_surface_mask,
            transform,
            (output_size, output_size),
            flags=cv2.INTER_NEAREST,
        )
        rectified_mask = cv2.erode(
            rectified_mask,
            cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (13, 13),
            ),
            iterations=1,
        )

        self.show_window(
            "Adaptive top face mask",
            top_surface_mask,
        )
        self.show_window(
            "Top surface edges",
            internal_edges,
        )
        self.show_window(
            "LAB colour gradient",
            colour_gradient_u8,
        )
        self.show_window(
            "Excluded compact pip edges",
            pip_edge_exclusion,
        )
        self.show_window(
            "Selected internal edge",
            line_debug_image,
        )
        self.show_window(
            "Rectified top face",
            rectified_surface,
        )
        self.show_window(
            "Rectified valid face mask",
            rectified_mask,
        )

        return (
            rectified_surface,
            rectified_mask,
            inverse_transform,
            top_area_ratio,
            top_surface_mask,
            source_box,
        )

    def count_adaptive_surface_pips(
        self,
        image: np.ndarray,
        rectangle,
    ):
        """Counts black or white pips on the adaptively isolated face."""

        extraction = self.extract_adaptive_top_surface(
            image,
            rectangle,
        )

        pip_mask = np.zeros(
            image.shape[:2],
            dtype=np.uint8,
        )

        if extraction is None:
            return 0, [], pip_mask

        (
            rectified_surface,
            rectified_mask,
            inverse_transform,
            _top_area_ratio,
            selected_surface_mask,
            source_box,
        ) = extraction

        grayscale = cv2.cvtColor(
            rectified_surface,
            cv2.COLOR_BGR2GRAY,
        )
        blurred = cv2.GaussianBlur(
            grayscale,
            (5, 5),
            1.2,
        )
        edge_image = cv2.Canny(
            blurred,
            35,
            105,
        )

        mask_points = cv2.findNonZero(
            rectified_mask
        )

        if mask_points is None:
            return 0, [], pip_mask

        _x, _y, face_width, face_height = (
            cv2.boundingRect(mask_points)
        )
        minimum_dimension = float(
            max(1, min(face_width, face_height))
        )

        # Reject circular features close to the selected face boundary.
        distance_from_edge = cv2.distanceTransform(
            rectified_mask,
            cv2.DIST_L2,
            5,
        )

        rectified_hsv = cv2.cvtColor(
            rectified_surface,
            cv2.COLOR_BGR2HSV,
        )
        yellow_h_min = int(
            self.get_parameter(
                "yellow_h_min"
            ).value
        )
        yellow_h_max = int(
            self.get_parameter(
                "yellow_h_max"
            ).value
        )
        rectified_yellow_mask = cv2.inRange(
            rectified_hsv,
            np.array(
                [yellow_h_min, 45, 45],
                dtype=np.uint8,
            ),
            np.array(
                [yellow_h_max, 255, 255],
                dtype=np.uint8,
            ),
        )
        rectified_yellow_mask = cv2.bitwise_and(
            rectified_yellow_mask,
            rectified_mask,
        )

        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=max(
                6,
                int(minimum_dimension * 0.05),
            ),
            param1=55,
            param2=6,
            minRadius=max(
                3,
                int(minimum_dimension * 0.025),
            ),
            maxRadius=max(
                7,
                int(minimum_dimension * 0.14),
            ),
        )

        if circles is None:
            self.angled_face_history.append(0)
            self.angled_face_history = (
                self.angled_face_history[-10:]
            )
            return 0, [], pip_mask

        verified = []
        detected_circles = np.round(
            circles[0]
        ).astype(int)

        blue, green, red = cv2.split(
            rectified_surface
        )
        blue = blue.astype(np.float32)
        green = green.astype(np.float32)
        red = red.astype(np.float32)

        for centre_x, centre_y, radius in (
            detected_circles
        ):
            if not (
                0 <= centre_x < grayscale.shape[1]
                and 0 <= centre_y < grayscale.shape[0]
            ):
                continue

            if rectified_mask[
                centre_y,
                centre_x,
            ] == 0:
                continue

            minimum_clearance = max(
                float(radius) * 1.60,
                minimum_dimension * 0.07,
            )

            if (
                distance_from_edge[
                    centre_y,
                    centre_x,
                ]
                < minimum_clearance
            ):
                continue

            centre_mask = np.zeros_like(
                grayscale
            )
            centre_radius = max(
                2,
                int(radius * 0.70),
            )
            cv2.circle(
                centre_mask,
                (centre_x, centre_y),
                centre_radius,
                255,
                -1,
            )
            centre_mask = cv2.bitwise_and(
                centre_mask,
                rectified_mask,
            )

            ring_mask = np.zeros_like(
                grayscale
            )
            ring_inner = max(
                centre_radius + 1,
                int(radius * 1.20),
            )
            ring_outer = max(
                ring_inner + 2,
                int(radius * 2.00),
            )
            cv2.circle(
                ring_mask,
                (centre_x, centre_y),
                ring_outer,
                255,
                -1,
            )
            cv2.circle(
                ring_mask,
                (centre_x, centre_y),
                ring_inner,
                0,
                -1,
            )

            complete_ring_area = cv2.countNonZero(
                ring_mask
            )
            ring_mask = cv2.bitwise_and(
                ring_mask,
                rectified_mask,
            )
            valid_ring_area = cv2.countNonZero(
                ring_mask
            )

            if (
                complete_ring_area == 0
                or valid_ring_area
                / complete_ring_area
                < 0.85
            ):
                continue

            yellow_ring = cv2.bitwise_and(
                rectified_yellow_mask,
                ring_mask,
            )
            yellow_fraction = (
                cv2.countNonZero(yellow_ring)
                / max(valid_ring_area, 1)
            )

            if yellow_fraction < 0.60:
                continue

            centre_indices = centre_mask > 0
            ring_indices = ring_mask > 0

            if (
                not np.any(centre_indices)
                or not np.any(ring_indices)
            ):
                continue

            centre_red = float(
                np.median(red[centre_indices])
            )
            centre_green = float(
                np.median(green[centre_indices])
            )
            centre_blue = float(
                np.median(blue[centre_indices])
            )
            ring_red = float(
                np.median(red[ring_indices])
            )
            ring_green = float(
                np.median(green[ring_indices])
            )
            ring_blue = float(
                np.median(blue[ring_indices])
            )

            centre_total = (
                centre_red
                + centre_green
                + centre_blue
                + 1.0
            )
            ring_total = (
                ring_red
                + ring_green
                + ring_blue
                + 1.0
            )
            centre_blue_ratio = (
                centre_blue / centre_total
            )
            ring_blue_ratio = (
                ring_blue / ring_total
            )
            blue_ratio_difference = (
                centre_blue_ratio
                - ring_blue_ratio
            )

            centre_brightness = (
                centre_red
                + centre_green
                + centre_blue
            ) / 3.0
            ring_brightness = (
                ring_red
                + ring_green
                + ring_blue
            ) / 3.0
            darkness_difference = (
                ring_brightness
                - centre_brightness
            )

            edge_ring = np.zeros_like(
                grayscale
            )
            cv2.circle(
                edge_ring,
                (centre_x, centre_y),
                int(radius + 2),
                255,
                -1,
            )
            cv2.circle(
                edge_ring,
                (centre_x, centre_y),
                max(1, int(radius - 2)),
                0,
                -1,
            )
            edge_support = (
                cv2.countNonZero(
                    cv2.bitwise_and(
                        edge_image,
                        edge_ring,
                    )
                )
                / max(cv2.countNonZero(edge_ring), 1)
            )

            is_white = (
                blue_ratio_difference >= 0.025
                and centre_brightness
                >= ring_brightness - 35.0
            )
            is_black = darkness_difference >= 30.0

            if (
                not (is_white or is_black)
                or edge_support < 0.025
            ):
                continue

            score = (
                edge_support
                + max(
                    blue_ratio_difference * 5.0,
                    darkness_difference / 255.0,
                )
            )
            verified.append(
                {
                    "x": float(centre_x),
                    "y": float(centre_y),
                    "radius": float(radius),
                    "score": float(score),
                }
            )

        verified.sort(
            key=lambda candidate: candidate["score"],
            reverse=True,
        )

        merged = []
        merge_distance = max(
            14.0,
            minimum_dimension * 0.18,
        )

        for candidate in verified:
            duplicate = False

            for existing in merged:
                distance = math.hypot(
                    candidate["x"] - existing["x"],
                    candidate["y"] - existing["y"],
                )
                overlap_distance = max(
                    merge_distance,
                    0.65
                    * (
                        candidate["radius"]
                        + existing["radius"]
                    ),
                )

                if distance < overlap_distance:
                    duplicate = True
                    break

            if not duplicate:
                merged.append(candidate)

        # Dice pips lie near inset 3-by-3 grid locations, not at
        # the physical corners of the face.
        grid_positions = [
            0.25,
            0.50,
            0.75,
        ]
        grid_candidates = {}

        for candidate in merged:
            normalised_x = (
                candidate["x"] / 219.0
            )
            normalised_y = (
                candidate["y"] / 219.0
            )

            grid_x = min(
                range(3),
                key=lambda index: abs(
                    normalised_x
                    - grid_positions[index]
                ),
            )
            grid_y = min(
                range(3),
                key=lambda index: abs(
                    normalised_y
                    - grid_positions[index]
                ),
            )

            x_error = abs(
                normalised_x
                - grid_positions[grid_x]
            )
            y_error = abs(
                normalised_y
                - grid_positions[grid_y]
            )

            if x_error > 0.17 or y_error > 0.17:
                continue

            cell = (grid_x, grid_y)
            existing = grid_candidates.get(cell)

            if (
                existing is None
                or candidate["score"] > existing["score"]
            ):
                grid_candidates[cell] = candidate

        selected = list(
            grid_candidates.values()
        )

        # Validate every candidate again in the original image. A valid
        # pip must lie inside both the four-corner polygon and the actual
        # segmented top surface, with a small safety distance from the
        # polygon boundary.
        source_contour = source_box.reshape(
            -1,
            1,
            2,
        ).astype(np.float32)
        surface_validated = []

        for candidate in selected:
            rectified_point = np.array(
                [
                    [
                        [candidate["x"], candidate["y"]]
                    ]
                ],
                dtype=np.float32,
            )
            original_point = cv2.perspectiveTransform(
                rectified_point,
                inverse_transform,
            )[0, 0]
            original_x = int(round(original_point[0]))
            original_y = int(round(original_point[1]))

            if not (
                0 <= original_x < selected_surface_mask.shape[1]
                and 0 <= original_y < selected_surface_mask.shape[0]
            ):
                continue

            polygon_distance = cv2.pointPolygonTest(
                source_contour,
                (
                    float(original_point[0]),
                    float(original_point[1]),
                ),
                True,
            )

            if (
                polygon_distance < 2.0
                or selected_surface_mask[
                    original_y,
                    original_x,
                ] == 0
            ):
                continue

            candidate["original_x"] = original_x
            candidate["original_y"] = original_y
            surface_validated.append(candidate)

        selected = surface_validated
        raw_face_number = len(selected)

        # Two detections close together cannot represent face two.
        if raw_face_number == 2:
            first = selected[0]
            second = selected[1]
            separation = math.hypot(
                first["x"] - second["x"],
                first["y"] - second["y"],
            ) / 219.0

            if separation < 0.25:
                raw_face_number = 1

        if not 1 <= raw_face_number <= 6:
            raw_face_number = 0

        self.raw_face_number = raw_face_number

        # Track normalized pip locations rather than only counts.
        current_positions = [
            (
                candidate["x"] / 219.0,
                candidate["y"] / 219.0,
            )
            for candidate in selected
        ]
        self.pip_position_history.append(
            current_positions
        )
        self.pip_position_history = (
            self.pip_position_history[-20:]
        )

        temporal_clusters = []
        temporal_merge_distance = 0.12

        for frame_index, frame_positions in enumerate(
            self.pip_position_history
        ):
            used_clusters = set()

            for position_x, position_y in frame_positions:
                closest_index = None
                closest_distance = None

                for cluster_index, cluster in enumerate(
                    temporal_clusters
                ):
                    if cluster_index in used_clusters:
                        continue

                    distance = math.hypot(
                        position_x - cluster["x"],
                        position_y - cluster["y"],
                    )

                    if (
                        distance <= temporal_merge_distance
                        and (
                            closest_distance is None
                            or distance < closest_distance
                        )
                    ):
                        closest_index = cluster_index
                        closest_distance = distance

                if closest_index is None:
                    temporal_clusters.append(
                        {
                            "x": position_x,
                            "y": position_y,
                            "points": [
                                (position_x, position_y)
                            ],
                            "frames": {frame_index},
                        }
                    )
                    used_clusters.add(
                        len(temporal_clusters) - 1
                    )
                else:
                    cluster = temporal_clusters[
                        closest_index
                    ]
                    cluster["points"].append(
                        (position_x, position_y)
                    )
                    cluster["frames"].add(
                        frame_index
                    )
                    cluster["x"] = float(
                        np.mean(
                            [
                                point[0]
                                for point in cluster["points"]
                            ]
                        )
                    )
                    cluster["y"] = float(
                        np.mean(
                            [
                                point[1]
                                for point in cluster["points"]
                            ]
                        )
                    )
                    used_clusters.add(closest_index)

        persistent_clusters = [
            cluster
            for cluster in temporal_clusters
            if len(cluster["frames"]) >= 8
        ]

        persistent_face_number = len(
            persistent_clusters
        )

        if persistent_face_number == 2:
            first = persistent_clusters[0]
            second = persistent_clusters[1]
            x_separation = abs(
                first["x"] - second["x"]
            )
            y_separation = abs(
                first["y"] - second["y"]
            )

            if (
                x_separation < 0.25
                or y_separation < 0.25
            ):
                persistent_face_number = 0

        if not 1 <= persistent_face_number <= 6:
            persistent_face_number = 0

        if persistent_face_number == 0:
            self.pending_face_number = 0
            self.pending_face_frames = 0
        elif (
            persistent_face_number
            == self.pending_face_number
        ):
            self.pending_face_frames += 1
        else:
            self.pending_face_number = (
                persistent_face_number
            )
            self.pending_face_frames = 1

        if self.pending_face_frames >= 5:
            self.stable_face_number = (
                self.pending_face_number
            )

        stable_face_number = self.stable_face_number

        pip_centres = []

        for candidate in selected:
            pip_x = int(candidate["original_x"])
            pip_y = int(candidate["original_y"])
            pip_centres.append(
                (pip_x, pip_y)
            )
            cv2.circle(
                pip_mask,
                (pip_x, pip_y),
                6,
                255,
                -1,
            )

        return (
            stable_face_number,
            pip_centres,
            pip_mask,
        )

    @staticmethod
    def ellipse_angle_difference(first_angle, second_angle):
        """Returns the unsigned difference between ellipse axes."""

        difference = abs(float(first_angle) - float(second_angle)) % 180.0
        return min(difference, 180.0 - difference)

    def count_shape_based_pips(
        self,
        image: np.ndarray,
        rectangle,
    ):
        """Counts pip groups using colour contrast and ellipse geometry."""

        pip_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        complete_box = cv2.boxPoints(rectangle).astype(np.float32)
        box_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.fillConvexPoly(box_mask, np.intp(complete_box), 255)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        yellow_h_min = int(self.get_parameter("yellow_h_min").value)
        yellow_h_max = int(self.get_parameter("yellow_h_max").value)
        yellow = cv2.inRange(
            hsv,
            np.array([yellow_h_min, 45, 45], dtype=np.uint8),
            np.array([yellow_h_max, 255, 255], dtype=np.uint8),
        )
        yellow = cv2.bitwise_and(yellow, box_mask)
        die_mask = cv2.morphologyEx(
            yellow,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13)),
            iterations=2,
        )
        die_contours, _hierarchy = cv2.findContours(
            die_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        if not die_contours:
            self.invalid_face_frames += 1
            return 0, [], pip_mask

        die_contour = max(die_contours, key=cv2.contourArea)
        die_hull = cv2.convexHull(die_contour)
        die_mask[:] = 0
        cv2.fillConvexPoly(die_mask, die_hull, 255)
        x, y, width, height = cv2.boundingRect(die_hull)
        padding = max(4, int(round(0.12 * max(width, height))))
        x0 = max(0, x - padding)
        y0 = max(0, y - padding)
        x1 = min(image.shape[1], x + width + padding)
        y1 = min(image.shape[0], y + height + padding)
        crop = image[y0:y1, x0:x1]
        crop_die_mask = die_mask[y0:y1, x0:x1]

        if crop.size == 0:
            self.invalid_face_frames += 1
            return 0, [], pip_mask

        scale = 4.0
        large = cv2.resize(
            crop,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_CUBIC,
        )
        large_die_mask = cv2.resize(
            crop_die_mask,
            (large.shape[1], large.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
        large_hsv = cv2.cvtColor(large, cv2.COLOR_BGR2HSV)
        large_lab = cv2.cvtColor(large, cv2.COLOR_BGR2LAB)
        lightness = large_lab[:, :, 0]
        yellow_blue = large_lab[:, :, 2]
        local_lightness = cv2.medianBlur(lightness, 31)
        local_yellow_blue = cv2.medianBlur(yellow_blue, 31)
        dark_difference = cv2.subtract(local_lightness, lightness)
        bright_difference = cv2.subtract(lightness, local_lightness)
        yellow_reduction = cv2.subtract(local_yellow_blue, yellow_blue)
        dark_mask = np.zeros_like(lightness)
        white_mask = np.zeros_like(lightness)
        dark_mask[dark_difference >= 18] = 255
        white_mask[
            (bright_difference >= 9)
            & (yellow_reduction >= 6)
        ] = 255

        # Repair only the white-pip mask. This joins fragmented white regions
        # without enlarging or weakening the black-pip candidates.
        white_mask = cv2.morphologyEx(
            white_mask,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (7, 7),
            ),
            iterations=1,
        )
        white_mask = cv2.morphologyEx(
            white_mask,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (3, 3),
            ),
            iterations=1,
        )
        candidate_mask = cv2.bitwise_or(dark_mask, white_mask)
        safe_die_mask = cv2.erode(
            large_die_mask,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
            iterations=1,
        )
        candidate_mask = cv2.bitwise_and(candidate_mask, safe_die_mask)
        candidate_mask = cv2.morphologyEx(
            candidate_mask,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        )
        candidate_mask = cv2.morphologyEx(
            candidate_mask,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
        )
        self.show_shape_mask_analysis(
            large,
            large_die_mask,
            dark_mask,
            white_mask,
            candidate_mask,
            lightness,
        )
        large_yellow = cv2.inRange(
            large_hsv,
            np.array([yellow_h_min, 35, 35], dtype=np.uint8),
            np.array([yellow_h_max, 255, 255], dtype=np.uint8),
        )
        contours, _hierarchy = cv2.findContours(
            candidate_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        minimum_dimension = float(max(1, min(width, height))) * scale
        die_area = max(cv2.contourArea(die_hull) * scale * scale, 1.0)
        distance_from_boundary = cv2.distanceTransform(
            large_die_mask,
            cv2.DIST_L2,
            5,
        )
        candidates = []
        diagnostic = large.copy()

        for contour in contours:
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)

            if perimeter <= 0.0 or len(contour) < 5:
                continue

            area_ratio = area / die_area

            if not 0.0012 <= area_ratio <= 0.040:
                continue

            ellipse = cv2.fitEllipse(contour)
            (centre_x, centre_y), (axis_a, axis_b), angle = ellipse
            minor_axis = min(axis_a, axis_b)
            major_axis = max(axis_a, axis_b)

            if minor_axis < 3.0 or major_axis <= 0.0:
                continue

            radius = 0.25 * (minor_axis + major_axis)
            axis_ratio = minor_axis / major_axis
            circularity = 4.0 * math.pi * area / (perimeter * perimeter)
            ellipse_area = math.pi * 0.25 * axis_a * axis_b
            fill_ratio = area / max(ellipse_area, 1.0)
            integer_x = int(round(centre_x))
            integer_y = int(round(centre_y))

            if not (
                0 <= integer_x < large.shape[1]
                and 0 <= integer_y < large.shape[0]
            ):
                continue

            if (
                axis_ratio < 0.38
                or circularity < 0.38
                or not 0.45 <= fill_ratio <= 1.40
            ):
                continue

            if distance_from_boundary[integer_y, integer_x] < max(
                1.35 * radius,
                0.035 * minimum_dimension,
            ):
                continue

            centre_mask = np.zeros_like(lightness)
            ring_mask = np.zeros_like(lightness)
            cv2.ellipse(centre_mask, ellipse, 255, thickness=-1)
            outer_radius = max(4, int(round(2.1 * radius)))
            inner_radius = max(2, int(round(1.25 * radius)))
            cv2.circle(
                ring_mask,
                (integer_x, integer_y),
                outer_radius,
                255,
                thickness=-1,
            )
            cv2.circle(
                ring_mask,
                (integer_x, integer_y),
                inner_radius,
                0,
                thickness=-1,
            )
            complete_ring_area = cv2.countNonZero(ring_mask)
            ring_mask = cv2.bitwise_and(ring_mask, large_die_mask)
            valid_ring_area = cv2.countNonZero(ring_mask)

            if (
                complete_ring_area == 0
                or valid_ring_area / complete_ring_area < 0.86
            ):
                continue

            yellow_ring = cv2.bitwise_and(ring_mask, large_yellow)
            yellow_fraction = cv2.countNonZero(yellow_ring) / max(
                valid_ring_area,
                1,
            )

            if yellow_fraction < 0.62:
                continue

            centre_pixels = centre_mask > 0
            ring_pixels = ring_mask > 0

            if not np.any(centre_pixels) or not np.any(ring_pixels):
                continue

            centre_l = float(np.median(lightness[centre_pixels]))
            ring_l = float(np.median(lightness[ring_pixels]))
            centre_b = float(np.median(yellow_blue[centre_pixels]))
            ring_b = float(np.median(yellow_blue[ring_pixels]))
            dark_contrast = ring_l - centre_l
            white_contrast = max(
                centre_l - ring_l,
                ring_b - centre_b,
            )
            pip_type = None
            contrast_score = 0.0

            if dark_contrast >= 14.0:
                pip_type = "black"
                contrast_score = dark_contrast / 60.0
            elif centre_l - ring_l >= 6.0 and ring_b - centre_b >= 4.0:
                pip_type = "white"
                contrast_score = white_contrast / 35.0

            if pip_type is None:
                continue

            shape_score = (
                0.42 * axis_ratio
                + 0.33 * min(circularity, 1.0)
                + 0.15 * min(fill_ratio, 1.0)
                + 0.10 * min(yellow_fraction, 1.0)
            )
            total_score = shape_score + 0.22 * min(contrast_score, 1.5)
            candidates.append(
                {
                    "x": centre_x,
                    "y": centre_y,
                    "radius": radius,
                    "axis_ratio": axis_ratio,
                    "angle": angle,
                    "score": total_score,
                    "type": pip_type,
                    "ellipse": ellipse,
                }
            )
            cv2.ellipse(diagnostic, ellipse, (0, 255, 255), 1)

        # Merge overlapping contours produced by the inner and outer parts
        # of the same white pip.
        merged = []

        for candidate in sorted(
            candidates,
            key=lambda item: item["score"],
            reverse=True,
        ):
            duplicate = False

            for accepted in merged:
                distance = math.hypot(
                    candidate["x"] - accepted["x"],
                    candidate["y"] - accepted["y"],
                )

                if distance < 0.75 * (
                    candidate["radius"] + accepted["radius"]
                ):
                    duplicate = True
                    break

            if not duplicate:
                merged.append(candidate)

        # Cascade: strict candidates identify the colour of their supporting
        # yellow surface. A conservative rounded face region then permits a
        # second, more sensitive search without relaxing detection globally.
        face_search_mask = np.zeros_like(large_die_mask)
        relaxed_candidate_mask = np.zeros_like(large_die_mask)

        if merged:
            primary_seed = max(
                merged,
                key=lambda item: item["score"],
            )
            seed_candidates = []

            for candidate in merged:
                ratio_close = abs(
                    candidate["axis_ratio"]
                    - primary_seed["axis_ratio"]
                ) <= 0.24
                radius_close = (
                    0.55 * primary_seed["radius"]
                    <= candidate["radius"]
                    <= 1.80 * primary_seed["radius"]
                )
                angle_close = True

                if (
                    candidate["axis_ratio"] < 0.80
                    and primary_seed["axis_ratio"] < 0.80
                ):
                    angle_close = self.ellipse_angle_difference(
                        candidate["angle"],
                        primary_seed["angle"],
                    ) <= 30.0

                if ratio_close and radius_close and angle_close:
                    seed_candidates.append(candidate)

            seed_ring_mask = np.zeros_like(large_die_mask)

            for seed in seed_candidates:
                seed_x = int(round(seed["x"]))
                seed_y = int(round(seed["y"]))
                outer_radius = max(
                    6,
                    int(round(2.6 * seed["radius"])),
                )
                inner_radius = max(
                    3,
                    int(round(1.35 * seed["radius"])),
                )
                cv2.circle(
                    seed_ring_mask,
                    (seed_x, seed_y),
                    outer_radius,
                    255,
                    thickness=-1,
                )
                cv2.circle(
                    seed_ring_mask,
                    (seed_x, seed_y),
                    inner_radius,
                    0,
                    thickness=-1,
                )

            seed_ring_mask = cv2.bitwise_and(
                seed_ring_mask,
                large_yellow,
            )
            seed_ring_mask = cv2.bitwise_and(
                seed_ring_mask,
                large_die_mask,
            )
            seed_surface_pixels = large_lab[
                seed_ring_mask > 0
            ].astype(np.float32)

            if seed_surface_pixels.shape[0] >= 12:
                target_lab = np.median(
                    seed_surface_pixels,
                    axis=0,
                )
                lab_distance = np.linalg.norm(
                    large_lab.astype(np.float32) - target_lab,
                    axis=2,
                )
                same_surface_mask = np.zeros_like(
                    large_die_mask
                )
                same_surface_mask[
                    (lab_distance <= 22.0)
                    & (large_die_mask > 0)
                ] = 255
                same_surface_mask = cv2.morphologyEx(
                    same_surface_mask,
                    cv2.MORPH_CLOSE,
                    cv2.getStructuringElement(
                        cv2.MORPH_ELLIPSE,
                        (17, 17),
                    ),
                    iterations=2,
                )
                same_surface_mask = cv2.morphologyEx(
                    same_surface_mask,
                    cv2.MORPH_OPEN,
                    cv2.getStructuringElement(
                        cv2.MORPH_ELLIPSE,
                        (5, 5),
                    ),
                )
                component_count, component_labels, component_stats, _centres = (
                    cv2.connectedComponentsWithStats(
                        same_surface_mask,
                        connectivity=8,
                    )
                )
                best_label = 0
                best_component_score = -1.0

                for label in range(1, component_count):
                    component = np.zeros_like(
                        large_die_mask
                    )
                    component[component_labels == label] = 255
                    seed_overlap = cv2.countNonZero(
                        cv2.bitwise_and(
                            component,
                            seed_ring_mask,
                        )
                    )
                    component_area = float(
                        component_stats[
                            label,
                            cv2.CC_STAT_AREA,
                        ]
                    )
                    area_ratio = component_area / max(
                        cv2.countNonZero(large_die_mask),
                        1,
                    )

                    if not 0.12 <= area_ratio <= 0.95:
                        continue

                    component_score = (
                        5.0 * seed_overlap
                        + min(component_area, die_area)
                        / max(die_area, 1.0)
                    )

                    if component_score > best_component_score:
                        best_component_score = component_score
                        best_label = label

                if best_label > 0:
                    selected_component = np.zeros_like(
                        large_die_mask
                    )
                    selected_component[
                        component_labels == best_label
                    ] = 255
                    surface_contours, _hierarchy = cv2.findContours(
                        selected_component,
                        cv2.RETR_EXTERNAL,
                        cv2.CHAIN_APPROX_SIMPLE,
                    )

                    if surface_contours:
                        surface_contour = max(
                            surface_contours,
                            key=cv2.contourArea,
                        )
                        rectangle_mask = np.zeros_like(
                            large_die_mask
                        )
                        surface_rectangle = cv2.minAreaRect(
                            surface_contour
                        )
                        rectangle_box = cv2.boxPoints(
                            surface_rectangle
                        )
                        cv2.fillConvexPoly(
                            rectangle_mask,
                            np.intp(rectangle_box),
                            255,
                        )
                        rounded_mask = rectangle_mask.copy()

                        if len(surface_contour) >= 5:
                            surface_ellipse = cv2.fitEllipse(
                                surface_contour
                            )
                            ellipse_centre, ellipse_axes, ellipse_angle = (
                                surface_ellipse
                            )
                            expanded_ellipse = (
                                ellipse_centre,
                                (
                                    1.08 * ellipse_axes[0],
                                    1.08 * ellipse_axes[1],
                                ),
                                ellipse_angle,
                            )
                            ellipse_mask = np.zeros_like(
                                large_die_mask
                            )
                            cv2.ellipse(
                                ellipse_mask,
                                expanded_ellipse,
                                255,
                                thickness=-1,
                            )
                            rounded_mask = cv2.bitwise_and(
                                rectangle_mask,
                                ellipse_mask,
                            )

                        face_search_mask = cv2.bitwise_and(
                            rounded_mask,
                            selected_component,
                        )
                        face_search_mask = cv2.morphologyEx(
                            face_search_mask,
                            cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(
                                cv2.MORPH_ELLIPSE,
                                (13, 13),
                            ),
                            iterations=1,
                        )
                        face_search_mask = cv2.erode(
                            face_search_mask,
                            cv2.getStructuringElement(
                                cv2.MORPH_ELLIPSE,
                                (5, 5),
                            ),
                            iterations=1,
                        )

                        relaxed_dark = np.zeros_like(lightness)
                        relaxed_white = np.zeros_like(lightness)
                        relaxed_dark[dark_difference >= 10] = 255
                        relaxed_white[
                            (bright_difference >= 4)
                            & (yellow_reduction >= 3)
                        ] = 255
                        relaxed_candidate_mask = cv2.bitwise_or(
                            relaxed_dark,
                            relaxed_white,
                        )
                        relaxed_candidate_mask = cv2.bitwise_and(
                            relaxed_candidate_mask,
                            face_search_mask,
                        )
                        relaxed_candidate_mask = cv2.morphologyEx(
                            relaxed_candidate_mask,
                            cv2.MORPH_OPEN,
                            cv2.getStructuringElement(
                                cv2.MORPH_ELLIPSE,
                                (3, 3),
                            ),
                        )
                        relaxed_candidate_mask = cv2.morphologyEx(
                            relaxed_candidate_mask,
                            cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(
                                cv2.MORPH_ELLIPSE,
                                (5, 5),
                            ),
                        )
                        relaxed_contours, _hierarchy = cv2.findContours(
                            relaxed_candidate_mask,
                            cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE,
                        )
                        reference_radius = float(
                            np.median(
                                [
                                    seed["radius"]
                                    for seed in seed_candidates
                                ]
                            )
                        )
                        reference_ratio = float(
                            np.median(
                                [
                                    seed["axis_ratio"]
                                    for seed in seed_candidates
                                ]
                            )
                        )

                        for relaxed_contour in relaxed_contours:
                            relaxed_area = cv2.contourArea(
                                relaxed_contour
                            )
                            relaxed_perimeter = cv2.arcLength(
                                relaxed_contour,
                                True,
                            )

                            if (
                                len(relaxed_contour) < 5
                                or relaxed_perimeter <= 0.0
                            ):
                                continue

                            relaxed_area_ratio = relaxed_area / die_area

                            if not 0.0007 <= relaxed_area_ratio <= 0.050:
                                continue

                            relaxed_ellipse = cv2.fitEllipse(
                                relaxed_contour
                            )
                            (
                                relaxed_centre_x,
                                relaxed_centre_y,
                            ), (
                                relaxed_axis_a,
                                relaxed_axis_b,
                            ), relaxed_angle = relaxed_ellipse
                            relaxed_minor = min(
                                relaxed_axis_a,
                                relaxed_axis_b,
                            )
                            relaxed_major = max(
                                relaxed_axis_a,
                                relaxed_axis_b,
                            )

                            if relaxed_minor < 2.5 or relaxed_major <= 0.0:
                                continue

                            relaxed_radius = 0.25 * (
                                relaxed_minor + relaxed_major
                            )
                            relaxed_ratio = relaxed_minor / relaxed_major
                            relaxed_circularity = (
                                4.0
                                * math.pi
                                * relaxed_area
                                / (
                                    relaxed_perimeter
                                    * relaxed_perimeter
                                )
                            )

                            if (
                                relaxed_ratio < 0.25
                                or relaxed_circularity < 0.24
                                or not (
                                    0.45 * reference_radius
                                    <= relaxed_radius
                                    <= 1.90 * reference_radius
                                )
                                or abs(
                                    relaxed_ratio - reference_ratio
                                ) > 0.36
                            ):
                                continue

                            relaxed_x = int(round(relaxed_centre_x))
                            relaxed_y = int(round(relaxed_centre_y))

                            if not (
                                0 <= relaxed_x < large.shape[1]
                                and 0 <= relaxed_y < large.shape[0]
                                and face_search_mask[
                                    relaxed_y,
                                    relaxed_x,
                                ] > 0
                            ):
                                continue

                            ellipse_pixels = np.zeros_like(
                                large_die_mask
                            )
                            cv2.ellipse(
                                ellipse_pixels,
                                relaxed_ellipse,
                                255,
                                thickness=-1,
                            )
                            ellipse_area_pixels = cv2.countNonZero(
                                ellipse_pixels
                            )
                            ellipse_inside = cv2.countNonZero(
                                cv2.bitwise_and(
                                    ellipse_pixels,
                                    face_search_mask,
                                )
                            )

                            if ellipse_inside / max(
                                ellipse_area_pixels,
                                1,
                            ) < 0.82:
                                continue

                            ring_mask = np.zeros_like(
                                large_die_mask
                            )
                            relaxed_outer = max(
                                5,
                                int(round(2.0 * relaxed_radius)),
                            )
                            relaxed_inner = max(
                                2,
                                int(round(1.20 * relaxed_radius)),
                            )
                            cv2.circle(
                                ring_mask,
                                (relaxed_x, relaxed_y),
                                relaxed_outer,
                                255,
                                thickness=-1,
                            )
                            cv2.circle(
                                ring_mask,
                                (relaxed_x, relaxed_y),
                                relaxed_inner,
                                0,
                                thickness=-1,
                            )
                            complete_ring = cv2.countNonZero(
                                ring_mask
                            )
                            ring_mask = cv2.bitwise_and(
                                ring_mask,
                                face_search_mask,
                            )
                            valid_ring = cv2.countNonZero(ring_mask)

                            if (
                                complete_ring == 0
                                or valid_ring / complete_ring < 0.72
                            ):
                                continue

                            ring_yellow_fraction = cv2.countNonZero(
                                cv2.bitwise_and(
                                    ring_mask,
                                    large_yellow,
                                )
                            ) / max(valid_ring, 1)

                            if ring_yellow_fraction < 0.42:
                                continue

                            centre_pixels = ellipse_pixels > 0
                            ring_pixels = ring_mask > 0
                            relaxed_centre_l = float(
                                np.median(lightness[centre_pixels])
                            )
                            relaxed_ring_l = float(
                                np.median(lightness[ring_pixels])
                            )
                            relaxed_centre_b = float(
                                np.median(yellow_blue[centre_pixels])
                            )
                            relaxed_ring_b = float(
                                np.median(yellow_blue[ring_pixels])
                            )
                            relaxed_dark_contrast = (
                                relaxed_ring_l - relaxed_centre_l
                            )
                            relaxed_white_l = (
                                relaxed_centre_l - relaxed_ring_l
                            )
                            relaxed_white_b = (
                                relaxed_ring_b - relaxed_centre_b
                            )

                            if relaxed_dark_contrast >= 9.0:
                                relaxed_type = "black"
                                relaxed_contrast = (
                                    relaxed_dark_contrast / 60.0
                                )
                            elif (
                                relaxed_white_l >= 3.0
                                and relaxed_white_b >= 2.0
                            ):
                                relaxed_type = "white"
                                relaxed_contrast = max(
                                    relaxed_white_l,
                                    relaxed_white_b,
                                ) / 35.0
                            else:
                                continue

                            relaxed_candidate = {
                                "x": relaxed_centre_x,
                                "y": relaxed_centre_y,
                                "radius": relaxed_radius,
                                "axis_ratio": relaxed_ratio,
                                "angle": relaxed_angle,
                                "score": (
                                    0.32 * relaxed_ratio
                                    + 0.28
                                    * min(relaxed_circularity, 1.0)
                                    + 0.18
                                    * min(ring_yellow_fraction, 1.0)
                                    + 0.16
                                    * min(relaxed_contrast, 1.5)
                                ),
                                "type": relaxed_type,
                                "ellipse": relaxed_ellipse,
                            }
                            duplicate = False

                            for accepted in merged:
                                centre_distance = math.hypot(
                                    relaxed_candidate["x"]
                                    - accepted["x"],
                                    relaxed_candidate["y"]
                                    - accepted["y"],
                                )

                                if centre_distance < 0.72 * (
                                    relaxed_candidate["radius"]
                                    + accepted["radius"]
                                ):
                                    duplicate = True
                                    break

                            if not duplicate:
                                merged.append(relaxed_candidate)
                                cv2.ellipse(
                                    diagnostic,
                                    relaxed_ellipse,
                                    (0, 128, 255),
                                    1,
                                )

        # Group pips that share similar ellipse deformation. Near circular
        # ellipses do not have a reliable major axis angle, so angle is
        # ignored for those candidates.
        groups = []

        for candidate in merged:
            chosen_group = None

            for group in groups:
                reference_ratio = float(
                    np.median([item["axis_ratio"] for item in group])
                )
                reference_radius = float(
                    np.median([item["radius"] for item in group])
                )
                ratio_close = abs(
                    candidate["axis_ratio"] - reference_ratio
                ) <= 0.18
                radius_close = (
                    0.58 * reference_radius
                    <= candidate["radius"]
                    <= 1.72 * reference_radius
                )
                angle_close = True

                if candidate["axis_ratio"] < 0.82 and reference_ratio < 0.82:
                    reference_angle = float(
                        np.median([item["angle"] for item in group])
                    )
                    angle_close = self.ellipse_angle_difference(
                        candidate["angle"],
                        reference_angle,
                    ) <= 25.0

                if ratio_close and radius_close and angle_close:
                    chosen_group = group
                    break

            if chosen_group is None:
                groups.append([candidate])
            else:
                chosen_group.append(candidate)

        valid_groups = [group for group in groups if 1 <= len(group) <= 6]
        selected = []

        if valid_groups:
            def group_score(group):
                mean_score = float(np.mean([item["score"] for item in group]))
                mean_ratio = float(
                    np.mean([item["axis_ratio"] for item in group])
                )
                type_consistency = max(
                    sum(item["type"] == "black" for item in group),
                    sum(item["type"] == "white" for item in group),
                ) / len(group)
                return (
                    mean_score
                    + 0.30 * mean_ratio
                    + 0.12 * type_consistency
                    + 0.025 * len(group)
                )

            selected = max(valid_groups, key=group_score)

        # Conservative recovery for a weak second white pip. This runs only
        # after the normal shape detector has found exactly one white pip.
        self.latest_white_recovery_used = False

        if (
            len(selected) == 1
            and selected[0]["type"] == "white"
        ):
            reference = selected[0]
            recovery_region = cv2.erode(
                large_die_mask,
                cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE,
                    (9, 9),
                ),
                iterations=1,
            )
            reference_ring = np.zeros_like(lightness)
            reference_x = int(round(reference["x"]))
            reference_y = int(round(reference["y"]))
            reference_outer = max(
                5,
                int(round(2.2 * reference["radius"])),
            )
            reference_inner = max(
                2,
                int(round(1.25 * reference["radius"])),
            )
            cv2.circle(
                reference_ring,
                (reference_x, reference_y),
                reference_outer,
                255,
                thickness=-1,
            )
            cv2.circle(
                reference_ring,
                (reference_x, reference_y),
                reference_inner,
                0,
                thickness=-1,
            )
            reference_ring = cv2.bitwise_and(
                reference_ring,
                recovery_region,
            )
            reference_ring_pixels = reference_ring > 0

            if np.any(reference_ring_pixels):
                reference_surface_lab = np.median(
                    large_lab[reference_ring_pixels].astype(np.float32),
                    axis=0,
                )
            else:
                reference_surface_lab = None

            recovery_mask = np.zeros_like(lightness)
            recovery_mask[
                (bright_difference >= 1)
                & (yellow_reduction >= 1)
            ] = 255
            recovery_mask = cv2.bitwise_and(
                recovery_mask,
                recovery_region,
            )
            recovery_mask = cv2.morphologyEx(
                recovery_mask,
                cv2.MORPH_CLOSE,
                cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE,
                    (5, 5),
                ),
            )
            recovery_mask = cv2.morphologyEx(
                recovery_mask,
                cv2.MORPH_OPEN,
                cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE,
                    (3, 3),
                ),
            )
            recovery_contours, _hierarchy = cv2.findContours(
                recovery_mask,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )
            _surface_x, _surface_y, surface_w, surface_h = cv2.boundingRect(
                recovery_region
            )
            surface_distance = cv2.distanceTransform(
                recovery_region,
                cv2.DIST_L2,
                5,
            )
            recovery_candidates = []

            for recovery_contour in recovery_contours:
                area = cv2.contourArea(recovery_contour)
                perimeter = cv2.arcLength(recovery_contour, True)

                if len(recovery_contour) < 5 or perimeter <= 0.0:
                    continue

                if not 0.00045 <= area / die_area <= 0.045:
                    continue

                ellipse = cv2.fitEllipse(recovery_contour)
                (candidate_x, candidate_y), axes, angle = ellipse
                minor_axis = min(axes)
                major_axis = max(axes)

                if minor_axis < 2.2 or major_axis <= 0.0:
                    continue

                radius = 0.25 * (minor_axis + major_axis)
                axis_ratio = minor_axis / major_axis
                circularity = (
                    4.0 * math.pi * area / (perimeter * perimeter)
                )
                ellipse_area = math.pi * 0.25 * axes[0] * axes[1]
                fill_ratio = area / max(ellipse_area, 1.0)

                if (
                    axis_ratio < 0.40
                    or circularity < 0.24
                    or not 0.32 <= fill_ratio <= 1.40
                ):
                    continue

                if not (
                    0.55 * reference["radius"]
                    <= radius
                    <= 1.70 * reference["radius"]
                ):
                    continue

                integer_x = int(round(candidate_x))
                integer_y = int(round(candidate_y))

                if not (
                    0 <= integer_x < large.shape[1]
                    and 0 <= integer_y < large.shape[0]
                    and recovery_region[integer_y, integer_x] > 0
                ):
                    continue

                if surface_distance[integer_y, integer_x] < 1.15 * radius:
                    continue

                separation = math.hypot(
                    candidate_x - reference["x"],
                    candidate_y - reference["y"],
                )
                minimum_separation = 1.35 * (
                    radius + reference["radius"]
                )
                maximum_separation = 0.72 * max(
                    1.0,
                    min(surface_w, surface_h),
                )

                if not minimum_separation <= separation <= maximum_separation:
                    continue

                ellipse_mask = np.zeros_like(lightness)
                cv2.ellipse(ellipse_mask, ellipse, 255, thickness=-1)
                ellipse_area_pixels = cv2.countNonZero(ellipse_mask)
                ellipse_inside = cv2.countNonZero(
                    cv2.bitwise_and(ellipse_mask, recovery_region)
                )

                if ellipse_inside / max(ellipse_area_pixels, 1) < 0.92:
                    continue

                ring_mask = np.zeros_like(lightness)
                outer_radius = max(5, int(round(2.2 * radius)))
                inner_radius = max(2, int(round(1.25 * radius)))
                cv2.circle(
                    ring_mask,
                    (integer_x, integer_y),
                    outer_radius,
                    255,
                    thickness=-1,
                )
                cv2.circle(
                    ring_mask,
                    (integer_x, integer_y),
                    inner_radius,
                    0,
                    thickness=-1,
                )
                ring_mask = cv2.bitwise_and(
                    ring_mask,
                    recovery_region,
                )
                valid_ring = cv2.countNonZero(ring_mask)

                if valid_ring == 0:
                    continue

                yellow_fraction = cv2.countNonZero(
                    cv2.bitwise_and(ring_mask, large_yellow)
                ) / valid_ring

                if yellow_fraction < 0.58:
                    continue

                if reference_surface_lab is not None:
                    candidate_surface_lab = np.median(
                        large_lab[ring_mask > 0].astype(np.float32),
                        axis=0,
                    )
                    surface_colour_distance = float(
                        np.linalg.norm(
                            candidate_surface_lab - reference_surface_lab
                        )
                    )

                    if surface_colour_distance > 18.0:
                        continue

                centre_pixels = ellipse_mask > 0
                ring_pixels = ring_mask > 0
                centre_l = float(np.median(lightness[centre_pixels]))
                ring_l = float(np.median(lightness[ring_pixels]))
                centre_b = float(np.median(yellow_blue[centre_pixels]))
                ring_b = float(np.median(yellow_blue[ring_pixels]))
                white_l = centre_l - ring_l
                white_b = ring_b - centre_b

                if white_l < 2.0 or white_b < 2.0:
                    continue

                score = (
                    0.28 * axis_ratio
                    + 0.22 * min(circularity, 1.0)
                    + 0.18 * min(fill_ratio, 1.0)
                    + 0.18 * min(yellow_fraction, 1.0)
                    + 0.14 * min((white_l + white_b) / 18.0, 1.0)
                )
                recovery_candidates.append(
                    {
                        "x": candidate_x,
                        "y": candidate_y,
                        "radius": radius,
                        "axis_ratio": axis_ratio,
                        "angle": angle,
                        "score": score,
                        "type": "white",
                        "ellipse": ellipse,
                    }
                )

            if recovery_candidates:
                selected.append(
                    max(
                        recovery_candidates,
                        key=lambda item: item["score"],
                    )
                )
                self.latest_white_recovery_used = True

        # A white-only face-2 branch. It cannot affect black-pip faces. The
        # pair must resemble two pips on opposite diagonal parts of one face.
        white_pair_valid = False
        self.latest_white_face_confidence = 0.0

        if (
            len(selected) == 2
            and all(candidate["type"] == "white" for candidate in selected)
        ):
            first, second = selected
            first_radius = max(float(first["radius"]), 1.0)
            second_radius = max(float(second["radius"]), 1.0)
            size_similarity = (
                min(first_radius, second_radius)
                / max(first_radius, second_radius)
            )
            delta_x = float(second["x"] - first["x"])
            delta_y = float(second["y"] - first["y"])
            separation = math.hypot(delta_x, delta_y)
            normalized_separation = separation / max(
                minimum_dimension,
                1.0,
            )
            diagonal_balance = (
                min(abs(delta_x), abs(delta_y))
                / max(abs(delta_x), abs(delta_y), 1.0)
            )
            mean_pair_score = float(
                np.mean(
                    [
                        float(first["score"]),
                        float(second["score"]),
                    ]
                )
            )
            separation_score = float(
                np.clip(
                    1.0
                    - abs(normalized_separation - 0.45) / 0.35,
                    0.0,
                    1.0,
                )
            )
            self.latest_white_face_confidence = float(
                np.clip(
                    0.45 * min(mean_pair_score / 1.10, 1.0)
                    + 0.25 * size_similarity
                    + 0.20 * diagonal_balance
                    + 0.10 * separation_score,
                    0.0,
                    1.0,
                )
            )
            white_pair_valid = (
                size_similarity >= 0.45
                and 0.18 <= normalized_separation <= 0.80
                and diagonal_balance >= 0.25
                and self.latest_white_face_confidence >= 0.68
            )

        self.white_face_history.append(
            2 if white_pair_valid else 0
        )
        self.white_face_history = self.white_face_history[-5:]
        white_face_votes = sum(
            detected_face == 2
            for detected_face in self.white_face_history
        )

        raw_face_number = len(selected)

        if not 1 <= raw_face_number <= 6:
            raw_face_number = 0

        if raw_face_number:
            mean_candidate_score = float(
                np.mean([candidate["score"] for candidate in selected])
            )
            self.latest_shape_confidence = float(
                np.clip(0.35 + 0.55 * mean_candidate_score, 0.0, 0.96)
            )

            if self.latest_white_recovery_used:
                self.latest_shape_confidence = max(
                    0.0,
                    self.latest_shape_confidence - 0.08,
                )
        else:
            self.latest_shape_confidence = 0.0

        if raw_face_number == 0:
            self.invalid_face_frames += 1

            if self.invalid_face_frames >= 3:
                self.stable_face_number = 0
                self.pending_face_number = 0
                self.pending_face_frames = 0
        else:
            self.invalid_face_frames = 0

            if raw_face_number == self.pending_face_number:
                self.pending_face_frames += 1
            else:
                self.pending_face_number = raw_face_number
                self.pending_face_frames = 1

            if self.pending_face_frames >= 3:
                self.stable_face_number = raw_face_number

        # Face 2 does not need three perfectly consecutive detections. A
        # validated white pair in three of the last five frames is sufficient.
        if white_face_votes >= 3:
            self.stable_face_number = 2
            self.pending_face_number = 2
            self.pending_face_frames = max(
                self.pending_face_frames,
                3,
            )
            self.latest_shape_confidence = max(
                self.latest_shape_confidence,
                self.latest_white_face_confidence,
            )
            self.latest_white_recovery_used = True

        self.raw_face_number = raw_face_number
        pip_centres = []

        for candidate in selected:
            original_x = int(round(candidate["x"] / scale + x0))
            original_y = int(round(candidate["y"] / scale + y0))
            pip_centres.append((original_x, original_y))
            cv2.circle(
                pip_mask,
                (original_x, original_y),
                6,
                255,
                thickness=-1,
            )
            cv2.ellipse(
                diagnostic,
                candidate["ellipse"],
                (255, 0, 255),
                2,
            )

        cv2.putText(
            diagnostic,
            f"raw={raw_face_number} stable={self.stable_face_number}",
            (8, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            diagnostic,
            (
                f"white pair votes={white_face_votes}/5 "
                f"conf={self.latest_white_face_confidence:.2f}"
            ),
            (8, 46),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (255, 0, 255),
            1,
            cv2.LINE_AA,
        )
        self.show_window("Shape candidate mask", candidate_mask)
        self.show_window("Cascaded face search mask", face_search_mask)
        self.show_window("Relaxed candidate mask", relaxed_candidate_mask)
        self.show_window("Shape based pip groups", diagnostic)
        return self.stable_face_number, pip_centres, pip_mask

    def orientation_box_hypothesis(
        self,
        image: np.ndarray,
        rectangle,
    ):
        """Votes across conservative rotated boxes around the die."""

        centre, size, _rectangle_angle = rectangle
        width, height = float(size[0]), float(size[1])
        orientation = self.get_orientation_degrees(rectangle)

        if width < height:
            width, height = height, width

        radians = math.radians(orientation)
        axis_x = np.array(
            [math.cos(radians), math.sin(radians)],
            dtype=np.float32,
        )
        axis_y = np.array(
            [-math.sin(radians), math.cos(radians)],
            dtype=np.float32,
        )
        centre_array = np.array(centre, dtype=np.float32)
        scale_candidates = (
            (0.82, 0.82),
            (0.94, 0.82),
            (0.82, 0.94),
        )
        offset_candidates = (
            (0.0, 0.0),
            (0.10, 0.0),
            (-0.10, 0.0),
            (0.0, 0.10),
            (0.0, -0.10),
        )
        results = []

        for scale_x, scale_y in scale_candidates:
            for offset_x, offset_y in offset_candidates:
                adjusted_centre = (
                    centre_array
                    + offset_x * width * axis_x
                    + offset_y * height * axis_y
                )
                adjusted_rectangle = (
                    (
                        float(adjusted_centre[0]),
                        float(adjusted_centre[1]),
                    ),
                    (
                        max(8.0, width * scale_x),
                        max(8.0, height * scale_y),
                    ),
                    float(orientation),
                )
                result = self.count_pips_threshold(
                    image,
                    adjusted_rectangle,
                )
                face_number = int(result[0])

                if 1 <= face_number <= 6:
                    results.append(
                        {
                            "face": face_number,
                            "centres": result[1],
                            "mask": result[2],
                            "rectangle": adjusted_rectangle,
                        }
                    )

        empty_mask = np.zeros(image.shape[:2], dtype=np.uint8)

        if not results:
            return {
                "method": "geometry",
                "face": 0,
                "confidence": 0.0,
                "centres": [],
                "mask": empty_mask,
            }

        counts = Counter(result["face"] for result in results)
        face_number, occurrences = counts.most_common(1)[0]
        matching = [
            result
            for result in results
            if result["face"] == face_number
        ]
        consistency = occurrences / max(len(results), 1)
        proposal_coverage = occurrences / float(
            len(scale_candidates) * len(offset_candidates)
        )
        confidence = min(
            0.88,
            0.28
            + 0.48 * consistency
            + 0.35 * proposal_coverage,
        )
        representative = matching[len(matching) // 2]
        geometry_debug = image.copy()
        box = cv2.boxPoints(
            representative["rectangle"]
        ).astype(np.int32)
        cv2.polylines(
            geometry_debug,
            [box],
            True,
            (255, 180, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            geometry_debug,
            f"geometry face={face_number} confidence={confidence:.2f}",
            (20, 225),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 180, 0),
            2,
            cv2.LINE_AA,
        )
        self.show_window("Orientation box proposal", geometry_debug)
        return {
            "method": "geometry",
            "face": int(face_number),
            "confidence": float(confidence),
            "centres": representative["centres"],
            "mask": representative["mask"],
        }

    def count_ensemble_pips(
        self,
        image: np.ndarray,
        rectangle,
    ):
        """Fuses edge, shape and orientation box hypotheses."""

        hypotheses = []
        empty_mask = np.zeros(image.shape[:2], dtype=np.uint8)

        edge_result = self.count_adaptive_surface_pips(
            image,
            rectangle,
        )
        edge_raw = int(self.raw_face_number)

        if 1 <= edge_raw <= 6:
            edge_confidence = 0.58

            if len(edge_result[1]) == edge_raw:
                edge_confidence += 0.16

            hypotheses.append(
                {
                    "method": "edge",
                    "face": edge_raw,
                    "confidence": min(edge_confidence, 0.86),
                    "centres": edge_result[1],
                    "mask": edge_result[2],
                }
            )

        shape_result = self.count_shape_based_pips(
            image,
            rectangle,
        )
        shape_raw = int(self.raw_face_number)

        if 1 <= shape_raw <= 6:
            shape_confidence = 0.62

            if len(shape_result[1]) == shape_raw:
                shape_confidence += 0.16

            hypotheses.append(
                {
                    "method": "shape",
                    "face": shape_raw,
                    "confidence": min(shape_confidence, 0.90),
                    "centres": shape_result[1],
                    "mask": shape_result[2],
                }
            )

        geometry_hypothesis = self.orientation_box_hypothesis(
            image,
            rectangle,
        )

        if 1 <= geometry_hypothesis["face"] <= 6:
            hypotheses.append(geometry_hypothesis)

        method_weights = {
            "edge": 0.90,
            "shape": 0.85,
            "geometry": 0.60,
        }
        face_scores = {face: 0.0 for face in range(1, 7)}
        face_support = {face: 0 for face in range(1, 7)}

        for hypothesis in hypotheses:
            face_number = hypothesis["face"]
            face_scores[face_number] += (
                method_weights[hypothesis["method"]]
                * hypothesis["confidence"]
            )
            face_support[face_number] += 1

        for face_number in range(1, 7):
            if face_support[face_number] >= 2:
                face_scores[face_number] += 0.20

            if face_support[face_number] >= 3:
                face_scores[face_number] += 0.18

        ranking = sorted(
            face_scores.items(),
            key=lambda item: item[1],
            reverse=True,
        )
        best_face, best_score = ranking[0]
        _second_face, second_score = ranking[1]
        score_margin = best_score - second_score
        raw_face_number = 0

        if (
            best_score >= 0.54
            and score_margin >= 0.16
            and face_support[best_face] >= 1
        ):
            raw_face_number = int(best_face)

        winner = None

        if raw_face_number != 0:
            matching_hypotheses = [
                hypothesis
                for hypothesis in hypotheses
                if hypothesis["face"] == raw_face_number
            ]
            winner = max(
                matching_hypotheses,
                key=lambda hypothesis: (
                    method_weights[hypothesis["method"]]
                    * hypothesis["confidence"]
                ),
            )

        self.latest_model_results = []

        for method in ("edge", "shape", "geometry"):
            hypothesis = next(
                (
                    item
                    for item in hypotheses
                    if item["method"] == method
                ),
                None,
            )
            self.latest_model_results.append(
                {
                    "method": method,
                    "face": (
                        int(hypothesis["face"])
                        if hypothesis is not None
                        else 0
                    ),
                    "confidence": (
                        float(hypothesis["confidence"])
                        if hypothesis is not None
                        else 0.0
                    ),
                }
            )
        self.latest_selected_model = (
            winner["method"] if winner is not None else "abstain"
        )
        self.latest_ensemble_score = float(best_score)
        self.latest_ensemble_margin = float(score_margin)

        if raw_face_number == 0:
            self.ensemble_invalid_frames += 1
            self.ensemble_pending_face = 0
            self.ensemble_pending_frames = 0

            if self.ensemble_invalid_frames >= 3:
                self.ensemble_stable_face = 0
        else:
            self.ensemble_invalid_frames = 0

            if raw_face_number == self.ensemble_pending_face:
                self.ensemble_pending_frames += 1
            else:
                self.ensemble_pending_face = raw_face_number
                self.ensemble_pending_frames = 1

            required_frames = 4

            if (
                face_support[raw_face_number] >= 2
                and best_score >= 1.15
            ):
                required_frames = 2
            elif best_score >= 0.78:
                required_frames = 3

            if self.ensemble_pending_frames >= required_frames:
                self.ensemble_stable_face = raw_face_number

        self.raw_face_number = raw_face_number
        self.stable_face_number = self.ensemble_stable_face
        confidence_display = np.full(
            (245, 620, 3),
            28,
            dtype=np.uint8,
        )
        display_y = 30

        for method in ("edge", "shape", "geometry"):
            hypothesis = next(
                (
                    item
                    for item in hypotheses
                    if item["method"] == method
                ),
                None,
            )

            if hypothesis is None:
                display_text = f"{method}: abstain"
                colour = (150, 150, 150)
            else:
                display_text = (
                    f"{method}: face {hypothesis['face']}  "
                    f"confidence {hypothesis['confidence']:.2f}"
                )
                colour = (
                    (0, 255, 0)
                    if hypothesis["face"] == raw_face_number
                    else (0, 180, 255)
                )

            cv2.putText(
                confidence_display,
                display_text,
                (16, display_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                colour,
                2,
                cv2.LINE_AA,
            )
            display_y += 45

        cv2.putText(
            confidence_display,
            (
                f"winner raw={raw_face_number} stable="
                f"{self.ensemble_stable_face} score={best_score:.2f} "
                f"margin={score_margin:.2f}"
            ),
            (16, display_y + 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (255, 0, 255),
            2,
            cv2.LINE_AA,
        )
        self.show_window("Ensemble confidence", confidence_display)

        if winner is None:
            return self.ensemble_stable_face, [], empty_mask

        return (
            self.ensemble_stable_face,
            winner["centres"],
            winner["mask"],
        )

    def detect_bright_pips(
        self,
        image: np.ndarray,
        rectangle,
    ):
        """Returns a reliable bright-only hypothesis for an angled die."""

        empty_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        complete_box = cv2.boxPoints(rectangle).astype(np.float32)
        box_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.fillConvexPoly(box_mask, np.intp(complete_box), 255)

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        yellow_h_min = int(self.get_parameter("yellow_h_min").value)
        yellow_h_max = int(self.get_parameter("yellow_h_max").value)
        yellow = cv2.inRange(
            hsv,
            np.array([yellow_h_min, 45, 45], dtype=np.uint8),
            np.array([yellow_h_max, 255, 255], dtype=np.uint8),
        )
        yellow = cv2.bitwise_and(yellow, box_mask)
        die_mask = cv2.morphologyEx(
            yellow,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13)),
            iterations=2,
        )
        die_contours, _hierarchy = cv2.findContours(
            die_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        if not die_contours:
            self.bright_face_history.append(0)
            self.bright_face_history = self.bright_face_history[-5:]
            return {
                "reliable": False,
                "face": 0,
                "confidence": 0.0,
                "centres": [],
                "mask": empty_mask,
                "votes": 0,
            }

        die_hull = cv2.convexHull(max(die_contours, key=cv2.contourArea))
        die_mask[:] = 0
        cv2.fillConvexPoly(die_mask, die_hull, 255)
        x, y, width, height = cv2.boundingRect(die_hull)
        padding = max(4, int(round(0.12 * max(width, height))))
        x0 = max(0, x - padding)
        y0 = max(0, y - padding)
        x1 = min(image.shape[1], x + width + padding)
        y1 = min(image.shape[0], y + height + padding)
        crop = image[y0:y1, x0:x1]
        crop_die_mask = die_mask[y0:y1, x0:x1]

        if crop.size == 0:
            self.bright_face_history.append(0)
            self.bright_face_history = self.bright_face_history[-5:]
            return {
                "reliable": False,
                "face": 0,
                "confidence": 0.0,
                "centres": [],
                "mask": empty_mask,
                "votes": 0,
            }

        scale = 4.0
        large = cv2.resize(
            crop,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_CUBIC,
        )
        large_die_mask = cv2.resize(
            crop_die_mask,
            (large.shape[1], large.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
        large_hsv = cv2.cvtColor(large, cv2.COLOR_BGR2HSV)
        large_lab = cv2.cvtColor(large, cv2.COLOR_BGR2LAB)
        lightness = large_lab[:, :, 0]
        yellow_blue = large_lab[:, :, 2]
        local_lightness = cv2.medianBlur(lightness, 31)
        local_yellow_blue = cv2.medianBlur(yellow_blue, 31)
        bright_difference = cv2.subtract(lightness, local_lightness)
        yellow_reduction = cv2.subtract(local_yellow_blue, yellow_blue)

        bright_mask = np.zeros_like(lightness)
        bright_mask[
            (bright_difference >= 9)
            & (yellow_reduction >= 6)
        ] = 255
        bright_mask = cv2.morphologyEx(
            bright_mask,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
            iterations=1,
        )
        bright_mask = cv2.morphologyEx(
            bright_mask,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
            iterations=1,
        )
        safe_die_mask = cv2.erode(
            large_die_mask,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
            iterations=1,
        )
        bright_mask = cv2.bitwise_and(bright_mask, safe_die_mask)
        large_yellow = cv2.inRange(
            large_hsv,
            np.array([yellow_h_min, 35, 35], dtype=np.uint8),
            np.array([yellow_h_max, 255, 255], dtype=np.uint8),
        )
        contours, _hierarchy = cv2.findContours(
            bright_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        die_area = max(cv2.contourArea(die_hull) * scale * scale, 1.0)
        minimum_dimension = float(max(1, min(width, height))) * scale
        boundary_distance = cv2.distanceTransform(
            large_die_mask,
            cv2.DIST_L2,
            5,
        )
        candidates = []

        for contour in contours:
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)

            if len(contour) < 5 or perimeter <= 0.0:
                continue

            area_ratio = area / die_area

            if not 0.0006 <= area_ratio <= 0.045:
                continue

            ellipse = cv2.fitEllipse(contour)
            (centre_x, centre_y), axes, ellipse_angle = ellipse
            minor_axis = min(axes)
            major_axis = max(axes)

            if minor_axis < 2.5 or major_axis <= 0.0:
                continue

            radius = 0.25 * (minor_axis + major_axis)
            axis_ratio = minor_axis / major_axis
            circularity = 4.0 * math.pi * area / (perimeter * perimeter)
            ellipse_area = math.pi * 0.25 * axes[0] * axes[1]
            fill_ratio = area / max(ellipse_area, 1.0)
            integer_x = int(round(centre_x))
            integer_y = int(round(centre_y))

            if not (
                0 <= integer_x < large.shape[1]
                and 0 <= integer_y < large.shape[0]
            ):
                continue

            if (
                axis_ratio < 0.28
                or circularity < 0.22
                or not 0.30 <= fill_ratio <= 1.50
            ):
                continue

            if boundary_distance[integer_y, integer_x] < max(
                radius,
                0.025 * minimum_dimension,
            ):
                continue

            centre_mask = np.zeros_like(lightness)
            ring_mask = np.zeros_like(lightness)
            cv2.ellipse(centre_mask, ellipse, 255, thickness=-1)

            ellipse_pixel_area = cv2.countNonZero(centre_mask)
            ellipse_inside_area = cv2.countNonZero(
                cv2.bitwise_and(centre_mask, large_die_mask)
            )
            ellipse_inside_fraction = (
                ellipse_inside_area
                / max(ellipse_pixel_area, 1)
            )

            if ellipse_inside_fraction < 0.88:
                continue

            cv2.circle(
                ring_mask,
                (integer_x, integer_y),
                max(5, int(round(2.2 * radius))),
                255,
                thickness=-1,
            )
            cv2.circle(
                ring_mask,
                (integer_x, integer_y),
                max(2, int(round(1.25 * radius))),
                0,
                thickness=-1,
            )
            complete_ring_area = cv2.countNonZero(ring_mask)
            ring_mask = cv2.bitwise_and(ring_mask, large_die_mask)
            valid_ring_area = cv2.countNonZero(ring_mask)

            if complete_ring_area == 0 or valid_ring_area == 0:
                continue

            ring_coverage = valid_ring_area / complete_ring_area

            if ring_coverage < 0.72:
                continue

            yellow_fraction = cv2.countNonZero(
                cv2.bitwise_and(ring_mask, large_yellow)
            ) / valid_ring_area

            if yellow_fraction < 0.50:
                continue

            centre_pixels = centre_mask > 0
            ring_pixels = ring_mask > 0

            if not np.any(centre_pixels) or not np.any(ring_pixels):
                continue

            centre_l = float(np.median(lightness[centre_pixels]))
            ring_l = float(np.median(lightness[ring_pixels]))
            centre_b = float(np.median(yellow_blue[centre_pixels]))
            ring_b = float(np.median(yellow_blue[ring_pixels]))
            white_l = centre_l - ring_l
            white_b = ring_b - centre_b

            if white_l < 3.0 or white_b < 2.0:
                continue

            axis_score = float(np.clip((axis_ratio - 0.20) / 0.70, 0.0, 1.0))
            circularity_score = float(
                np.clip((circularity - 0.18) / 0.72, 0.0, 1.0)
            )
            contrast_score = float(
                np.clip((white_l + white_b) / 18.0, 0.0, 1.0)
            )
            clearance_ratio = (
                boundary_distance[integer_y, integer_x]
                / max(radius, 1.0)
            )
            boundary_score = float(
                np.clip(
                    (clearance_ratio - 1.0) / 1.5,
                    0.0,
                    1.0,
                )
            )
            score = (
                0.25 * axis_score
                + 0.20 * circularity_score
                + 0.15 * min(fill_ratio, 1.0)
                + 0.15 * min(yellow_fraction, 1.0)
                + 0.10 * contrast_score
                + 0.10 * ring_coverage
                + 0.05 * boundary_score
            )
            candidates.append(
                {
                    "x": centre_x,
                    "y": centre_y,
                    "radius": radius,
                    "score": score,
                    "ellipse": ellipse,
                    "angle": ellipse_angle,
                    "ring_coverage": ring_coverage,
                    "boundary_score": boundary_score,
                    "ellipse_inside_fraction": ellipse_inside_fraction,
                }
            )

        merged = []

        for candidate in sorted(
            candidates,
            key=lambda item: item["score"],
            reverse=True,
        ):
            duplicate = any(
                math.hypot(
                    candidate["x"] - accepted["x"],
                    candidate["y"] - accepted["y"],
                )
                < 0.72 * (candidate["radius"] + accepted["radius"])
                for accepted in merged
            )

            if not duplicate:
                merged.append(candidate)

        merged = merged[:6]

        # A true face 2 can occasionally acquire one weak bright fragment at
        # the die boundary. When three candidates exist, test all possible
        # pairs and remove the third only when it is demonstrably weaker in
        # boundary support than a valid diagonal pair.
        if len(merged) == 3:
            pair_hypotheses = []

            for first, second in combinations(merged, 2):
                delta_x = float(second["x"] - first["x"])
                delta_y = float(second["y"] - first["y"])
                separation = math.hypot(delta_x, delta_y)
                normalized_separation = separation / max(
                    minimum_dimension,
                    1.0,
                )
                size_similarity = (
                    min(first["radius"], second["radius"])
                    / max(first["radius"], second["radius"], 1.0)
                )
                diagonal_balance = (
                    min(abs(delta_x), abs(delta_y))
                    / max(abs(delta_x), abs(delta_y), 1.0)
                )
                separation_score = float(
                    np.clip(
                        1.0
                        - abs(normalized_separation - 0.45) / 0.35,
                        0.0,
                        1.0,
                    )
                )
                mean_candidate_score = 0.5 * (
                    first["score"] + second["score"]
                )
                pair_boundary_score = 0.5 * (
                    first["boundary_score"]
                    + second["boundary_score"]
                )
                pair_score = (
                    0.30 * mean_candidate_score
                    + 0.20 * size_similarity
                    + 0.20 * diagonal_balance
                    + 0.15 * separation_score
                    + 0.15 * pair_boundary_score
                )
                valid_geometry = (
                    size_similarity >= 0.45
                    and 0.18 <= normalized_separation <= 0.80
                    and diagonal_balance >= 0.25
                )

                if valid_geometry:
                    pair_hypotheses.append(
                        (pair_score, first, second)
                    )

            if pair_hypotheses:
                best_pair = max(
                    pair_hypotheses,
                    key=lambda hypothesis: hypothesis[0],
                )
                third_candidate = next(
                    candidate
                    for candidate in merged
                    if candidate is not best_pair[1]
                    and candidate is not best_pair[2]
                )
                pair_mean_score = 0.5 * (
                    best_pair[1]["score"]
                    + best_pair[2]["score"]
                )
                pair_mean_boundary = 0.5 * (
                    best_pair[1]["boundary_score"]
                    + best_pair[2]["boundary_score"]
                )
                pair_mean_ring = 0.5 * (
                    best_pair[1]["ring_coverage"]
                    + best_pair[2]["ring_coverage"]
                )
                third_is_weaker = (
                    third_candidate["score"] < 0.88 * pair_mean_score
                    or third_candidate["boundary_score"]
                    < pair_mean_boundary - 0.12
                    or third_candidate["ring_coverage"]
                    < pair_mean_ring - 0.08
                )

                if best_pair[0] >= 0.62 and third_is_weaker:
                    merged = [best_pair[1], best_pair[2]]

        face_number = len(merged)
        confidence = 0.0

        if 1 <= face_number <= 6:
            confidence = float(
                np.clip(
                    np.mean([candidate["score"] for candidate in merged]),
                    0.0,
                    1.0,
                )
            )

        if face_number == 2:
            first, second = merged
            delta_x = float(second["x"] - first["x"])
            delta_y = float(second["y"] - first["y"])
            separation = math.hypot(delta_x, delta_y)
            normalized_separation = separation / max(minimum_dimension, 1.0)
            size_similarity = (
                min(first["radius"], second["radius"])
                / max(first["radius"], second["radius"], 1.0)
            )
            diagonal_balance = (
                min(abs(delta_x), abs(delta_y))
                / max(abs(delta_x), abs(delta_y), 1.0)
            )
            valid_pair = (
                size_similarity >= 0.45
                and 0.18 <= normalized_separation <= 0.80
                and diagonal_balance >= 0.25
            )

            if valid_pair:
                confidence = float(
                    np.clip(
                        confidence
                        + 0.05 * size_similarity
                        + 0.05 * diagonal_balance,
                        0.0,
                        1.0,
                    )
                )
            else:
                face_number = 0
                confidence = 0.0

        history_value = (
            face_number
            if 1 <= face_number <= 6 and confidence >= 0.65
            else 0
        )
        self.bright_face_history.append(history_value)
        self.bright_face_history = self.bright_face_history[-5:]
        votes = sum(
            historical_face == face_number
            for historical_face in self.bright_face_history
            if face_number != 0
        )
        required_votes = 3

        if self.stable_face_number == 2 and face_number != 2:
            required_votes = 5

        reliable = (
            face_number != 0
            and confidence >= 0.70
            and votes >= required_votes
        )
        pip_centres = [
            (
                int(round(candidate["x"] / scale + x0)),
                int(round(candidate["y"] / scale + y0)),
            )
            for candidate in merged
        ]
        result_mask = np.zeros(image.shape[:2], dtype=np.uint8)

        for pip_centre in pip_centres:
            cv2.circle(result_mask, pip_centre, 6, 255, thickness=-1)

        bright_diagnostic = large.copy()

        for candidate in merged:
            cv2.ellipse(
                bright_diagnostic,
                candidate["ellipse"],
                (255, 0, 255),
                2,
            )

        cv2.putText(
            bright_diagnostic,
            (
                f"bright face={face_number} conf={confidence:.2f} "
                f"votes={votes}/{required_votes} reliable={reliable}"
            ),
            (8, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (255, 0, 255),
            1,
            cv2.LINE_AA,
        )
        self.show_window("Bright-only candidates", bright_diagnostic)

        return {
            "reliable": reliable,
            "face": face_number,
            "confidence": confidence,
            "centres": pip_centres,
            "mask": result_mask,
            "votes": votes,
        }

    def count_pips(
        self,
        image: np.ndarray,
        rectangle,
    ):
        """Selects the original or adaptive detector from orientation."""

        orientation_degrees = (
            self.get_orientation_degrees(
                rectangle
            )
        )

        if abs(orientation_degrees) <= 5.0:
            result = self.count_pips_threshold(
                image,
                rectangle,
            )

            self.raw_face_number = result[0]
            self.stable_face_number = result[0]
            self.angled_face_history.clear()
            self.pip_position_history.clear()
            self.pending_face_number = 0
            self.pending_face_frames = 0
            self.invalid_face_frames = 0
            self.ensemble_pending_face = 0
            self.ensemble_pending_frames = 0
            self.ensemble_invalid_frames = 0
            self.ensemble_stable_face = 0
            self.previous_face_polygon = None
            self.previous_selected_surface_mask = None
            self.previous_die_centre = None
            self.previous_die_orientation = None
            self.face_polygon_missing_frames = 0
            self.bright_face_history.clear()
            threshold_confidence = (
                1.0 if 1 <= int(result[0]) <= 6 else 0.0
            )
            self.latest_model_results = [
                {
                    "method": "threshold",
                    "face": int(result[0]),
                    "confidence": threshold_confidence,
                }
            ]
            self.latest_selected_model = "threshold"
            self.latest_ensemble_score = threshold_confidence
            self.latest_ensemble_margin = threshold_confidence

            return result

        bright_result = self.detect_bright_pips(
            image,
            rectangle,
        )

        if bright_result["reliable"]:
            bright_face = int(bright_result["face"])
            bright_confidence = float(bright_result["confidence"])
            self.raw_face_number = bright_face
            self.stable_face_number = bright_face
            self.latest_white_recovery_used = False
            self.latest_model_results = [
                {
                    "method": "bright-only",
                    "face": bright_face,
                    "confidence": bright_confidence,
                }
            ]
            self.latest_selected_model = "bright-only"
            self.latest_ensemble_score = bright_confidence
            self.latest_ensemble_margin = bright_confidence
            return (
                bright_face,
                bright_result["centres"],
                bright_result["mask"],
            )

        result = self.count_shape_based_pips(
            image,
            rectangle,
        )
        shape_confidence = self.latest_shape_confidence

        selected_name = (
            "shape + white recovery"
            if self.latest_white_recovery_used
            else "shape"
        )
        self.latest_model_results = [
            {
                "method": selected_name,
                "face": int(self.raw_face_number),
                "confidence": shape_confidence,
            }
        ]
        self.latest_selected_model = selected_name
        self.latest_ensemble_score = shape_confidence
        self.latest_ensemble_margin = shape_confidence
        return result

    def create_pose(
        self,
        position,
        orientation_degrees: float,
        header,
    ) -> PoseStamped:
        """Creates a PoseStamped detection result."""

        pose = PoseStamped()
        pose.header.stamp = header.stamp
        pose.header.frame_id = (
            self.camera_frame
            or header.frame_id
        )

        pose.pose.position.x = position[0]
        pose.pose.position.y = position[1]
        pose.pose.position.z = position[2]

        orientation_radians = math.radians(
            orientation_degrees
        )
        half_angle = (
            orientation_radians / 2.0
        )

        pose.pose.orientation.z = math.sin(
            half_angle
        )
        pose.pose.orientation.w = math.cos(
            half_angle
        )

        return pose

    def _pose_to_world(self, pose: PoseStamped):
        """
        Transform a detection from ``pose.header.frame_id`` (camera optical
        frame) into ``self.world_frame``, via the calibrated camera
        extrinsic already published as TF (see the "hand-eye" static
        transform in ``camera_calibration_cellN.launch.py``).

        Position is a plain point transform (always valid, whatever the
        camera's mounting angle). Orientation is NOT the raw composed
        quaternion -- see the comment above ``world_yaw`` below for why
        that would be wrong for a top-down-ish camera mount, and how this
        rebuilds a clean yaw-only-about-world-Z quaternion instead, which
        is what ``dice_manipulation_node``'s whole "dice_tf's +Z points
        up" convention (``GRASP_DOWN_QUAT``) actually needs.

        Returns ``None`` (logging once) if the transform is not available
        yet -- e.g. the camera calibration static publisher or the camera
        driver itself is not up yet -- rather than blocking this
        per-frame image callback.
        """
        try:
            transform = self.tf_buffer.lookup_transform(
                self.world_frame,
                pose.header.frame_id,
                rclpy.time.Time(),
            )
        except Exception:  # noqa: BLE001 -- TF not ready yet
            if not self._tf_warned:
                self.get_logger().warning(
                    f"No TF from '{pose.header.frame_id}' to "
                    f"'{self.world_frame}' yet (camera calibration static "
                    f"transform / camera driver not up?); "
                    f"/dice_identification will report failure until it is."
                )
                self._tf_warned = True
            return None
        self._tf_warned = False

        t = transform.transform.translation
        q = (
            transform.transform.rotation.x,
            transform.transform.rotation.y,
            transform.transform.rotation.z,
            transform.transform.rotation.w,
        )

        position = (
            pose.pose.position.x,
            pose.pose.position.y,
            pose.pose.position.z,
        )
        rotated_position = _rotate_vector(position, q)

        orientation = (
            pose.pose.orientation.x,
            pose.pose.orientation.y,
            pose.pose.orientation.z,
            pose.pose.orientation.w,
        )

        # create_pose() only ever encodes a rotation about the camera's
        # OWN local Z (its optical/looking axis) -- a rotation about Z
        # leaves that same Z axis unchanged, so this detection's local +Z
        # *is* the camera's +Z. For a top-down-ish mount (this node
        # ray-casts onto a horizontal "board" plane, so the camera looks
        # down at the table) the camera's +Z (forward/depth, standard
        # optical-frame convention) points roughly *into* the table, i.e.
        # opposite of what dice_tf needs (+Z pointing up, away from the
        # top face -- see dice_manipulation_node's GRASP_DOWN_QUAT).
        # Composing the raw quaternion through the extrinsic would carry
        # that wrong Z straight into world and flip every grasp upside
        # down. Fix: only the in-plane box angle is ever physically
        # meaningful here (the die's own +Z is world +Z *by construction*
        # -- it is resting flat, "up" is not something the camera needs
        # to tell us). So rotate the LOCAL X axis (the box's own in-plane
        # reference direction) all the way into world through the real
        # calibrated rotation, drop whatever Z component that picks up
        # (the mount's tilt, irrelevant once projected), and rebuild a
        # clean yaw-only-about-world-Z quaternion from what is left.
        world_local_x = _rotate_vector(
            (1.0, 0.0, 0.0), quaternion_multiply(q, orientation))
        world_yaw = math.atan2(world_local_x[1], world_local_x[0])
        half_world_yaw = world_yaw / 2.0
        world_orientation = (
            0.0, 0.0, math.sin(half_world_yaw), math.cos(half_world_yaw))

        world_pose = PoseStamped()
        world_pose.header.stamp = pose.header.stamp
        world_pose.header.frame_id = self.world_frame
        world_pose.pose.position.x = rotated_position[0] + t.x
        world_pose.pose.position.y = rotated_position[1] + t.y
        world_pose.pose.position.z = rotated_position[2] + t.z
        (
            world_pose.pose.orientation.x,
            world_pose.pose.orientation.y,
            world_pose.pose.orientation.z,
            world_pose.pose.orientation.w,
        ) = world_orientation

        return world_pose

    def _broadcast_dice_tf(self, world_pose: PoseStamped) -> None:
        """
        Publish ``dice_tf_frame`` live, from an already world-frame pose.

        Kept for RViz/debugging and anything else that looks up
        ``dice_tf_frame`` live; ``dice_manipulation_node`` itself no
        longer resolves its actual grasp poses against this broadcast --
        it grasps from the same world-frame pose carried in the
        ``/dice_identification`` response, read once and reused for both
        the approach and the grasp move (see that module's
        ``pick_dice()``/``grasp_orientation()`` and the module docstring,
        "Why the grasp yaw is picked before grasping" and "never a fresh
        TF lookup mid-sequence"). The service response is what actually
        makes grasping work; this broadcast does not need to be perfectly
        jitter-free for that anymore.
        """
        if not self.publish_dice_tf:
            return

        transform = TransformStamped()
        transform.header.stamp = self.get_clock().now().to_msg()
        transform.header.frame_id = self.world_frame
        transform.child_frame_id = self.dice_tf_frame
        transform.transform.translation.x = world_pose.pose.position.x
        transform.transform.translation.y = world_pose.pose.position.y
        transform.transform.translation.z = world_pose.pose.position.z
        transform.transform.rotation = world_pose.pose.orientation
        self.tf_broadcaster.sendTransform(transform)

    def publish_images(
        self,
        debug_image: np.ndarray,
        yellow_mask: np.ndarray,
        header,
    ) -> None:
        """Publishes debug results as ROS images."""

        debug_message = self.bridge.cv2_to_imgmsg(
            debug_image,
            encoding="bgr8",
        )
        debug_message.header = header

        mask_message = self.bridge.cv2_to_imgmsg(
            yellow_mask,
            encoding="mono8",
        )
        mask_message.header = header

        self.debug_publisher.publish(
            debug_message
        )
        self.mask_publisher.publish(
            mask_message
        )

    def image_callback(
        self,
        message: CompressedImage,
    ) -> None:
        """Runs the complete vision pipeline."""

        try:
            image = (
                self.bridge.compressed_imgmsg_to_cv2(
                    message,
                    desired_encoding="bgr8",
                )
            )

            debug_image = image.copy()
            yellow_mask = self.create_yellow_mask(
                image
            )

            self.frame_count += 1
            self.latest_face = None
            self.latest_pose = None

            detection = self.find_die_rectangle(
                yellow_mask
            )

            if detection is None:
                cv2.putText(
                    debug_image,
                    "Dice not detected",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )

                self.publish_images(
                    debug_image,
                    yellow_mask,
                    message.header,
                )

                self.show_window(
                    "Dice detection",
                    debug_image,
                )
                self.show_window(
                    "Yellow mask",
                    yellow_mask,
                )
                cv2.waitKey(1)
                return

            (
                _area,
                _contour,
                rectangle,
                box,
                centre,
            ) = detection

            centre_x = float(centre[0])
            centre_y = float(centre[1])

            cv2.drawContours(
                debug_image,
                [box],
                0,
                (255, 0, 0),
                3,
            )

            cv2.circle(
                debug_image,
                (
                    int(centre_x),
                    int(centre_y),
                ),
                6,
                (0, 0, 255),
                -1,
            )

            orientation_degrees = (
                self.get_orientation_degrees(
                    rectangle
                )
            )

            orientation_radians = math.radians(
                orientation_degrees
            )

            arrow_length = 60

            arrow_end = (
                int(
                    centre_x
                    + arrow_length
                    * math.cos(
                        orientation_radians
                    )
                ),
                int(
                    centre_y
                    + arrow_length
                    * math.sin(
                        orientation_radians
                    )
                ),
            )

            cv2.arrowedLine(
                debug_image,
                (
                    int(centre_x),
                    int(centre_y),
                ),
                arrow_end,
                (0, 0, 255),
                3,
            )

            (
                face_number,
                pip_centres,
                pip_mask,
            ) = self.count_pips(
                image,
                rectangle,
            )

            for pip_centre in pip_centres:
                cv2.circle(
                    debug_image,
                    pip_centre,
                    8,
                    (255, 0, 255),
                    2,
                )

            position = self.calculate_position(
                centre_x,
                centre_y,
            )

            if position is not None:
                pose = self.create_pose(
                    position,
                    orientation_degrees,
                    message.header,
                )

                # Published as-is (camera frame) for debugging/RViz
                # continuity -- unrelated to the world-frame pose used
                # below for /dice_identification and dice_tf.
                self.pose_publisher.publish(pose)

                if 1 <= face_number <= 6:
                    world_pose = self._pose_to_world(pose)

                    if world_pose is not None:
                        self.latest_face = face_number
                        self.latest_pose = world_pose
                        self._broadcast_dice_tf(world_pose)

            # Publish confidence before the final face. The receiver uses the
            # face message as the trigger to print one combined result.
            confidence_message = Float32()
            confidence_message.data = float(self.latest_ensemble_score)
            self.confidence_publisher.publish(confidence_message)

            if 1 <= face_number <= 6:
                face_message = Int32()
                face_message.data = int(face_number)
                self.face_publisher.publish(face_message)

                cv2.putText(
                    debug_image,
                    (
                        f"Position: "
                        f"x={position[0]:.3f}, "
                        f"y={position[1]:.3f}, "
                        f"z={position[2]:.3f}"
                    ),
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )
            else:
                cv2.putText(
                    debug_image,
                    "Waiting for camera_info",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )

            cv2.putText(
                debug_image,
                (
                    f"Orientation: "
                    f"{orientation_degrees:.1f} deg"
                ),
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.putText(
                debug_image,
                f"Raw face: {self.raw_face_number}",
                (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 0, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.putText(
                debug_image,
                f"Stable face: {face_number}",
                (20, 160),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255, 0, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.putText(
                debug_image,
                f"Frame: {self.frame_count}",
                (20, 200),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

            self.draw_model_summary(debug_image)

            self.publish_images(
                debug_image,
                yellow_mask,
                message.header,
            )

            self.show_window(
                "Dice detection",
                debug_image,
            )
            self.show_window(
                "Yellow mask",
                yellow_mask,
            )
            self.show_window(
                "Pip mask",
                pip_mask,
            )

            cv2.waitKey(1)

        except Exception as exception:
            self.get_logger().error(
                f"Detection failed: {exception}"
            )

    def identification_callback(
        self,
        _request,
        response,
    ):
        """Returns the latest valid detection."""

        if (
            self.latest_face is None
            or self.latest_pose is None
        ):
            response.success = False
            return response

        response.face_number = (
            self.latest_face
        )
        response.pose = self.latest_pose
        response.success = True

        return response


def main() -> None:
    rclpy.init()

    node = DiceDetectorNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        cv2.destroyAllWindows()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()


