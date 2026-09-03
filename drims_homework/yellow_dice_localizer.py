# Copyright 2026 DRIMS3 Summer School
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Locate a yellow die and publish its metric pose in the camera frame.

The node segments the yellow die, fits a square to its contour, and estimates
the square pose with ``cv2.solvePnP`` using the intrinsics in ``CameraInfo``.
The position is therefore expressed in metres in the optical camera frame
(x right, y down, z forward), rather than in image pixels.
"""

from typing import Optional, Tuple

import cv2
import numpy as np

import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import PointStamped, PoseStamped
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from sensor_msgs.msg import CameraInfo, CompressedImage, Image
from std_msgs.msg import Float64, Int32


def order_corners(corners: np.ndarray) -> np.ndarray:
    """Return four image corners in a consistent circular order."""
    center = np.mean(corners, axis=0)
    angles = np.arctan2(corners[:, 1] - center[1], corners[:, 0] - center[0])
    ordered = corners[np.argsort(angles)]
    first = np.argmin(np.sum(ordered, axis=1))
    return np.roll(ordered, -first, axis=0).astype(np.float32)


def detect_yellow_square(image: np.ndarray, lower_hsv: np.ndarray,
                         upper_hsv: np.ndarray, min_area_px: float
                         ) -> Tuple[Optional[np.ndarray], np.ndarray]:
    """Find the largest yellow contour and return its square and mask."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_hsv, upper_hsv)
    kernel = np.ones((5, 5), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, mask

    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < min_area_px:
        return None, mask
    corners = cv2.boxPoints(cv2.minAreaRect(contour))
    return order_corners(corners), mask


def detect_top_face(image: np.ndarray, outer_corners: np.ndarray,
                    lower_hsv: np.ndarray, upper_hsv: np.ndarray,
                    min_value: int, brightness_percentile: float
                    ) -> Tuple[Optional[np.ndarray], np.ndarray]:
    """Find the bright upper face inside the complete yellow die contour."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    die_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(die_mask, outer_corners.astype(np.int32), 255)
    yellow_mask = cv2.inRange(hsv, lower_hsv, upper_hsv)
    yellow_mask = cv2.bitwise_and(yellow_mask, die_mask)
    values = hsv[:, :, 2][yellow_mask > 0]
    if len(values) == 0:
        return None, np.zeros(die_mask.shape, dtype=np.uint8)
    threshold = max(
        min_value, int(np.percentile(values, brightness_percentile)))
    bright_lower = lower_hsv.copy()
    bright_lower[2] = threshold
    bright_mask = cv2.inRange(hsv, bright_lower, upper_hsv)
    bright_mask = cv2.bitwise_and(bright_mask, die_mask)
    kernel = np.ones((3, 3), dtype=np.uint8)
    bright_mask = cv2.morphologyEx(bright_mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(
        bright_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, bright_mask
    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < 100.0:
        return None, bright_mask
    return order_corners(cv2.boxPoints(cv2.minAreaRect(contour))), bright_mask


def rectify_face_and_count_pips(image: np.ndarray, corners: np.ndarray,
                                face_size_px: int = 240
                                ) -> Tuple[np.ndarray, np.ndarray, np.ndarray,
                                           np.ndarray]:
    """Rectify the top face and count its black circular pips."""
    destination = np.array([
        [0.0, 0.0], [face_size_px - 1.0, 0.0],
        [face_size_px - 1.0, face_size_px - 1.0], [0.0, face_size_px - 1.0],
    ], dtype=np.float32)
    transform = cv2.getPerspectiveTransform(corners, destination)
    face = cv2.warpPerspective(image, transform, (face_size_px, face_size_px))
    gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    # Estimate the local illumination of the yellow face, then retain only
    # pixels darker than that local background. This does not impose an
    # absolute "black" value, but avoids selecting bright/dark boundaries and
    # highlights, which an absolute contrast operation would include.
    illumination = cv2.GaussianBlur(gray, (0, 0), 13)
    contrast = cv2.subtract(illumination, gray)
    _, pip_mask = cv2.threshold(
        contrast, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Ignore the outer edge, which can contain the green table or a shadow.
    border = 20
    pip_mask[:border, :] = 0
    pip_mask[-border:, :] = 0
    pip_mask[:, :border] = 0
    pip_mask[:, -border:] = 0
    kernel = np.ones((3, 3), dtype=np.uint8)
    pip_mask = cv2.morphologyEx(pip_mask, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(
        pip_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    pip_centers = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if not 45.0 <= area <= 2200.0:
            continue
        perimeter = cv2.arcLength(contour, True)
        circularity = 0.0 if perimeter == 0.0 else (
            4.0 * np.pi * area / (perimeter * perimeter))
        if circularity >= 0.35:
            moments = cv2.moments(contour)
            pip_centers.append([
                moments['m10'] / moments['m00'],
                moments['m01'] / moments['m00'],
            ])
            cv2.drawContours(face, [contour], -1, (255, 0, 255), 2)
    return face, pip_mask, np.asarray(pip_centers, dtype=np.float32), transform


def face_angle_degrees(corners: np.ndarray) -> float:
    """Return square orientation in [-45, 45) degrees, modulo 90 degrees."""
    edge = corners[1] - corners[0]
    angle = float(np.degrees(np.arctan2(edge[1], edge[0])))
    return (angle + 45.0) % 90.0 - 45.0


class YellowDiceLocalizer(Node):
    """ROS node that localizes the centre of a known-size yellow die."""

    def __init__(self) -> None:
        """Configure subscriptions, publishers, and detection parameters."""
        super().__init__('yellow_dice_localizer')
        self.declare_parameter('image_topic', '/oak/rgb/image_raw/compressed')
        self.declare_parameter('camera_info_topic', '/oak/rgb/camera_info')
        self.declare_parameter('pose_topic', '/yellow_dice/pose')
        self.declare_parameter('point_topic', '/yellow_dice/position')
        self.declare_parameter('debug_image_topic', '/yellow_dice/debug_image')
        self.declare_parameter('die_mask_topic', '/yellow_dice/die_mask')
        self.declare_parameter('top_face_mask_topic', '/yellow_dice/top_face_mask')
        self.declare_parameter('pip_mask_topic', '/yellow_dice/pip_mask')
        self.declare_parameter('top_face_topic', '/yellow_dice/top_face')
        self.declare_parameter('face_number_topic', '/yellow_dice/face_number')
        self.declare_parameter(
            'face_orientation_topic', '/yellow_dice/face_orientation_deg')
        self.declare_parameter('die_size_m', 0.025)
        self.declare_parameter('top_face_min_value', 200)
        self.declare_parameter('top_face_brightness_percentile', 70.0)
        self.declare_parameter('hsv_lower', [15, 80, 80])
        self.declare_parameter('hsv_upper', [32, 255, 255])
        self.declare_parameter('min_area_px', 200.0)
        self.declare_parameter('publish_debug_image', True)
        self.declare_parameter('show_window', True)
        self.declare_parameter('window_name', 'Yellow die vision')
        self.declare_parameter('mask_window_name', 'Yellow die masks')

        self._bridge = CvBridge()
        self._camera_info: Optional[CameraInfo] = None
        result_qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self._pose_pub = self.create_publisher(
            PoseStamped, self.get_parameter('pose_topic').value, result_qos)
        self._point_pub = self.create_publisher(
            PointStamped, self.get_parameter('point_topic').value, result_qos)
        self._debug_pub = self.create_publisher(
            Image, self.get_parameter('debug_image_topic').value, 10)
        self._die_mask_pub = self.create_publisher(
            Image, self.get_parameter('die_mask_topic').value, 10)
        self._top_face_mask_pub = self.create_publisher(
            Image, self.get_parameter('top_face_mask_topic').value, 10)
        self._pip_mask_pub = self.create_publisher(
            Image, self.get_parameter('pip_mask_topic').value, 10)
        self._top_face_pub = self.create_publisher(
            Image, self.get_parameter('top_face_topic').value, 10)
        self._face_number_pub = self.create_publisher(
            Int32, self.get_parameter('face_number_topic').value, result_qos)
        self._orientation_pub = self.create_publisher(
            Float64, self.get_parameter('face_orientation_topic').value,
            result_qos)
        self.create_subscription(
            CameraInfo, self.get_parameter('camera_info_topic').value,
            self._camera_info_callback, 10)
        self.create_subscription(
            CompressedImage, self.get_parameter('image_topic').value,
            self._image_callback, 10)
        self.get_logger().info(
            'Waiting for image and CameraInfo; die_size_m must match the '
            'physical side length of the die.')

    def _camera_info_callback(self, message: CameraInfo) -> None:
        """Keep the most recent camera calibration."""
        self._camera_info = message

    def _image_callback(self, message: CompressedImage) -> None:
        """Detect the die in an image and publish its pose when available."""
        if self._camera_info is None:
            return
        try:
            image = self._bridge.compressed_imgmsg_to_cv2(
                message, desired_encoding='bgr8')
        except Exception as error:  # cv_bridge reports malformed JPEGs here.
            self.get_logger().warning(f'Cannot decode compressed image: {error}')
            return

        lower = np.array(self.get_parameter('hsv_lower').value, dtype=np.uint8)
        upper = np.array(self.get_parameter('hsv_upper').value, dtype=np.uint8)
        outer_corners, die_mask = detect_yellow_square(
            image, lower, upper,
            float(self.get_parameter('min_area_px').value))
        if outer_corners is None:
            self._publish_debug(
                image, message, None, None, None, None, None, None, die_mask,
                None, None)
            return
        corners, top_face_mask = detect_top_face(
            image, outer_corners, lower, upper,
            int(self.get_parameter('top_face_min_value').value),
            float(self.get_parameter('top_face_brightness_percentile').value))
        if corners is None:
            self._publish_debug(
                image, message, outer_corners, None, None, None, None, None,
                die_mask, top_face_mask, None)
            return

        pose = self._estimate_pose(corners, message)
        top_face, pip_mask, pip_centers, transform = rectify_face_and_count_pips(
            image, corners)
        pip_count = len(pip_centers)
        image_pips = self._image_pip_centers(pip_centers, transform)
        angle = face_angle_degrees(corners)
        face_number = pip_count if 1 <= pip_count <= 6 else None
        self._publish_debug(
            image, message, outer_corners, corners, pose, face_number, angle,
            image_pips, die_mask, top_face_mask, pip_mask)
        top_face_message = self._bridge.cv2_to_imgmsg(top_face, encoding='bgr8')
        top_face_message.header = message.header
        self._top_face_pub.publish(top_face_message)
        self._publish_mask(self._die_mask_pub, die_mask, message)
        self._publish_mask(self._top_face_mask_pub, top_face_mask, message)
        self._publish_mask(self._pip_mask_pub, pip_mask, message)
        if face_number is not None:
            self._face_number_pub.publish(Int32(data=face_number))
        self._orientation_pub.publish(Float64(data=angle))
        if pose is None:
            self.get_logger().warning('PnP failed for detected yellow die')
            return
        self._pose_pub.publish(pose)
        point = PointStamped()
        point.header = pose.header
        point.point = pose.pose.position
        self._point_pub.publish(point)

    @staticmethod
    def _image_pip_centers(pip_centers: np.ndarray,
                           transform: np.ndarray) -> np.ndarray:
        """Map pip centres from the rectified face back into the camera image."""
        if len(pip_centers) == 0:
            return pip_centers
        inverse = np.linalg.inv(transform)
        return cv2.perspectiveTransform(
            pip_centers.reshape((-1, 1, 2)), inverse).reshape((-1, 2))

    def _publish_mask(self, publisher, mask: np.ndarray,
                      message: CompressedImage) -> None:
        """Publish one mono8 processing mask with the input image timestamp."""
        mask_message = self._bridge.cv2_to_imgmsg(mask, encoding='mono8')
        mask_message.header = message.header
        publisher.publish(mask_message)

    def _estimate_pose(self, corners: np.ndarray,
                       image_message: CompressedImage) -> Optional[PoseStamped]:
        """Solve the camera-to-top-face pose from calibrated image corners."""
        assert self._camera_info is not None
        side = float(self.get_parameter('die_size_m').value)
        if side <= 0.0:
            self.get_logger().error('die_size_m must be positive')
            return None
        half_side = side / 2.0
        object_points = np.array([
            [-half_side, -half_side, 0.0],
            [half_side, -half_side, 0.0],
            [half_side, half_side, 0.0],
            [-half_side, half_side, 0.0],
        ], dtype=np.float32)
        camera_matrix = np.asarray(self._camera_info.k, dtype=np.float64)
        camera_matrix = camera_matrix.reshape((3, 3))
        distortion = np.asarray(self._camera_info.d, dtype=np.float64)
        success, rotation, translation = cv2.solvePnP(
            object_points, corners, camera_matrix, distortion,
            flags=cv2.SOLVEPNP_ITERATIVE)
        if not success or translation[2, 0] <= 0.0:
            return None

        rotation_matrix, _ = cv2.Rodrigues(rotation)
        quaternion = self._quaternion_from_matrix(rotation_matrix)
        pose = PoseStamped()
        pose.header = image_message.header
        # CameraInfo is authoritative for the camera coordinate frame.
        pose.header.frame_id = self._camera_info.header.frame_id
        pose.pose.position.x = float(translation[0, 0])
        pose.pose.position.y = float(translation[1, 0])
        pose.pose.position.z = float(translation[2, 0])
        pose.pose.orientation.x = quaternion[0]
        pose.pose.orientation.y = quaternion[1]
        pose.pose.orientation.z = quaternion[2]
        pose.pose.orientation.w = quaternion[3]
        return pose

    @staticmethod
    def _quaternion_from_matrix(
            matrix: np.ndarray) -> Tuple[float, float, float, float]:
        """Convert a 3x3 rotation matrix to an xyzw quaternion."""
        trace = float(np.trace(matrix))
        if trace > 0.0:
            scale = 2.0 * np.sqrt(trace + 1.0)
            return ((matrix[2, 1] - matrix[1, 2]) / scale,
                    (matrix[0, 2] - matrix[2, 0]) / scale,
                    (matrix[1, 0] - matrix[0, 1]) / scale, 0.25 * scale)
        index = int(np.argmax(np.diag(matrix)))
        next_index = (index + 1) % 3
        last_index = (index + 2) % 3
        values = [0.0, 0.0, 0.0]
        scale = 2.0 * np.sqrt(
            1.0 + matrix[index, index] - matrix[next_index, next_index] -
            matrix[last_index, last_index])
        values[index] = 0.25 * scale
        values[next_index] = (
            matrix[next_index, index] + matrix[index, next_index]) / scale
        values[last_index] = (
            matrix[last_index, index] + matrix[index, last_index]) / scale
        w = (matrix[last_index, next_index] -
             matrix[next_index, last_index]) / scale
        return values[0], values[1], values[2], w

    def _publish_debug(self, image: np.ndarray, message: CompressedImage,
                       outer_corners: Optional[np.ndarray],
                       corners: Optional[np.ndarray], pose: Optional[PoseStamped],
                       pip_count: Optional[int], angle: Optional[float],
                       pip_points: Optional[np.ndarray] = None,
                       die_mask: Optional[np.ndarray] = None,
                       top_face_mask: Optional[np.ndarray] = None,
                       pip_mask: Optional[np.ndarray] = None) -> None:
        """Publish an annotated image when debug output is enabled."""
        debug = image.copy()
        if outer_corners is not None:
            cv2.polylines(debug, [outer_corners.astype(np.int32)], True,
                          (255, 255, 255), 2)
        if corners is not None:
            cv2.polylines(debug, [corners.astype(np.int32)], True,
                          (0, 0, 0), 2)
            center = np.mean(corners, axis=0)
            x_axis = (corners[1] + corners[2]) / 2.0 - center
            y_axis = (corners[2] + corners[3]) / 2.0 - center
            origin = tuple(np.round(center).astype(int))
            x_end = tuple(np.round(center + 0.7 * x_axis).astype(int))
            y_end = tuple(np.round(center + 0.7 * y_axis).astype(int))
            cv2.arrowedLine(debug, origin, x_end, (0, 0, 255), 2)
            cv2.arrowedLine(debug, origin, y_end, (255, 0, 0), 2)
            cv2.putText(debug, 'x', x_end, cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 0, 255), 1, cv2.LINE_AA)
            cv2.putText(debug, 'y', y_end, cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 0, 0), 1, cv2.LINE_AA)
        if pip_points is not None:
            for index, point in enumerate(pip_points, start=1):
                location = tuple(np.round(point).astype(int))
                cv2.circle(debug, location, 7, (255, 255, 255), 2)
                cv2.putText(debug, str(index), (location[0] + 8, location[1]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1,
                            cv2.LINE_AA)
        label = 'not localized'
        if pose is not None:
            position = pose.pose.position
            label = f'x={position.x:.3f} y={position.y:.3f} z={position.z:.3f} m'
        cv2.putText(debug, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (255, 0, 255), 2, cv2.LINE_AA)
        if angle is not None:
            number = '?' if pip_count is None else str(pip_count)
            face_label = f'face={number}, angle={angle:.1f} deg'
            cv2.putText(debug, face_label, (20, 75),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2,
                        cv2.LINE_AA)
        if self.get_parameter('publish_debug_image').value:
            debug_message = self._bridge.cv2_to_imgmsg(debug, encoding='bgr8')
            debug_message.header = message.header
            self._debug_pub.publish(debug_message)
        if self.get_parameter('show_window').value:
            cv2.imshow(self.get_parameter('window_name').value, debug)
            if die_mask is not None:
                masks = [
                    self._mask_preview(die_mask, 'yellow die'),
                    self._mask_preview(top_face_mask, 'top face'),
                    self._mask_preview(pip_mask, 'pip contrast'),
                ]
                cv2.imshow(self.get_parameter('mask_window_name').value,
                           cv2.hconcat(masks))
            cv2.waitKey(1)

    @staticmethod
    def _mask_preview(mask: Optional[np.ndarray], label: str) -> np.ndarray:
        """Make a labelled, consistently sized preview of a binary mask."""
        if mask is None:
            preview = np.zeros((240, 320), dtype=np.uint8)
        else:
            preview = cv2.resize(mask, (320, 240),
                                 interpolation=cv2.INTER_NEAREST)
        preview = cv2.cvtColor(preview, cv2.COLOR_GRAY2BGR)
        cv2.putText(preview, label, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (180, 180, 180), 2, cv2.LINE_AA)
        return preview


def main(args=None) -> None:
    """Start the yellow-die localization node."""
    rclpy.init(args=args)
    node = YellowDiceLocalizer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()
