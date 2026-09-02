import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from easy_motion.motion_client import MotionClient
from easy_motion_msgs.srv import DiceIdentification
from moveit_msgs.msg import MoveItErrorCodes
from typing import Tuple

GRIPPER_ACTION_NAME = '/gripper_action_controller/gripper_cmd'


class VisionClientNode(Node):
    def __init__(self):
        super().__init__('vision_client_node', use_global_arguments=False)
        self.dice_identification_client = self.create_client(
            DiceIdentification, 'dice_identification')

    def dice_identification(self) -> Tuple[int, PoseStamped, bool]:
        if not self.dice_identification_client.wait_for_service(timeout_sec=5.0):
            raise RuntimeError("DiceIdentification service not available")

        request = DiceIdentification.Request()

        result_future = self.dice_identification_client.call_async(request)
        rclpy.spin_until_future_complete(self, result_future)

        face_number = result_future.result().face_number
        pose = result_future.result().pose
        success = result_future.result().success

        return face_number, pose, success


def main():
    rclpy.init()

    # MotionClient is already a Node, so we can use its logger directly
    motion_client_node = MotionClient(gripper_action_name=GRIPPER_ACTION_NAME)
    vision_client_node = VisionClientNode()
    demo_node = rclpy.create_node(node_name="demo_node", use_global_arguments=False)

    logger = demo_node.get_logger()

    # --- 0) Move to home configuration and open gripper---
    home_joints = [0.0, -1.97, 2.13, -1.83, -1.50, 0.0]

    logger.info("Moving to home configuration...")
    result = motion_client_node.move_to_joint(home_joints)
    logger.info(f"Home reached: {result.val}")
    reached_goal, stalled = motion_client_node.gripper_command(position=0.045)  # Open gripper

    if result.val != MoveItErrorCodes.SUCCESS:
        logger.error(f"Failed to reach home configuration: {result.val}")
        motion_client_node.destroy_node()
        vision_client_node.destroy_node()
        demo_node.destroy_node()
        rclpy.shutdown()
        return 0

    # --- 1) Dice Identification
    _, dice_pose, success = vision_client_node.dice_identification()
    if not success:
        logger.error('Dice identification failed')
        motion_client_node.destroy_node()
        vision_client_node.destroy_node()
        demo_node.destroy_node()
        rclpy.shutdown()
        return 0

    logger.info(f'Dice position {dice_pose}')

    pick_pose = PoseStamped()
    pick_pose.header.frame_id = "dice_tf"
    pick_pose.header.stamp = demo_node.get_clock().now().to_msg()
    pick_pose.pose.position.x = 0.0
    pick_pose.pose.position.y = 0.0
    pick_pose.pose.position.z = 0.0
    pick_pose.pose.orientation.x = 1.0
    pick_pose.pose.orientation.y = 0.0
    pick_pose.pose.orientation.z = 0.0
    pick_pose.pose.orientation.w = 0.0

    logger.info("Moving to pick pose...")
    result = motion_client_node.move_to_pose(pick_pose, cartesian_motion=False)

    if result.val != MoveItErrorCodes.SUCCESS:
        logger.error(f"Failed to reach pick pose: {result.val}")
        motion_client_node.destroy_node()
        vision_client_node.destroy_node()
        demo_node.destroy_node()
        rclpy.shutdown()
        return 0

    logger.info(f"Pick pose reached: {result.val}")

    # --- 3) Close gripper + attach object ---
    logger.info("Closing gripper...")
    reached_goal, stalled = motion_client_node.gripper_command(position=0.0)  # 0.0 = closed
    logger.info(f"Gripper closed (reached_goal={reached_goal}, stalled={stalled})")

    logger.info("Attaching object...")
    # "tool0" is the default TCP frame of UR10e
    success = motion_client_node.attach_object("dice", "tool0")
    logger.info(f"Attach success: {success}")

    # --- 4) Move to another pose (place pose) ---
    place_pose = PoseStamped()
    place_pose.header.frame_id = "tip"
    place_pose.header.stamp = demo_node.get_clock().now().to_msg()
    place_pose.pose.position.x = 0.0
    place_pose.pose.position.y = 0.0
    place_pose.pose.position.z = -0.1
    place_pose.pose.orientation.x = 0.0
    place_pose.pose.orientation.y = 0.0
    place_pose.pose.orientation.z = 0.0
    place_pose.pose.orientation.w = 1.0

    logger.info("Moving to place pose...")
    result = motion_client_node.move_to_pose(place_pose, cartesian_motion=False)
    logger.info(f"Place pose reached: {result.val}")
    if result.val != MoveItErrorCodes.SUCCESS:
        logger.error(f"Failed to reach home configuration: {result.val}")
        motion_client_node.destroy_node()
        vision_client_node.destroy_node()
        demo_node.destroy_node()
        rclpy.shutdown()
        return 0

    # --- 5) Open gripper + detach object ---
    logger.info("Opening gripper...")
    # 0.045 = open, adjust depending on gripper model
    reached_goal, stalled = motion_client_node.gripper_command(position=0.045)
    logger.info(f"Gripper opened (reached_goal={reached_goal}, stalled={stalled})")

    logger.info("Detaching object...")
    success = motion_client_node.detach_object("dice")
    logger.info(f"Detach success: {success}")

    motion_client_node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
