import numpy as np

from arm.lpr.const import COLLISION, NO_COLLISION
from pyrep import PyRep
from pyrep.const import ObjectType
from pyrep.errors import IKError, ConfigurationPathError
from pyrep.objects import Dummy, CartesianPath
from pyrep.robots.arms.panda import Panda
from pyrep.robots.end_effectors.panda_gripper import PandaGripper

IK_STEPS_ON_PATH = 50


class RLBenchPathSampler(object):

    def __init__(self,
                 trajectory_points: int,
                 trajectory_samples: int,
                 trajectory_point_noise: float):
        self._trajectory_points = trajectory_points
        self._trajectory_samples = trajectory_samples
        self._trajectory_point_noise = trajectory_point_noise
        self._dummy = None
        self._robot = None

    def _get_path_points(self, cart_path: CartesianPath, config_path):
        points = []
        configs = []
        points_on_path = len(config_path) - 1
        for i in np.linspace(0.0, 1.0, self._trajectory_points):
            pos, euler = cart_path.get_pose_on_path(i)
            self._dummy.set_orientation(euler)
            q = self._dummy.get_quaternion()
            if q[-1] < 0:
                q = -q
            points.append(np.concatenate([pos, q]))
            configs.append(config_path[int(i * points_on_path)])
        return np.array(points).flatten(), np.array(configs).flatten()

    def _get_config_path(self, cart_path, distance):
        valid = True
        config_path = []
        initial_robot_config = self._robot.get_joint_positions()
        steps = np.maximum(
            (IK_STEPS_ON_PATH * np.clip(distance, 0, 1)).astype(int),
            self._trajectory_points)
        for t in np.linspace(0.0, 1.0, steps):
            trans, euler = cart_path.get_pose_on_path(t)
            try:
                new_config = self._robot.solve_ik_via_jacobian(trans, euler)
                self._robot.set_joint_positions(new_config)
                if self._robot.check_arm_collision():
                    valid = False
                config_path.append(new_config)
            except IKError as e:
                valid = False
                config_path = []
                break
        self._robot.set_joint_positions(initial_robot_config)
        return valid, np.array(config_path)

    def _path_plan_sample(self, pose_tp1, ignore_collisions=False):
        try:
            path = self._robot.get_path(
                position=pose_tp1[:3],
                quaternion=pose_tp1[3:],
                ignore_collisions=ignore_collisions,
                trials=100,
                max_configs=10,
                trials_per_goal=10,
            )
        except ConfigurationPathError as e:
            return [], []
        tip = self._robot.get_tip()
        init_angles = self._robot.get_joint_positions()
        cart_poses = []
        for i in range(len(self._robot.joints), len(path._path_points),
                       len(self._robot.joints)):
            points = path._path_points[i:i + len(self._robot.joints)]
            self._robot.set_joint_positions(points)
            cart_poses.append(
                list(tip.get_position()) + list(tip.get_orientation()))
        self._robot.set_joint_positions(init_angles)
        cart_path = CartesianPath.create(automatic_orientation=False)
        cart_path.insert_control_points(cart_poses)

        valid, joint_config_path = self._get_config_path(cart_path, 1)
        cart_p, cfg_p = [], []
        if valid:
            cart_p, cfg_p = (self._get_path_points(cart_path, joint_config_path))
            cart_p = np.concatenate([cart_p, NO_COLLISION], 0)
            cfg_p = np.concatenate([cfg_p, NO_COLLISION], 0)
        cart_path.remove()
        return cart_p, cfg_p

    def sample_paths_to_valued(self, next_best_pose: np.ndarray):
        valid_cart_paths = []
        valid_config_paths = []
        if self._dummy is None:
            self._dummy = Dummy.create()
        if self._robot is None:
            self._pyrep = PyRep()
            self._robot, self._gripper = Panda(), PandaGripper()
            self._robot_shapes = self._robot.get_objects_in_tree(
                object_type=ObjectType.SHAPE)

        pose_t = np.array(self._robot.get_tip().get_pose())
        pose_tp1 = next_best_pose

        # First check if we are colliding with anything
        colliding = self._robot.check_arm_collision()
        colliding_shapes = []
        if colliding:
            # Disable collisions with the objects that we are colliding with
            grasped_objects = self._gripper.get_grasped_objects()
            colliding_shapes = [s for s in self._pyrep.get_objects_in_tree(
                object_type=ObjectType.SHAPE) if (
                                        s.is_collidable() and
                                        s not in self._robot_shapes and
                                        s not in grasped_objects and
                                        self._robot.check_arm_collision(s))]
            [s.set_collidable(False) for s in colliding_shapes]

        cart_p, cfg_p = self._path_plan_sample(pose_tp1)
        if len(cart_p) > 0:
            valid_cart_paths.append(cart_p)
            valid_config_paths.append(cfg_p)

        for i, trial in enumerate(range(self._trajectory_samples)):
            if len(valid_cart_paths) >= self._trajectory_samples:
                break
            cart_path = CartesianPath.create(automatic_orientation=False)
            poses = np.linspace(pose_t, pose_tp1, 3)
            distance = np.linalg.norm(pose_t[:3] - pose_tp1[:3])
            if i > 0:
                poses[1:-1, :3] += np.random.normal(
                    size=poses[1:-1, :3].shape) * self._trajectory_point_noise
            eulers = []
            for gp in poses:
                self._dummy.set_pose(gp)
                eulers.append(self._dummy.get_orientation())
            cart_path.insert_control_points(
                np.concatenate([poses[:, :3], np.array(eulers)], 1))
            valid, config_path = self._get_config_path(cart_path, distance)
            if len(config_path) > 0:  # valid:
                cart_points, config_points = (
                    self._get_path_points(cart_path, config_path))
                valid_cart_paths.append(np.concatenate(
                    [cart_points, NO_COLLISION if valid else COLLISION], 0))
                valid_config_paths.append(np.concatenate(
                    [config_points, NO_COLLISION if valid else COLLISION], 0))
            cart_path.remove()
        [s.set_collidable(True) for s in colliding_shapes]
        return valid_cart_paths, valid_config_paths

    def check_cartesian_path(self, poses):
        # Will be (N, 7)
        cart_path = CartesianPath.create(automatic_orientation=False)
        eulers = []
        for gp in poses:
            self._dummy.set_pose(gp)
            eulers.append(self._dummy.get_orientation())
        cart_path.insert_control_points(
            np.concatenate([poses[:, :3], np.array(eulers)], 1))
        distance = np.linalg.norm(poses[0, :3] - poses[-1, :3])
        valid, config_path = self._get_config_path(cart_path, distance)
        valid_cart_path = valid_config_path = None
        if len(config_path) > 0:  # valid:
            cart_points, config_points = (
                self._get_path_points(cart_path, config_path))
            valid_cart_path = np.concatenate(
                [cart_points, NO_COLLISION if valid else COLLISION], 0)
            valid_config_path = np.concatenate(
                [config_points, NO_COLLISION if valid else COLLISION], 0)
        cart_path.remove()
        return valid_cart_path, valid_config_path
