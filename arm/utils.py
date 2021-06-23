import numpy as np
import pyrender
import torch
from rlbench.backend.const import DEPTH_SCALE
from scipy.spatial.transform import Rotation

SCALE_FACTOR = DEPTH_SCALE
DEFAULT_SCENE_SCALE = 2.0


def loss_weights(replay_sample, beta=1.0):
    loss_weights = 1.0
    if 'sampling_probabilities' in replay_sample:
        probs = replay_sample['sampling_probabilities']
        loss_weights = 1.0 / torch.sqrt(probs + 1e-10)
        loss_weights = (loss_weights / torch.max(loss_weights)) ** beta
    return loss_weights


def soft_updates(net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(
            tau * param.data + (1 - tau) * target_param.data
        )


def stack_on_channel(x):
    # expect (B, T, C, ...)
    return torch.cat(torch.split(x, 1, dim=1), dim=2).squeeze(1)


def normalize_quaternion(quat):
    return np.array(quat) / np.linalg.norm(quat)


def quaternion_to_discrete_euler(quaternion, resolution):
    euler = Rotation.from_quat(quaternion).as_euler('xyz', degrees=True) + 180
    assert np.min(euler) >= 0 and np.max(euler) <= 360
    disc = np.around((euler / resolution)).astype(int)
    disc[disc == int(360 / resolution)] = 0
    return disc


def discrete_euler_to_quaternion(discrete_euler, resolution):
    euluer = (discrete_euler * resolution) - 180
    return Rotation.from_euler('xyz', euluer, degrees=True).as_quat()


def point_to_voxel_index(
        point: np.ndarray,
        voxel_size: np.ndarray,
        coord_bounds: np.ndarray):
    bb_mins = np.array(coord_bounds[0:3])
    bb_maxs = np.array(coord_bounds[3:])
    dims_m_one = np.array([voxel_size] * 3) - 1
    bb_ranges = bb_maxs - bb_mins
    res = bb_ranges / (np.array([voxel_size] * 3) + 1e-12)
    voxel_indicy = np.minimum(
        np.floor((point - bb_mins) / (res + 1e-12)).astype(
            np.int32), dims_m_one)
    return voxel_indicy


def point_to_pixel_index(
        point: np.ndarray,
        extrinsics: np.ndarray,
        intrinsics: np.ndarray):
    point = np.array([point[0], point[1], point[2], 1])
    world_to_cam = np.linalg.inv(extrinsics)
    point_in_cam_frame = world_to_cam.dot(point)
    px, py, pz = point_in_cam_frame[:3]
    px = 2 * intrinsics[0, 2] - int(-intrinsics[0, 0] * (px / pz) + intrinsics[0, 2])
    py = 2 * intrinsics[1, 2] - int(-intrinsics[1, 1] * (py / pz) + intrinsics[1, 2])
    return px, py


def _compute_initial_camera_pose(scene):
    # Adapted from:
    # https://github.com/mmatl/pyrender/blob/master/pyrender/viewer.py#L1032
    centroid = scene.centroid
    scale = scene.scale
    if scale == 0.0:
        scale = DEFAULT_SCENE_SCALE
    s2 = 1.0 / np.sqrt(2.0)
    cp = np.eye(4)
    cp[:3, :3] = np.array([[0.0, -s2, s2], [1.0, 0.0, 0.0], [0.0, s2, s2]])
    hfov = np.pi / 6.0
    dist = scale / (2.0 * np.tan(hfov))
    cp[:3, 3] = dist * np.array([1.0, 0.0, 1.0]) + centroid
    return cp


def _from_trimesh_scene(
        trimesh_scene, bg_color=None, ambient_light=None):
    # convert trimesh geometries to pyrender geometries
    geometries = {name: pyrender.Mesh.from_trimesh(geom, smooth=False)
                  for name, geom in trimesh_scene.geometry.items()}
    # create the pyrender scene object
    scene_pr = pyrender.Scene(bg_color=bg_color, ambient_light=ambient_light)
    # add every node with geometry to the pyrender scene
    for node in trimesh_scene.graph.nodes_geometry:
        pose, geom_name = trimesh_scene.graph[node]
        scene_pr.add(geometries[geom_name], pose=pose)
    return scene_pr
