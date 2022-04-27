from omegaconf import DictConfig

from arm.qte.networks import Qattention3DNet
from arm.qte.qattention_agent import QAttentionAgent
from arm.c2farm.qattention_stack_agent import QAttentionStackAgent
from arm.preprocess_agent import PreprocessAgent


def create_agent(cfg: DictConfig, env, depth_0bounds=None, cam_resolution=None):
    VOXEL_FEATS = 3
    LATENT_SIZE = 64
    cam_resolution = cam_resolution or [128, 128]

    include_prev_layer = False

    num_rotation_classes = int(360. // cfg.method.rotation_resolution)
    qattention_agents = []
    for depth, vox_size in enumerate(cfg.method.voxel_sizes):
        last = depth == len(cfg.method.voxel_sizes) - 1
        unet3d = Qattention3DNet(
            in_channels=VOXEL_FEATS + 3 + 1 + 3,
            out_channels=1,
            voxel_size=vox_size,
            timesteps=cfg.replay.timesteps,
            out_dense=((num_rotation_classes * 3) + 2) if last else 0,
            kernels=LATENT_SIZE,
            norm=None if 'None' in cfg.method.norm else cfg.method.norm,
            dense_feats=128,
            activation=cfg.method.activation,
            low_dim_size=env.low_dim_state_len,
            include_prev_layer=include_prev_layer and depth > 0)


        qattention_agent = QAttentionAgent(
            layer=depth,
            coordinate_bounds=depth_0bounds,
            unet3d=unet3d,
            camera_names=cfg.rlbench.cameras,
            voxel_size=vox_size,
            bounds_offset=cfg.method.bounds_offset[depth - 1] if depth > 0 else None,
            image_crop_size=cfg.method.image_crop_size,
            tau=cfg.method.tau,
            lr=cfg.method.lr,
            lambda_trans_qreg=cfg.method.lambda_trans_qreg,
            lambda_rot_qreg=cfg.method.lambda_rot_qreg,
            include_low_dim_state=True,
            image_resolution=cam_resolution,
            batch_size=cfg.replay.batch_size,
            timesteps=cfg.replay.timesteps,
            voxel_feature_size=VOXEL_FEATS,
            exploration_strategy=cfg.method.exploration_strategy,
            lambda_weight_l2=cfg.method.lambda_weight_l2,
            num_rotation_classes=num_rotation_classes,
            rotation_resolution=cfg.method.rotation_resolution,
            grad_clip=0.01,
            gamma=0.99,
            tree_search_breadth=cfg.method.tree_search_breadth,
            tree_during_update=cfg.method.tree_during_update,
            tree_during_act=cfg.method.tree_during_act
        )
        qattention_agents.append(qattention_agent)

    for i in range(len(qattention_agents) - 1):
        qattention_agents[i].give_next_layer_qattention(qattention_agents[i + 1])

    rotation_agent = QAttentionStackAgent(
        qattention_agents=qattention_agents,
        rotation_resolution=cfg.method.rotation_resolution,
        camera_names=cfg.rlbench.cameras,
    )
    preprocess_agent = PreprocessAgent(pose_agent=rotation_agent)
    return preprocess_agent
