import arm.baselines.sac.launch_utils as sac
from arm.baselines.dac.dac_agent import DACAgent
from arm.baselines.td3 import launch_utils as td3_launch_utils
from arm.network_utils import SiameseNet, CNNAndFcsNet
from arm.preprocess_agent import PreprocessAgent

REWARD_SCALE = 100.0


def create_replay(*args, **kwargs):
    return td3_launch_utils.create_replay(*args, **kwargs)


def fill_replay(*args, **kwargs):
    return td3_launch_utils.fill_replay(*args, **kwargs)


def create_agent(camera_name: str,
                 activation: str,
                 action_min_max,
                 image_resolution: list,
                 critic_lr,
                 actor_lr,
                 critic_weight_decay,
                 actor_weight_decay,
                 tau,
                 critic_grad_clip,
                 actor_grad_clip,
                 low_dim_state_len,
                 lambda_gp: float,
                 discriminator_lr: float,
                 discriminator_grad_clip: float,
                 discriminator_weight_decay: float
                 ):

    siamese_net = SiameseNet(
        input_channels=[3, 3],
        filters=[8],
        kernel_sizes=[7],
        strides=[1],
        activation=activation,
        norm=None,
    )

    discrim_net = CNNAndFcsNet(
        siamese_net=siamese_net,
        input_resolution=image_resolution,
        filters=[32, 64, 64],
        kernel_sizes=[3, 3, 3],
        strides=[2, 2, 2],
        norm=None,
        activation=activation,
        fc_layers=[128, 64, 1],
        low_dim_state_len=low_dim_state_len + 8)


    latent_size = 50
    decoder_net = sac.Decoder(image_resolution, latent_size, activation)
    encoder_net = sac.Encoder(image_resolution, latent_size, activation)
    critic_net = sac.MLP(1, latent_size, low_dim_state_len + 8, activation)
    actor_net = sac.MLP(8 * 2, latent_size, low_dim_state_len, activation)

    decoder_weight_decay=0.000001
    decoder_grad_clip=5
    decoder_lr=0.001
    decoder_latent_lambda=0.000001
    encoder_tau=0.05

    alpha=1.0
    alpha_auto_tune=True
    alpha_lr=0.0005

    dac_agent = DACAgent(
        discriminator_network=discrim_net,
        lambda_gp=lambda_gp,
        discriminator_lr=discriminator_lr,
        discriminator_grad_clip=discriminator_grad_clip,
        discriminator_weight_decay=discriminator_weight_decay,
        critic_network=critic_net,
        actor_network=actor_net,
        decoder_network=decoder_net,
        encoder_network=encoder_net,
        action_min_max=action_min_max,
        camera_name=camera_name,
        critic_lr=critic_lr,
        actor_lr=actor_lr,
        critic_weight_decay=critic_weight_decay,
        actor_weight_decay=actor_weight_decay,
        critic_tau=tau,
        critic_grad_clip=critic_grad_clip,
        actor_grad_clip=actor_grad_clip,
        alpha=alpha,
        alpha_auto_tune=alpha_auto_tune,
        alpha_lr=alpha_lr,
        decoder_weight_decay=decoder_weight_decay,
        decoder_grad_clip=decoder_grad_clip,
        decoder_lr=decoder_lr,
        decoder_latent_lambda=decoder_latent_lambda,
        encoder_tau=encoder_tau
    )

    return PreprocessAgent(pose_agent=dac_agent)
