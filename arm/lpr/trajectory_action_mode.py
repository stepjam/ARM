import numpy as np
from rlbench import ArmActionMode
from rlbench.action_modes.arm_action_modes import assert_action_shape
from rlbench.backend.exceptions import InvalidActionError
from rlbench.backend.scene import Scene

from pyrep.robots.configuration_paths.arm_configuration_path import \
    ArmConfigurationPath


class TrajectoryActionMode(ArmActionMode):
    """A sequence of joint configurations representing a trajectory.
    """

    def __init__(self, points: int):
        self._points = points

    def action(self, scene: Scene, action: np.ndarray):
        assert_action_shape(action, (7 * self._points,))
        if np.all(action == 0):
            raise InvalidActionError('No valid trajectory given.')
        path = ArmConfigurationPath(scene.robot.arm, action)
        done = False
        while not done:
            done = path.step()
            scene.step()
            success, terminate = scene.task.success()
            # If the task succeeds while traversing path, then break early
            if success:
                break

    def action_shape(self, scene: Scene) -> tuple:
        return 7 * self._points,
