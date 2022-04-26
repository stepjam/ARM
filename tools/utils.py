import numpy as np
from pyrep.const import RenderMode
from pyrep.objects import Dummy, VisionSensor


class RLBenchCinematic(object):

    def __init__(self):
        cam_placeholder = Dummy('cam_cinematic_placeholder')
        self._cam_base = Dummy('cam_cinematic_base')
        self._cam = VisionSensor.create([640, 480])
        self._cam.set_explicit_handling(True)
        self._cam.set_pose(cam_placeholder.get_pose())
        self._cam.set_parent(cam_placeholder)
        self._cam.set_render_mode(RenderMode.OPENGL3)
        self._frames = []

    def callback(self):
        self._cam.handle_explicitly()
        cap = (self._cam.capture_rgb() * 255).astype(np.uint8)
        self._frames.append(cap)

    def empty(self):
        self._frames.clear()

    @property
    def frames(self):
        return list(self._frames)
