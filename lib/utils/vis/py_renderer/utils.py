import numpy as np
import pyrender

from typing import List, Optional, Union, Tuple
from lib.utils.data import to_numpy


def create_camera(K4:List, Rt:Optional[Tuple]):
    if Rt is not None:
        cam_R, cam_t = Rt
        cam_R = to_numpy(cam_R).copy()
        cam_t = to_numpy(cam_t).copy()
        cam_t[0] *= -1
    else:
        cam_R = np.eye(3)
        cam_t = np.zeros(3)
    fx, fy, cx, cy = K4
    cam_pose = np.eye(4)
    cam_pose[:3, :3] = cam_R
    cam_pose[:3,  3] = cam_t
    camera = pyrender.IntrinsicsCamera( fx=fx, fy=fy, cx=cx, cy=cy, zfar=1e12 )
    return camera, cam_pose


def create_raymond_lights() -> List[pyrender.Node]:
    ''' Return raymond light nodes for the scene. '''
    thetas = np.pi * np.array([1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0])
    phis = np.pi * np.array([0.0, 2.0 / 3.0, 4.0 / 3.0])

    nodes = []

    for phi, theta in zip(phis, thetas):
        xp = np.sin(theta) * np.cos(phi)
        yp = np.sin(theta) * np.sin(phi)
        zp = np.cos(theta)

        z = np.array([xp, yp, zp])
        z = z / np.linalg.norm(z)
        x = np.array([-z[1], z[0], 0.0])
        if np.linalg.norm(x) == 0:
            x = np.array([1.0, 0.0, 0.0])
        x = x / np.linalg.norm(x)
        y = np.cross(z, x)

        matrix = np.eye(4)
        matrix[:3,:3] = np.c_[x,y,z]
        nodes.append(pyrender.Node(
            light  = pyrender.DirectionalLight(color=np.ones(3), intensity=0.75),
            matrix = matrix
        ))

    return nodes