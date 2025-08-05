from lib.kits.basic import *


def compute_mesh_volume(
    verts: Union[torch.Tensor, np.ndarray],
    faces: Union[torch.Tensor, np.ndarray],
) -> torch.Tensor:
    ''' 
    Computes the volume of a mesh object through triangles.
    References:
    1. https://github.com/muelea/shapy/blob/a5daa70ce619cbd2a218cebbe63ae3a4c0b771fd/mesh-mesh-intersection/body_measurements/body_measurements.py#L201-L215
    2. https://stackoverflow.com/questions/1406029/how-to-calculate-the-volume-of-a-3d-mesh-object-the-surface-of-which-is-made-up

    ### Args
    - verts: `torch.Tensor` or `np.ndarray`, shape = ((...B,) V, C=3)
    - faces: `torch.Tensor` or `np.ndarray`, shape = (T, K=3) where T = #triangles

    ### Returns
    - volume: `torch.Tensor`, shape = (...B,) or (,), in m^3.
    '''
    faces = to_numpy(faces)

    # Get triangles' xyz.
    batch_shape = verts.shape[:-2]
    V = verts.shape[-2]
    verts = verts.reshape(-1, V, 3)  # (B', V, C=3)

    tris = verts[:, faces]  # (B', T, K=3, C=3)
    tris = tris.reshape(*batch_shape, -1, 3, 3)  # (..., T, K=3, C=3)

    x = tris[..., 0]  # (..., T, K=3)
    y = tris[..., 1]  # (..., T, K=3)
    z = tris[..., 2]  # (..., T, K=3)

    volume = (
            -x[..., 2] * y[..., 1] * z[..., 0] +
             x[..., 1] * y[..., 2] * z[..., 0] +
             x[..., 2] * y[..., 0] * z[..., 1] -
             x[..., 0] * y[..., 2] * z[..., 1] -
             x[..., 1] * y[..., 0] * z[..., 2] +
             x[..., 0] * y[..., 1] * z[..., 2]
        ).sum(dim=-1).abs() / 6.0
    return volume