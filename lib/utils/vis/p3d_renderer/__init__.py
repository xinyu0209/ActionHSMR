from lib.kits.basic import *

import cv2
import imageio

from tqdm import tqdm

from lib.utils.vis import ColorPalette
from lib.utils.media import save_img

from .renderer import *


def render_mesh_overlay_img(
    faces       : Union[torch.Tensor, np.ndarray],
    verts       : torch.Tensor,
    K4          : List,
    img         : np.ndarray,
    output_fn   : Optional[Union[str, Path]] = None,
    device      : str = 'cuda',
    resize      : float = 1.0,
    Rt          : Optional[Tuple[torch.Tensor]] = None,
    mesh_color : Optional[Union[List[float], str]] = 'blue',
) -> Any:
    '''
    Render the mesh overlay on the input video frames.

    ### Args
    - faces: Union[torch.Tensor, np.ndarray], (V, 3)
    - verts: torch.Tensor, (V, 3)
    - K4: List
        - [fx, fy, cx, cy], the components of intrinsic camera matrix.
    - img: np.ndarray, (H, W, 3)
    - output_fn: Union[str, Path] or None
        - The output file path, if None, return the rendered img.
    - fps: int, default 30
    - device: str, default 'cuda'
    - resize: float, default 1.0
        - The resize factor of the output video.
    - Rt: Tuple of Tensor, default None
        - The extrinsic camera matrix, in the form of (R, t).
    '''
    frame = render_mesh_overlay_video(
            faces      = faces,
            verts      = verts[None],
            K4         = K4,
            frames     = img[None],
            device     = device,
            resize     = resize,
            Rt         = Rt,
            mesh_color = mesh_color,
        )[0]

    if output_fn is None:
        return frame
    else:
        save_img(frame, output_fn)


def render_mesh_overlay_video(
    faces      : Union[torch.Tensor, np.ndarray],
    verts      : torch.Tensor,
    K4         : List,
    frames     : np.ndarray,
    output_fn  : Optional[Union[str, Path]] = None,
    fps        : int = 30,
    device     : str = 'cuda',
    resize     : float = 1.0,
    Rt         : Tuple = None,
    mesh_color : Optional[Union[List[float], str]] = 'blue',
) -> Any:
    '''
    Render the mesh overlay on the input video frames.

    ### Args
    - faces: Union[torch.Tensor, np.ndarray], (V, 3)
    - verts: torch.Tensor, (L, V, 3)
    - K4: List
        - [fx, fy, cx, cy], the components of intrinsic camera matrix.
    - frames: np.ndarray, (L, H, W, 3)
    - output_fn: Union[str, Path] or None
        - The output file path, if None, return the rendered frames.
    - fps: int, default 30
    - device: str, default 'cuda'
    - resize: float, default 1.0
        - The resize factor of the output video.
    - Rt: Tuple, default None
        - The extrinsic camera matrix, in the form of (R, t).
    '''
    if isinstance(faces, torch.Tensor):
        faces = faces.cpu().numpy()
    assert len(K4) == 4, 'K4 must be a list of 4 elements.'
    assert frames.shape[0] == verts.shape[0], 'The length of frames and verts must be the same.'
    assert frames.shape[-1] == 3, 'The last dimension of frames must be 3.'
    if isinstance(mesh_color, str):
        mesh_color = ColorPalette.presets_float[mesh_color]

    # Prepare the data.
    L = frames.shape[0]
    focal_length = (K4[0] + K4[1]) / 2 # f = (fx + fy) / 2
    width, height = frames.shape[-2], frames.shape[-3]
    cx2, cy2 = int(K4[2] * 2), int(K4[3] * 2)
    # Prepare the renderer.
    renderer = Renderer(cx2, cy2, focal_length, device, faces)
    if Rt is not None:
        Rt = (to_tensor(Rt[0], device), to_tensor(Rt[1], device))
        renderer.create_camera(*Rt)

    if output_fn is None:
        result_frames = []
        for i in range(L):
            img = renderer.render_mesh(verts[i].to(device), frames[i], mesh_color)
            img = cv2.resize(img, (int(width * resize), int(height * resize)))
            result_frames.append(img)
        result_frames = np.stack(result_frames, axis=0)
        return result_frames
    else:
        writer = imageio.get_writer(output_fn, fps=fps, mode='I', format='FFMPEG', macro_block_size=1)
        # Render the video.
        output_seq_name = str(output_fn).split('/')[-1]
        for i in tqdm(range(L), desc=f'Rendering [{output_seq_name}]...'):
            img = renderer.render_mesh(verts[i].to(device), frames[i])
            writer.append_data(img)
            img = cv2.resize(img, (int(width * resize), int(height * resize)))
        writer.close()