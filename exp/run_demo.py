from lib.kits.demo import *
from lib.utils.vis.p3d_renderer import Renderer
import cv2
import torch
import numpy as np
import numpy as np
import trimesh
import pyrender

from typing import List, Optional, Union, Tuple
from pathlib import Path

from lib.utils.vis import ColorPalette
from lib.utils.data import to_numpy
from lib.utils.media import save_img

import cv2
import torch
import numpy as np

from pytorch3d.renderer import (
    PerspectiveCameras,
    TexturesVertex,
    PointLights,
    Materials,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
)
from pytorch3d.structures import Meshes
from pytorch3d.structures.meshes import join_meshes_as_scene
from pytorch3d.renderer.cameras import look_at_rotation
from pytorch3d.transforms import axis_angle_to_matrix

def main():
    # ⛩️ 0. Preparation.
    args = parse_args()
    outputs_root = Path(args.output_path)
    outputs_root.mkdir(parents=True, exist_ok=True)

    monitor = TimeMonitor()

    # ⛩️ 1. Preprocess.

    with monitor('Data Preprocessing'):
        with monitor('Load Inputs'):
            raw_imgs, inputs_meta = load_inputs(args)

        with monitor('Detector Initialization'):
            get_logger(brief=True).info('🧱 Building detector.')
            detector = build_detector(
                    batch_size   = args.det_bs,
                    max_img_size = args.det_mis,
                    device       = args.device,
                )

        with monitor('Detecting'):
            get_logger(brief=True).info(f'🖼️ Detecting...')
            detector_outputs = detector(raw_imgs)

        with monitor('Patching & Loading'):
            patches, det_meta = imgs_det2patches(raw_imgs, *detector_outputs, args.max_instances)  # N * (256, 256, 3)
        if len(patches) == 0:
            get_logger(brief=True).error(f'🚫 No human instance detected. Please ensure the validity of your inputs!')
        get_logger(brief=True).info(f'🔍 Totally {len(patches)} human instances are detected.')


    # ⛩️ 2. Human skeleton and mesh recovery.
    with monitor('Pipeline Initialization'):
        get_logger(brief=True).info(f'🧱 Building recovery pipeline.')
        pipeline = build_inference_pipeline(model_root=args.model_root, device=args.device)

    with monitor('Recovery'):
        get_logger(brief=True).info(f'🏃 Recovering with B={args.rec_bs}...')
        pd_params, pd_cam_t = [], []
        pd_focal_length = []
        for bw in asb(total=len(patches), bs_scope=args.rec_bs, enable_tqdm=True):
            patches_i = patches[bw.sid:bw.eid]  # (N, 256, 256, 3)
            patches_normalized_i = (patches_i - IMG_MEAN_255) / IMG_STD_255  # (N, 256, 256, 3)
            patches_normalized_i = patches_normalized_i.transpose(0, 3, 1, 2)  # (N, 3, 256, 256)
            with torch.no_grad():
                outputs = pipeline(patches_normalized_i)
            pd_params.append({k: v.detach().cpu().clone() for k, v in outputs['pd_params'].items()})
            pd_cam_t.append(outputs['pd_cam_t'].detach().cpu().clone())
            pd_focal_length.append(outputs['focal_length'].detach().cpu().clone())

        pd_params = assemble_dict(pd_params, expand_dim=False)  # [{k:[x]}, {k:[y]}] -> {k:[x, y]}
        pd_cam_t = torch.cat(pd_cam_t, dim=0)
        pd_focal_length = torch.cat(pd_focal_length, dim=0)
        dump_data = {
                'patch_cam_t' : pd_cam_t.numpy(),
                **{k: v.numpy() for k, v in pd_params.items()},
            }

        get_logger(brief=True).info(f'🤌 Preparing meshes...')
        m_skin, m_skel = prepare_mesh(pipeline, pd_params)
        get_logger(brief=True).info(f'🏁 Done.')

        vertices = m_skin['v'][0:1].float()    # (1, V, 3)
        center = vertices.mean(dim=1, keepdim=True)
        verts_centered = vertices - center     # (1, V, 3)
        verts_centered = verts_centered.to('cuda')
        verts = verts_centered[0].cpu().numpy()  # (V, 3)
        import matplotlib.pyplot as plt
        plt.scatter(verts[:,0], verts[:,1], s=1)
        plt.gca().set_aspect('equal')
        plt.show()

        # 1. Initialize the Renderer
        fx, fy = pd_focal_length[0]
        cx, cy = 128, 128  # 通常为 width/2, height/2
        K = torch.tensor([[fx, 0, cx],
                  [0, fy, cy],
                  [0, 0, 1]], dtype=torch.float32).to('cuda')
        renderer = Renderer(
            width=1024,   # It can be changed to the output resolution you need
            height=1024,
            faces=m_skin['f'],   
            device='cuda',
            focal_length=1500
        )
        from pytorch3d.renderer import PointLights
        renderer.lights = PointLights(device='cuda', location=[[0, 0, 10]])


        # 2. For mesh vertex data, it is necessary to ensure that shape=(N, V, 3) or (V, 3)
        #print(type(m_skin['v']), getattr(m_skin['v'], "shape", None))
        #vertices = m_skin['v'].float().to('cuda')


        # 3. Design different perspectives: such as front view, left view, top view, oblique view, etc
        def get_view_camera(view='front', t_z=2.0):
            if view == 'front':
                R = torch.eye(3)[None].float().to('cuda')
                T = torch.zeros(1, 3).float().to('cuda')
            elif view == 'side':
                angle = -90 / 180 * np.pi
                R = axis_angle_to_matrix(torch.tensor([[0, 1, 0]]).float().to('cuda') * angle)[None]
                T = torch.zeros(1, 3).float().to('cuda')
            elif view == 'top':
                angle = 90 / 180 * np.pi
                R = axis_angle_to_matrix(torch.tensor([[1, 0, 0]]).float().to('cuda') * angle)[None]
                T = torch.zeros(1, 3).float().to('cuda')
            elif view == 'oblique':
                angle = 45 / 180 * np.pi
                R = axis_angle_to_matrix(torch.tensor([[0, 1, 1]]).float().to('cuda') * angle)[None]
                T = torch.zeros(1, 3).float().to('cuda')
            else:
                R = torch.eye(3)[None].float().to('cuda')
                T = torch.zeros(1, 3).float().to('cuda')
            T = torch.tensor([[0, 0, t_z]]).float().to('cuda')
            return R, T

        # 4. Render each perspective and save
        views = ['front', 'side', 'top', 'oblique']
        t_z = 4.0 
        for i in range(m_skin['v'].shape[0]):
            vertices = m_skin['v'][i:i+1].float()    # (1, V, 3)
            center = vertices.mean(dim=1, keepdim=True)
            verts_centered = vertices - center
            verts_centered = verts_centered.to('cuda')

            for view in views:
                R, T = get_view_camera(view, t_z=t_z)
                device = verts_centered.device
                renderer.R = R.to(device)
                renderer.T = T.to(device)
                renderer.cameras = renderer.create_camera(R, T).to(device)
                img_bg = np.ones((1024, 1024, 3), dtype=np.uint8) * 255  
                image = renderer.render_mesh(verts_centered, background=img_bg, colors=[0.72, 0.88, 0.97])        
                cv2.imwrite(f"mesh_{i}_{view}_Tz{t_z}.png", image[..., ::-1])

        get_logger(brief=True).info(f'📸 Meshes rendered and saved for all instances (t_z={t_z}).')
        
    # ⛩️ 3. Postprocess.
    with monitor('Visualization'):
        if args.ignore_skel:
            m_skel = None
        results, full_cam_t = visualize_full_img(pd_cam_t, raw_imgs, det_meta, m_skin, m_skel, args.have_caption)
        dump_data['full_cam_t'] = full_cam_t
        # Save rendering and dump results.
        if inputs_meta['type'] == 'video':
            seq_name = f'{pipeline.name}-' + inputs_meta['seq_name']
            save_video(results, outputs_root / f'{seq_name}.mp4')
            # Dump data for each frame, here `i` refers to frames, `j` refers to image patches.
            dump_results = []
            cur_patch_j = 0
            for i in range(len(raw_imgs)):
                n_patch_cur_img = det_meta['n_patch_per_img'][i]
                dump_results_i = {k: v[cur_patch_j:cur_patch_j+n_patch_cur_img] for k, v in dump_data.items()}
                dump_results_i['bbx_cs'] = det_meta['bbx_cs_per_img'][i]
                cur_patch_j += n_patch_cur_img
                dump_results.append(dump_results_i)
            np.save(outputs_root / f'{seq_name}.npy', dump_results)
        elif inputs_meta['type'] == 'imgs':
            img_names = [f'{pipeline.name}-{fn.name}' for fn in inputs_meta['img_fns']]
            # Dump data for each image separately, here `i` refers to images, `j` refers to image patches.
            cur_patch_j = 0
            for i, img_name in enumerate(tqdm(img_names, desc='Saving images')):
                n_patch_cur_img = det_meta['n_patch_per_img'][i]
                dump_results_i = {k: v[cur_patch_j:cur_patch_j+n_patch_cur_img] for k, v in dump_data.items()}
                dump_results_i['bbx_cs'] = det_meta['bbx_cs_per_img'][i]
                cur_patch_j += n_patch_cur_img
                save_img(results[i], outputs_root / f'{img_name}.jpg')
                np.savez(outputs_root / f'{img_name}.npz', **dump_results_i)

        get_logger(brief=True).info(f'🎨 Rendering results are under {outputs_root}.')

    get_logger(brief=True).info(f'🎊 Everything is done!')
    monitor.report()


if __name__ == '__main__':
    main()