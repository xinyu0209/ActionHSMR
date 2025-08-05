## [Pyrender Renderer](https://github.com/mmatl/pyrender)

> The code was modified from HMR2.0: https://github.com/shubham-goel/4D-Humans/blob/main/hmr2/utils/renderer.py

This render is used to solve the compatibility issue of [Pytorch3D Renderer](../p3d_renderer/README.md). So, the API is fully adapted, but not all functions are implemented (e.g., `output_fn` is useless but still listed in the API).

Since this renderer is fully torch-independent, and I found the color of mesh is strange. So I tend to use Pytorch3D Renderer by default instead.