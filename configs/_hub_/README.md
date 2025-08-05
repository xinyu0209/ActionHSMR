# _hub

Configs here shouldn't be used as what Hydra calls a "config group". Configs here serve as a hub for other configs to reference. Most things here usually won't be used in a certain experiment.

For example, the details of each datasets can be defined here, and than I only need to reference the `Human3.6M` dataset through `${_hub_.datasets.h36m}` in other configs.

Each `.yaml` file here will be loaded separately in `base.yaml`. Check the `base.yaml` to understand how it works.
