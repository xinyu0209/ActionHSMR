import sys
import json
import rich
import rich.text
import rich.tree
import rich.syntax
import hydra
from typing import List, Optional, Union, Any
from pathlib import Path
from omegaconf import OmegaConf, DictConfig, ListConfig
from pytorch_lightning.utilities import rank_zero_only

from lib.info.log import get_logger

from .proj_manager import ProjManager as PM


def get_PM_info_dict():
    ''' Get a OmegaConf object containing the information from the ProjManager. '''
    PM_info = OmegaConf.create({
            '_pm_': {
                'root'   : str(PM.root),
                'inputs' : str(PM.inputs),
                'outputs': str(PM.outputs),
            }
        })
    return PM_info


def get_PM_info_list():
    ''' Get a list containing the information from the ProjManager. '''
    PM_info = [
        f'_pm_.root={str(PM.root)}',
        f'_pm_.inputs={str(PM.inputs)}',
        f'_pm_.outputs={str(PM.outputs)}',
    ]
    return PM_info


def entrypoint_with_args(*args, log_cfg=True, **kwargs):
    '''
    This decorator extends the `hydra.main` decorator in these parts:
    - Inject some runtime-known arguments, e.g., `proj_root`.
    - Enable additional arguments that needn't to be specified in command line.
        - Positional arguments are added to the command line arguments directly, so make sure they are valid.
            - e.g., \'exp=<...>\', \'+extra=<...>\', etc.
        - Key-specified arguments have the same effect as command line arguments {k}={v}.
    - Check the validation of experiment name.
    '''

    overrides = get_PM_info_list()

    for arg in args:
        overrides.append(arg)

    for k, v in kwargs.items():
        overrides.append(f'{k}={v}')

    overrides.extend(sys.argv[1:])

    def entrypoint_wrapper(func):
        # Import extra pre-specified arguments.
        if len(overrides) > 0:
            # The args from command line have higher priority, so put them in the back.
            sys.argv = sys.argv[:1] + overrides + sys.argv[1:]
            _log_exp_info(func.__name__, overrides)

        @hydra.main(version_base=None, config_path=str(PM.configs), config_name='base.yaml')
        def entrypoint_preprocess(cfg:DictConfig):
            # Resolve the references and make it editable.
            cfg = unfold_cfg(cfg)

            # Print out the configuration files.
            if log_cfg and cfg.get('show_cfg', True):
                sum_keys = ['output_dir', 'pipeline.name', 'data.name', 'exp_name', 'exp_tag']
                print_cfg(cfg, sum_keys=sum_keys)

            # Check the validation of experiment name.
            if cfg.get('exp_name') is None:
                get_logger(brief=True).fatal(f'`exp_name` is not given! You may need to add `exp=<certain_exp>` to the command line.')
                raise ValueError('`exp_name` is not given!')

            # Bind config.
            PM.init_with_cfg(cfg)
            try:
                with PM.time_monitor('exp', f'Main part of experiment `{cfg.exp_name}`.'):
                    # Enter the main function.
                    func(cfg)
            except Exception as e:
                raise e
            finally:
                PM.time_monitor.report(level='global')

            # TODO: Wrap a notifier here.

        return entrypoint_preprocess


    return entrypoint_wrapper

    #! This implementation can't dump the config files in default ways. In order to keep c
    # def entrypoint_wrapper(func):
    #     def entrypoint_preprocess():
    #         # Initialize the configuration module.
    #         with hydra.initialize_config_dir(version_base=None, config_dir=str(PM.configs)):
    #             get_logger(brief=True).info(f'Exp entry `{func.__name__}` is called with overrides: {overrides}')
    #             cfg = hydra.compose(config_name='base', overrides=overrides)

    #         cfg4dump_raw = cfg.copy()  # store the folded raw configuration files
    #         # Resolve the references and make it editable.
    #         cfg = unfold_cfg(cfg)

    #         # Print out the configuration files.
    #         if log_cfg:
    #             sum_keys = ['pipeline.name', 'data.name', 'exp_name']
    #             print_cfg(cfg, sum_keys=sum_keys)
    #         # Check the validation of experiment name.
    #         if cfg.get('exp_name') is None:
    #             get_logger().fatal(f'`exp_name` is not given! You may need to add `exp=<certain_exp>` to the command line.')
    #             raise ValueError('`exp_name` is not given!')
    #         # Enter the main function.
    #         func(cfg)
    #     return entrypoint_preprocess
    # return entrypoint_wrapper

def entrypoint(func):
    '''
    This decorator extends the `hydra.main` decorator in these parts:
    - Inject some runtime-known arguments, e.g., `proj_root`.
    - Check the validation of experiment name.
    '''
    return entrypoint_with_args()(func)


def unfold_cfg(
    cfg : Union[DictConfig, Any],
):
    '''
    Unfold the configuration files, i.e. from structured mode to container mode and recreate the
    configuration files. It will resolve all the references and make the config editable.

    ### Args
    - cfg: DictConfig or None

    ### Returns
    - cfg: DictConfig or None
    '''
    if cfg is None:
        return None

    cfg_container = OmegaConf.to_container(cfg, resolve=True)
    cfg = OmegaConf.create(cfg_container)
    return cfg


def recursively_simplify_cfg(
    node      : DictConfig,
    hide_misc : bool = True,
):
    if isinstance(node, DictConfig):
        for k in list(node.keys()):
            # We delete some terms that are not commonly concerned.
            if hide_misc:
                if k in ['_hub_', 'hydra', 'job_logging','backbone_ckpt']:
                    node.__delattr__(k)
                    continue
            node[k] = recursively_simplify_cfg(node[k], hide_misc)
    elif isinstance(node, ListConfig):
        if len(node) > 0 and all([
                not isinstance(x, DictConfig) \
            and not isinstance(x, ListConfig) \
            for x in node
        ]):
            # We fold all lists of basic elements (int, float, ...) into a single line if possible.
            folded_list_str = '*' + str(list(node))
            node = folded_list_str if len(folded_list_str) < 320 else node
        else:
            for i in range(len(node)):
                node[i] = recursively_simplify_cfg(node[i], hide_misc)
    return node


@rank_zero_only
def print_cfg(
    cfg     : Optional[DictConfig],
    title   : str  ='cfg',
    sum_keys: List[str] = [],
    show_all: bool = False
):
    '''
    Print configuration files using rich.

    ### Args
    - cfg: DictConfig or None
        - If None, print nothing.
    - sum_keys: List[str], default []
        - If keys given in the list exist in the first level of the configuration files,
          they will be printed in the summary part.
    - show_all: bool, default False
        - If False, hide terms starts with `_` in the configuration files's first level
          and some hydra supporting configs.
    '''

    theme = 'coffee'
    style = 'dim'

    tf_dict = { True: '◼', False: '◻' }
    print_setting = f'<< {tf_dict[show_all]} SHOW_ALL >>'
    tree = rich.tree.Tree(f'⌾ {title} - {print_setting}', style=style, guide_style=style)

    if cfg is None:
        tree.add('None')
        rich.print(tree)
        return

    # Clone a new one to avoid changing the original configuration files.
    cfg = cfg.copy()
    cfg = unfold_cfg(cfg)

    if not show_all:
        cfg = recursively_simplify_cfg(cfg)

    cfg_yaml = OmegaConf.to_yaml(cfg)
    cfg_yaml = rich.syntax.Syntax(cfg_yaml, 'yaml', theme=theme, line_numbers=True)
    tree.add(cfg_yaml)

    # Add a summary containing information only is commonly concerned.
    if len(sum_keys) > 0:
        concerned = {}
        for k_str in sum_keys:
            k_list = k_str.split('.')
            tgt = cfg
            for k in k_list:
                if tgt is not None:
                    tgt = tgt.get(k)
            if tgt is not None:
                concerned[k_str] = tgt
            else:
                get_logger().warning(f'Key `{k_str}` is not found in the configuration files.')

        tree.add(rich.syntax.Syntax(OmegaConf.to_yaml(concerned), 'yaml', theme=theme))

    rich.print(tree)


@rank_zero_only
def _log_exp_info(
    func_name : str,
    overrides : List[str],
):
    get_logger(brief=True).info(f'Exp entry `{func_name}` is called with overrides: {overrides}')