from pathlib import Path
from .monitor import TimeMonitor, TimeMonitorDisabled

class ProjManager():
    root = Path(__file__).parent.parent.parent # root / lib / utils / path_manager.py
    assert (root.exists()), 'Can\'t find the path of project root.'

    configs = root / 'configs'  # Generally, you are not supposed to access deep config through path.
    inputs  = root / 'data_inputs'
    outputs = root / 'data_outputs'
    assert (configs.exists()), 'Make sure you have a \'configs\' folder in the root directory.'
    assert (inputs.exists()), 'Make sure you have a \'data_inputs\' folder in the root directory.'
    assert (outputs.exists()), 'Make sure you have a \'data_outputs\' folder in the root directory.'

    # Default values.
    cfg = None
    time_monitor = TimeMonitorDisabled()

    @staticmethod
    def init_with_cfg(cfg):
        ProjManager.cfg = cfg
        ProjManager.exp_outputs = Path(cfg.output_dir)
        if cfg.get('enable_time_monitor', False):
            ProjManager.time_monitor = TimeMonitor(ProjManager.exp_outputs, record_birth_block=False)