import torch
import numpy as np
import configparser

class Parameters:
    # Class that contains all the parameters of the experiment

    def __init__(self):
        # Parametri training
        self.batch_size = 50
        self.n_epochs = 30
        self.learning_rate = 3e-4     # instead of 1e-4
        self.weight_decay = 1e-2      # instead of 5e-4

        # Optimization:
        self.warmup_ratio = 0.05   # 5% of total steps warmup
        self.min_lr_ratio = 0.01   # final LR = base_lr * 0.01


        # Parametri modello
        self.code_len = 64
        self.inputWH = 170
        self.use_dropout = True
        self.dropout_p = 0.5

        # Tipo modello
        self.model_type = 'dual'  # 'dual' o 'shared'

        # Loss parameters
        self.Cpos = 0.2
        self.Cneg = 10.0
        self.alpha = 1.0
        self.sw = 0.5  # peso regolarizzazione sketch intra-modale
        self.vw = 0.5  # peso regolarizzazione view intra-modale

        # Sampling probabilities
        self.ppos = 0.5
        self.pneg = 0.06

        # Dataset (no longer needs specific .mat paths, will use new loader)
        self.dataset = 'origin' # This might still be useful for conceptual grouping
        self.data_aug = True

        # Paths (da configurare)
        self.exp_name = 'shrec'
        self.exp_suffix = 'pytorch'
        self.model_dir = f'cache-pytorch/model-{self.exp_name}-{self.exp_suffix}'
        self.feats_dir = f'cache-pytorch/feats-{self.exp_name}-{self.exp_suffix}'

        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Random number generator for reproducibility in sampling
        self.rng = np.random.default_rng(seed=0)

def setup_experiment(cfg: configparser.ConfigParser, case: str) -> Parameters:
    """
    Create a Parameters() instance from a config section.
    Mirrors legacy setupExperiment() behavior at a high level.
    """
    prm = Parameters()

    # Required
    prm.exp_name = cfg.get(case, "name", fallback=prm.exp_name)
    prm.exp_suffix = case  # keep case name like legacy

    # Optional overrides
    if cfg.has_option(case, "batch_size"):
        prm.batch_size = cfg.getint(case, "batch_size")
    if cfg.has_option(case, "n_epochs"):
        prm.n_epochs = cfg.getint(case, "n_epochs")
    if cfg.has_option(case, "learning_rate"):
        prm.learning_rate = cfg.getfloat(case, "learning_rate")
    if cfg.has_option(case, "code_len"):
        prm.code_len = cfg.getint(case, "code_len")
    if cfg.has_option(case, "test_mode"):
        prm.test_mode = cfg.get(case, "test_mode")
    if cfg.has_option(case, "inputWH"):
        prm.inputWH = cfg.getint(case, "inputWH")

    # Loss / sampling (keep same names as your Parameters)
    for k in ["Cpos", "Cneg", "alpha", "sw", "vw", "ppos", "pneg"]:
        if cfg.has_option(case, k):
            setattr(prm, k, cfg.getfloat(case, k))

    # Epoch list for testing
    if cfg.has_option(case, "epoch_list"):
        prm.epoch_list = [int(x) for x in cfg.get(case, "epoch_list").split()]
    else:
        prm.epoch_list = [prm.n_epochs]

    # Dataset selection (optional)
    if cfg.has_option(case, "dataset"):
        prm.dataset = cfg.get(case, "dataset")
    if cfg.has_option(case, "data_aug"):
        # allow "True/False" or "0/1"
        prm.data_aug = cfg.getboolean(case, "data_aug")
    if cfg.has_option(case, "data_root"):
        prm.data_root = cfg.get(case, "data_root")
    if cfg.has_option(case, "cnn_spec_ss"):
        prm.cnn_spec_ss = cfg.get(case, "cnn_spec_ss")
    if cfg.has_option(case, "cnn_spec_vs"):
        prm.cnn_spec_vs = cfg.get(case, "cnn_spec_vs")

    # Paths (match your existing folder scheme)
    prm.model_dir = f'cache-pytorch/model-{prm.exp_name}-{prm.exp_suffix}'
    prm.feats_dir = f'cache-pytorch/feats-{prm.exp_name}-{prm.exp_suffix}'

    # NOTE: dataset file paths are now handled by load_shrec13_data
    return prm