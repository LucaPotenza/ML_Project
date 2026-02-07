import torch

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