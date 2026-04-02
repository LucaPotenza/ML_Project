import torch
from torch.utils.data import Dataset
import numpy as np

# ============================================================================
# DATASET PERSONALIZZATO
# ============================================================================

class SketchViewDataset(Dataset):
    """Dataset per coppie (sketch, view, sketch_peer, view_peer, labels)"""

    def __init__(self, sketches, views, triples, labels):
        """
        Args:
            sketches: array numpy (N_sketch, pixels)
            views: array numpy (N_view, pixels)
            triples: array numpy (N_pairs, 4) → indici [sketch, view, sketch_peer, view_peer]
            labels: array numpy (N_pairs, 3) → [pos/neg, sketch_class, view_class]
        """
        self.sketches = torch.FloatTensor(sketches)
        self.views = torch.FloatTensor(views)
        self.triples = triples
        self.labels = labels

        # Calculate image size (assuming square image nxn)
        self.img_size = int(np.sqrt(sketches.shape[1]))

    def __len__(self):
        return len(self.triples)

    def __getitem__(self, idx):
        # Extract Triple index
        sketch_idx, view_idx, sketch_peer_idx, view_peer_idx = self.triples[idx]

        # Extract images and reshape at (1, H, W)
        sketch = self.sketches[sketch_idx].view(1, self.img_size, self.img_size)
        view = self.views[view_idx].view(1, self.img_size, self.img_size)
        sketch_peer = self.sketches[sketch_peer_idx].view(1, self.img_size, self.img_size)
        view_peer = self.views[view_peer_idx].view(1, self.img_size, self.img_size)

        # Extract label
        pos_neg, sketch_label, view_label = self.labels[idx].tolist()

        return {
            'sketch': sketch,
            'view': view,
            'sketch_peer': sketch_peer,
            'view_peer': view_peer,
            'sketch_label': torch.tensor(int(sketch_label), dtype=torch.long),
            'view_label': torch.tensor(int(view_label), dtype=torch.long)
        }
