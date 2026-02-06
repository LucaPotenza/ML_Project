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

        # Calcola dimensione immagine (assumendo quadrata)
        self.img_size = int(np.sqrt(sketches.shape[1]))

    def __len__(self):
        return len(self.triples)

    def __getitem__(self, idx):
        # Estrae indici dalla tripla
        sketch_idx, view_idx, sketch_peer_idx, view_peer_idx = self.triples[idx]

        # Estrae immagini e reshape a (1, H, W)
        sketch = self.sketches[sketch_idx].view(1, self.img_size, self.img_size)
        view = self.views[view_idx].view(1, self.img_size, self.img_size)
        sketch_peer = self.sketches[sketch_peer_idx].view(1, self.img_size, self.img_size)
        view_peer = self.views[view_peer_idx].view(1, self.img_size, self.img_size)

        # Estrae label
        pos_neg, sketch_label, view_label = self.labels[idx].tolist()

        return {
            'sketch': sketch,
            'view': view,
            'sketch_peer': sketch_peer,
            'view_peer': view_peer,
            'sketch_label': torch.tensor(int(sketch_label), dtype=torch.long),
            'view_label': torch.tensor(int(view_label), dtype=torch.long)
        }


class SketchOnlyDataset(Dataset):
    def __init__(self, sketches, labels=None):
        self.sketches = torch.from_numpy(sketches).float()
        self.labels = labels
        self.img_size = int(np.sqrt(sketches.shape[1]))

    def __len__(self):
        return self.sketches.shape[0]

    def __getitem__(self, idx):
        x = self.sketches[idx].view(1, self.img_size, self.img_size)
        out = {"x": x}
        if self.labels is not None:
            out["label"] = torch.tensor(int(self.labels[idx]), dtype=torch.long)
        return out


class ViewOnlyDataset(Dataset):
    def __init__(self, views, labels=None):
        self.views = torch.from_numpy(views).float()
        self.labels = labels
        self.img_size = int(np.sqrt(views.shape[1]))

    def __len__(self):
        return self.views.shape[0]

    def __getitem__(self, idx):
        x = self.views[idx].view(1, self.img_size, self.img_size)
        out = {"x": x}
        if self.labels is not None:
            out["label"] = torch.tensor(int(self.labels[idx]), dtype=torch.long)
        return out