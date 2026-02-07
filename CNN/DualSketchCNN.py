import torch.nn as nn
import SketchCNN

# ============================================================================
# DUAL MODEL (two separate CNNs)
# ============================================================================

class DualSketchCNN(nn.Module):
    """Two independent CNNs for sketch and view streams"""

    def __init__(self, code_len=64, use_dropout=False, dropout_p=0.5):
        super().__init__()
        self.sketch_net = SketchCNN(code_len, use_dropout, dropout_p)
        self.view_net = SketchCNN(code_len, use_dropout, dropout_p)

    def encode_sketch(self, sketch):
        return self.sketch_net(sketch)

    def encode_view(self, view):
        return self.view_net(view)

    # Optional: a forward that returns both (nice for debugging)
    def forward(self, sketch, view):
        return self.encode_sketch(sketch), self.encode_view(view)