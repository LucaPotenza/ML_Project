import torch.nn as nn
import torch.nn.functional as F

# ============================================================================
# CNN MODEL
# ============================================================================

class SketchCNN(nn.Module):
    """CNN equivalent to the Theano model"""

    def __init__(self, code_len=64, use_dropout=False, dropout_p=0.5):
        super().__init__()

        self.use_dropout = use_dropout

        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=13)
        self.pool1 = nn.MaxPool2d(kernel_size=4)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=7)
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.conv3 = nn.Conv2d(64, 256, kernel_size=3)
        self.pool3 = nn.MaxPool2d(kernel_size=2)

        # Final fully connected layer
        # Make model resolution-agnostic:
        self.adapt = nn.AdaptiveAvgPool2d((3, 3))
        self.fc = nn.Linear(256 * 3 * 3, code_len)


        # Optional dropout
        self.dropout = nn.Dropout(p=dropout_p)

        # Xavier initialization
        self._init_weights()

    def _init_weights(self):
        """Xavier initialization identical to the Theano version"""
        layers = [self.conv1, self.conv2, self.conv3, self.fc]
        for layer in layers:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, x):
        # Layer 1: Conv → Pool → ReLU
        x = F.relu(self.pool1(self.conv1(x)))

        # Layer 2: Conv → Pool → ReLU
        x = F.relu(self.pool2(self.conv2(x)))

        # Layer 3: Conv → Pool → ReLU
        x = F.relu(self.pool3(self.conv3(x)))
        x = self.adapt(x)  # ensures fixed 3x3 regardless of input resolution


        # Flatten
        x = x.view(x.size(0), -1)

        # Fully connected + ReLU (as per final_active_func='relu')
        x = F.relu(self.fc(x))

        # Dropout only if requested and during training
        if self.use_dropout and self.training:
            x = self.dropout(x)

        return x