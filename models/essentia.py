"""Essentia, the source cnn, for ADDA."""

import torch.nn.functional as F
from torch import nn


class EssentiaEncoder(nn.Module):
    """Essentia encoder model for ADDA."""

    def __init__(self):
        """Init LeNet encoder."""
        super(EssentiaEncoder, self).__init__()

        self.restored = False

        self.encoder = nn.Sequential(

            nn.Conv1d(in_channels=3072, out_channels=3072, kernel_size=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=0),
            nn.Dropout(),

            nn.Conv1d(in_channels=3072, out_channels=6144, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=0),
            nn.Dropout(),

            nn.Conv1d(in_channels=6144, out_channels=6144, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=0),
            nn.Dropout(),

            nn.Conv1d(in_channels=6144, out_channels=3072, kernel_size=1, stride=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=1, stride=1, padding=0),
            nn.Dropout(),
        )

        self.fc1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3072, 4096),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.Dropout()
            #nn.Linear(4096, 5)
        )

    def forward(self, input):
        """Forward the LeNet."""
        conv_out = self.encoder(input)
        feat = self.fc1(conv_out.view(-1, 3072 * 1))
        return feat


class EssentiaClassifier(nn.Module):
    """LeNet classifier model for ADDA."""
    def __init__(self):
        """Init LeNet encoder."""
        super(EssentiaClassifier, self).__init__()
        #self.fc2 = nn.Linear(4096, 2)
        self.fc2 = nn.Linear(4096, 2)

    def forward(self, feat):
        """Forward the LeNet classifier."""
        out = F.dropout(F.relu(feat), training=self.training)
        out = self.fc2(out)
        return out
