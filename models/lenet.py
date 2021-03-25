"""LeNet model for ADDA."""

import torch.nn.functional as F
from torch import nn


class LeNetEncoder(nn.Module):
    """LeNet encoder model for ADDA."""

    def __init__(self):
        """Init LeNet encoder."""
        super(LeNetEncoder, self).__init__()

        self.restored = False

        self.encoder = nn.Sequential(
            # 1st conv layer
            # input [1 x 28 x 28]
            # output [20 x 12 x 12]
            nn.Conv1d(in_channels=272, out_channels=20, kernel_size=5),
            nn.MaxPool1d(kernel_size=2),
            nn.ReLU(),
            # 2nd conv layer
            # input [20 x 12 x 12]
            # output [50 x 4 x 4]
            nn.Conv1d(20, 50, kernel_size=5),
            nn.Dropout(),
            nn.MaxPool1d(kernel_size=2),
            nn.ReLU()
        )
        #self.fc1 = nn.Linear(50 * 4 * 4, 500)

        self.fc1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(50 * 4 * 4, 500)
            #nn.Linear(4096, 5)
        )

    def forward(self, input):
        """Forward the LeNet."""
        conv_out = self.encoder(input)
        feat = self.fc1(conv_out.view(-1, 3072))
        return feat


class LeNetClassifier(nn.Module):
    """LeNet classifier model for ADDA."""

    def __init__(self):
        """Init LeNet encoder."""
        super(LeNetClassifier, self).__init__()
        self.fc2 = nn.Linear(500, 2)

    def forward(self, feat):
        """Forward the LeNet classifier."""
        out = F.dropout(F.relu(feat), training=self.training)
        out = self.fc2(out)
        return out
