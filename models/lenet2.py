"""LeNet2, the source cnn, for ADDA."""

import torch.nn.functional as F
from torch import nn
import torch

class LeNet2Encoder(nn.Module):
    """LeNet2 encoder model for ADDA."""

    def __init__(self):
        """Init LeNet2 encoder."""
        super(LeNet2Encoder, self).__init__()

        self.restored = False

        self.encoder = nn.Sequential(
            # 1st conv layer
            # input [48 x 272]
            # output [46 x 3072]
            nn.Conv1d(in_channels=272, out_channels=20, kernel_size=5),
            nn.MaxPool1d(kernel_size=2),
            nn.ReLU(),
            #nn.Dropout(),

            # 4th conv layer
            # input [4, 6144]
            # output [2, 3072]
            nn.Conv1d(in_channels=20, out_channels=50, kernel_size=5),
            nn.Dropout(),
            nn.MaxPool1d(kernel_size=2),
            nn.ReLU(),
        )

        self.fc1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(50, 50),
            #nn.Linear(4096, 5)
        )

    def forward(self, input):
        """Forward the LeNet2."""
        conv_out = self.encoder(input)
        conv_out = torch.unsqueeze(torch.mean(self.encoder(input), 2), 2)
        feat = self.fc1(conv_out.view(conv_out.shape))
        return feat


class LeNet2Classifier(nn.Module):
    """LeNet2 classifier model for ADDA."""
    def __init__(self):
        """Init LeNet2 encoder."""
        super(LeNet2Classifier, self).__init__()
        self.fc2 = nn.Linear(50, 2)
        #self.fc2 = nn.Linear(4096, 5)

    def forward(self, feat):
        """Forward the LeNet2 classifier."""
        out = F.dropout(F.relu(feat), training=self.training)
        out = self.fc2(out)
        return out
