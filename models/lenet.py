"""LeNet, the source cnn, for ADDA."""

import torch.nn.functional as F
from torch import nn
import torch

class LeNetEncoder(nn.Module):
    """LeNet encoder model for ADDA."""

    def __init__(self):
        """Init LeNet encoder."""
        super(LeNetEncoder, self).__init__()

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
            nn.Conv1d(in_channels=20, out_channels=3072, kernel_size=5),
            nn.Dropout(),
            nn.MaxPool1d(kernel_size=2),
            nn.ReLU(),
        )

        self.fc1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3072, 4096),
            #nn.Linear(4096, 5)
        )

    def forward(self, input):
        """Forward the LeNet."""
        conv_out = self.encoder(input)
        print('--------------')
        print(conv_out.shape)
        conv_out = torch.unsqueeze(torch.mean(self.encoder(input), 2), 2)
        feat = self.fc1(conv_out.view(-1, 3072 * 1))
        print(feat.shape)
        print('--------------')
        return feat


class LeNetClassifier(nn.Module):
    """LeNet classifier model for ADDA."""
    def __init__(self):
        """Init LeNet encoder."""
        super(LeNetClassifier, self).__init__()
        self.fc2 = nn.Linear(4096, 2)
        #self.fc2 = nn.Linear(4096, 5)

    def forward(self, feat):
        """Forward the LeNet classifier."""
        out = F.dropout(F.relu(feat), training=self.training)
        out = self.fc2(out)
        return out
