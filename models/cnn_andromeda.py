import torch.nn.functional as F
from torch import nn

class AndromedaEncoder(nn.Module):
    """Andromeda encoder model for ADDA."""

    def __init__(self):
        """Init LeNet encoder."""
        super(AndromedaEncoder, self).__init__()

        self.restored = False

        self.encoder = nn.Sequential(
            # 1st conv layer
            nn.Conv1d(in_channels=272, out_channels=512, kernel_size=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=0),
            nn.Dropout(),

            # 2nd conv layer
            nn.Conv1d(in_channels=512, out_channels=256, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=0),
            nn.Dropout(),

            # 3rd conv layer
            nn.Conv1d(in_channels=256, out_channels=128, kernel_size=1, stride=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=1, stride=1, padding=0),
            nn.Dropout(),

            # 4th conv layer
            nn.Conv1d(in_channels=128, out_channels=64, kernel_size=1, stride=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=1, stride=1, padding=0),
            nn.Dropout(),
        )

        self.fc1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.Dropout(),
            nn.Linear(128, 64),
            nn.Dropout()
        )

    def forward(self, input):
        """Forward the LeNet."""
        conv_out = self.encoder(input)
        feat = self.fc1(conv_out.view(-1, 3072 * 1))
        return feat


class AndromedaClassifier(nn.Module):
    """LeNet classifier model for ADDA."""
    def __init__(self):
        """Init LeNet encoder."""
        super(AndromedaClassifier, self).__init__()
        self.fc2 = nn.Linear(64, 2)

    def forward(self, feat):
        """Forward the LeNet classifier."""
        out = F.dropout(F.relu(feat), training=self.training)
        out = self.fc2(out)
        return out
