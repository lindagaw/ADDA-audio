"""Discriminator model for ADDA."""

from torch import nn


class Discriminator(nn.Module):
    """Discriminator model for source domain."""

    def __init__(self, input_dims, hidden_dims, output_dims):
        """Init discriminator."""
        super(Discriminator, self).__init__()

        self.restored = False

        self.layer = nn.Sequential(
            nn.Conv1d(in_channels=4096, out_channels=4096, kernel_size=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=1, stride=2, padding=0),
            nn.Dropout(),
            nn.Flatten(),

            nn.Linear(input_dims, hidden_dims),
            nn.ReLU(),
            nn.Linear(hidden_dims, hidden_dims),
            nn.ReLU(),
            nn.Linear(hidden_dims, output_dims),
            nn.LogSoftmax()
        )

    def forward(self, input):
        """Forward the discriminator."""
        print(input.shape)
        out = self.layer(input)
        return out
