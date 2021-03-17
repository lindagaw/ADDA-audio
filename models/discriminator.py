"""Discriminator model for ADDA."""

from torch import nn


class Discriminator(nn.Module):
    """Discriminator model for source domain."""

    def __init__(self, input_dims, hidden_dims, output_dims):
        """Init discriminator."""
        super(Discriminator, self).__init__()

        self.restored = False

        self.encoder = nn.Sequential(
            nn.Linear(in_features=4096, out_features=2048),
            nn.Dropout(),
            nn.Linear(2048, 1024),
            nn.Dropout(),
            nn.Linear(1024, 512),
            nn.Dropout(),
            nn.Linear(512, 2),
            nn.LogSoftmax()
        )

    def forward(self, input):
        """Forward the discriminator."""
        """Forward the LeNet."""
        out = self.encoder(input)
        #out = self.fc1(conv_out.view(-1, 3072 * 1))
        return out
