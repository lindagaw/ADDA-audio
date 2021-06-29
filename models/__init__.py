from .discriminator import Discriminator
from .lenet import LeNetClassifier, LeNetEncoder
from .lenet2 import LeNet2Classifier, LeNet2Encoder
from .cnn_galatea import GalateaEncoder, GalateaClassifier
from .cnn_andromeda import AndromedaEncoder, AndromedaClassifier

from .auriel import AurielEncoder, AurielClassifier
from .beatrice import BeatriceEncoder, BeatriceClassifier
from .ciel import CielEncoder, CielClassifier
from .dione import DioneEncoder, DioneClassifier
from .essentia import EssentiaEncoder, EssentiaClassifier


__all__ = (GalateaClassifier, GalateaEncoder, AndromedaEncoder, AndromedaClassifier, \
            Discriminator, \
            AurielEncoder, AurielClassifier, BeatriceEncoder, BeatriceClassifier, \
            CielEncoder, CielClassifier, DioneEncoder, DioneClassifier, \
            EssentiaEncoder, EssentiaClassifier)
