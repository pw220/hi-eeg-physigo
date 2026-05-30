from __future__ import annotations

import torch


class DrowEEGModel(torch.nn.Module):
    """Base documentation class for DrowEEG models.

    Model contract:
    - input shape: (batch, 1, channels, samples)
    - output shape: (batch, num_classes) raw logits
    - optional future SFDA hooks: get_features(x), encoder, classifier
    """

