# Custom Models In DrowEEG

DrowEEG currently ships with `eegnet`. Advanced users can register a custom model from Python:

```python
import torch
import droweeg


class MyModel(torch.nn.Module):
    def __init__(self, channels: int, samples: int, num_classes: int):
        super().__init__()
        self.classifier = torch.nn.Linear(channels * samples, num_classes)

    def forward(self, x):
        # x shape: (batch, 1, channels, samples)
        return self.classifier(x.flatten(1))


droweeg.register_model("my_model", MyModel)
model = droweeg.model("my_model", channels=17, samples=1600, num_classes=2)
```

Minimum contract:

- Inherit `torch.nn.Module`.
- Accept input shape `(batch, 1, channels, samples)`.
- Return raw logits with shape `(batch, num_classes)`.

Future source-free adaptation methods may optionally use:

- `get_features(x)`
- `encoder`
- `classifier`

These hooks are documented for future compatibility only. No SFDA method is implemented in this step.
