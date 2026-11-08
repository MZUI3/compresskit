import copy

import pytest
import torch
import torch.nn as nn


class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 8, 3, padding=1), nn.BatchNorm2d(8), nn.ReLU(inplace=True),
            nn.Conv2d(8, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        )
        self.classifier = nn.Linear(16, 10)

    def forward(self, x):
        return self.classifier(self.features(x))


class SharedConvOut(nn.Module):
    """The conv output feeds the BatchNorm and a residual add: BatchNorm must not be folded."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 3, 3, padding=1)
        self.bn = nn.BatchNorm2d(3)

    def forward(self, x):
        y = self.conv(x)
        return self.bn(y) + y


class TinyAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = nn.MultiheadAttention(8, 2, batch_first=True)
        self.fc = nn.Linear(8, 4)

    def forward(self, x):
        y, _ = self.attn(x, x, x)
        return self.fc(y.mean(1))


def make_batches(n=4, bs=8, shape=(3, 8, 8), classes=10, seed=0):
    g = torch.Generator().manual_seed(seed)
    return [(torch.randn(bs, *shape, generator=g), torch.randint(0, classes, (bs,), generator=g)) for _ in range(n)]


def warm_bn(model, batches):
    """Run a few batches in train mode so BatchNorm has non-trivial running statistics."""
    model.train()
    with torch.no_grad():
        for x, _ in batches:
            model(x)
    model.eval()
    return model


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)


@pytest.fixture
def batches():
    return make_batches()


@pytest.fixture
def cnn(batches):
    return warm_bn(SmallCNN(), batches)


@pytest.fixture
def mlp():
    return nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 10)).eval()


@pytest.fixture
def mlp_batches():
    return make_batches(shape=(16,))


@pytest.fixture
def attention():
    return TinyAttention().eval()


def rel_error(a, b):
    return float((a - b).norm() / b.norm().clamp_min(1e-12))


def clone(model):
    return copy.deepcopy(model)
