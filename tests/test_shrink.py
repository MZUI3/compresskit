import torch
import torch.nn as nn

import compresskit as ck
from conftest import SharedConvOut, warm_bn


def silence_dead_channels(conv, bn=None):
    """Zero the bias / BatchNorm shift of all-zero channels so removing them is exact."""
    dead = ~conv.weight.flatten(1).any(1)
    if conv.bias is not None:
        conv.bias.data[dead] = 0
    if bn is not None:
        bn.weight.data[dead] = 0
        bn.bias.data[dead] = 0


def test_shrink_cnn(cnn, batches):
    x = batches[0][0]
    ck.prune_filters(cnn, 0.5, scope="layer").detach()
    silence_dead_channels(cnn.features[0], cnn.features[1])
    silence_dead_channels(cnn.features[3], cnn.features[4])
    y0 = cnn(x)
    macs0 = ck.count_macs(cnn, x)["macs"]
    removed = ck.remove_pruned_channels(cnn, x)
    assert removed == {"features.0": 4, "features.3": 8}
    assert cnn.features[0].weight.shape == (4, 3, 3, 3) and cnn.features[0].out_channels == 4
    assert cnn.features[1].num_features == 4 and cnn.features[1].running_mean.shape == (4,)
    assert cnn.features[3].weight.shape == (8, 4, 3, 3) and cnn.features[3].in_channels == 4
    assert cnn.features[4].num_features == 8
    assert cnn.classifier.weight.shape == (10, 8) and cnn.classifier.in_features == 8  # outputs stay
    assert torch.allclose(cnn(x), y0, atol=1e-5)
    assert ck.count_macs(cnn, x)["macs"] < 0.5 * macs0
    assert ck.count_params(cnn) < 0.5 * 1610


def test_shrink_skips_shared_tensors(batches):
    x = batches[0][0]
    m = warm_bn(SharedConvOut(), batches)
    ck.prune_filters(m, 0.5).detach()
    y0 = m(x)
    assert ck.remove_pruned_channels(m, x) == {}
    assert m.conv.weight.shape == (3, 3, 3, 3)
    assert torch.equal(m(x), y0)


def test_shrink_mlp_with_flatten_block(mlp_batches):
    x = mlp_batches[0][0]
    mlp = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Dropout(0.1), nn.Linear(32, 10)).eval()
    ck.prune_filters(mlp, 0.5, scope="layer").detach()
    silence_dead_channels(mlp[0])
    y0 = mlp(x)
    assert ck.remove_pruned_channels(mlp, x) == {"0": 16}
    assert mlp[0].weight.shape == (16, 16) and mlp[3].weight.shape == (10, 16)
    assert torch.allclose(mlp(x), y0, atol=1e-6)

    # conv -> flatten with a spatial block: each removed channel drops 2x2 = 4 linear columns
    net = nn.Sequential(nn.Conv2d(3, 6, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(2), nn.Flatten(),
                        nn.Linear(6 * 4, 5)).eval()
    img = torch.randn(4, 3, 8, 8)
    ck.prune_filters(net, 0.5, scope="layer").detach()
    silence_dead_channels(net[0])
    y0 = net(img)
    assert ck.remove_pruned_channels(net, img) == {"0": 3}
    assert net[0].weight.shape == (3, 3, 3, 3) and net[4].weight.shape == (5, 12)
    assert torch.allclose(net(img), y0, atol=1e-6)


def test_shrink_quantized_model(cnn, batches):
    x = batches[0][0]
    ck.prune_filters(cnn, 0.5, scope="layer").detach()
    q = ck.quantize_ptq(cnn, batches)
    removed = ck.remove_pruned_channels(q, x)
    assert removed["features.0"] == 4 and removed["features.3"] == 8
    assert q.features[0].wq.scale.numel() == 4 and q.features[3].wq.scale.numel() == 8
    assert torch.isfinite(q(x)).all()


def test_shrink_in_pipeline(cnn, batches):
    x = batches[0][0]
    result = ck.compress(cnn, [("filters", {"ratio": 0.5}), ("shrink", {"example": x})])
    assert result.masks is None and "dropped" in result.steps[1]["masks"]
    assert result.steps[1]["output"] == {"features.0": 4, "features.3": 8}
    assert result.model.features[3].weight.shape == (8, 4, 3, 3)
    assert cnn.features[3].weight.shape == (16, 8, 3, 3)  # the original is untouched
