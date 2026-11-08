import copy

import pytest
import torch
import torch.nn as nn

import compresskit as ck
from conftest import SmallCNN, clone, make_batches, warm_bn


def zero_rows(w):
    return int((w.flatten(1) == 0).all(1).sum())


def test_magnitude_global(cnn):
    masks = ck.prune_magnitude(cnn, 0.5)
    assert abs(masks.sparsity() - 0.5) < 0.01
    assert abs(ck.weight_sparsity(cnn) - 0.5) < 0.01
    assert set(masks) == {"features.0.weight", "features.3.weight", "classifier.weight"}
    assert "total" in masks.report()
    for name, w in masks.targets.items():
        assert torch.equal(w == 0, ~masks[name])


def test_magnitude_layer_scope(cnn):
    masks = ck.prune_magnitude(cnn, 0.5, scope="layer")
    for s in masks.layer_sparsity().values():
        assert abs(s - 0.5) < 0.02


def test_kept_weights_are_the_largest(mlp):
    masks = ck.prune_magnitude(mlp, 0.5, scope="layer")
    for name, w in masks.targets.items():
        kept = w[masks[name]].abs()
        pruned_max = w.detach().abs()[~masks[name]].max() if (~masks[name]).any() else 0
        assert kept.min() >= pruned_max or torch.isclose(kept.min(), torch.as_tensor(pruned_max))


def test_exclude_and_include(cnn):
    masks = ck.prune_magnitude(cnn, 0.5, exclude=("first", "last"))
    assert set(masks) == {"features.3.weight"}
    masks = ck.prune_magnitude(clone(cnn), 0.5, include="classifier")
    assert set(masks) == {"classifier.weight"}
    with pytest.raises(KeyError):
        ck.prune_magnitude(clone(cnn), 0.5, exclude="does.not.exist")


def test_iterative_pruning_never_revives(cnn):
    masks = ck.prune_magnitude(cnn, 0.3)
    first = masks.state_dict()
    ck.prune_magnitude(cnn, 0.6, masks=masks)
    assert abs(masks.sparsity() - 0.6) < 0.01
    for name in masks:
        assert not (~first[name] & masks[name]).any()


def test_random_is_reproducible():
    a, b = SmallCNN(), SmallCNN()
    b.load_state_dict(a.state_dict())
    ma = ck.prune_random(a, 0.5, seed=3)
    mb = ck.prune_random(b, 0.5, seed=3)
    for name in ma:
        assert torch.equal(ma[name], mb[name])
    assert abs(ma.sparsity() - 0.5) < 0.01


def test_nm_sparsity(mlp, cnn):
    masks = ck.prune_nm(mlp, 2, 4)
    for name in masks:
        groups = masks[name].reshape(-1, 4).sum(1)
        assert torch.all(groups == 2)
    assert abs(masks.sparsity() - 0.5) < 1e-6
    masks = ck.prune_nm(cnn, 2, 4)
    assert masks["features.0.weight"].all()  # in=3 is not a multiple of 4: left dense
    conv2 = masks["features.3.weight"]  # (16, 8, 3, 3): groups along "in"
    assert torch.all(conv2.movedim(1, -1).reshape(-1, 4).sum(1) == 2)


def test_filter_pruning(cnn):
    masks = ck.prune_filters(cnn, 0.5, criterion="l2", scope="layer")
    assert zero_rows(cnn.features[0].weight) == 4
    assert zero_rows(cnn.features[3].weight) == 8
    assert zero_rows(cnn.classifier.weight) == 5
    assert set(masks) == {"features.0.weight", "features.3.weight", "classifier.weight"}
    masks = ck.prune_filters(clone(cnn), 0.5, criterion="l1", scope="global")
    assert 0.3 < masks.sparsity() < 0.7


def test_filter_pruning_keeps_pruned_channels(cnn):
    masks = ck.prune_filters(cnn, 0.25)
    before = {n: m.flatten(1).any(1).clone() for n, m in masks.masks.items()}
    ck.prune_filters(cnn, 0.5, masks=masks)
    for n, m in masks.masks.items():
        alive = m.flatten(1).any(1)
        assert not (alive & ~before[n]).any()


def test_gradient_based(cnn, batches):
    masks = ck.prune_snip(cnn, batches, 0.5)
    assert abs(masks.sparsity() - 0.5) < 0.01
    cnn2 = warm_bn(SmallCNN(), batches)
    masks = ck.prune_obd(cnn2, batches, 0.5, num_batches=2)
    assert abs(masks.sparsity() - 0.5) < 0.01
    cnn3 = warm_bn(SmallCNN(), batches)
    masks = ck.prune_snip(cnn3, [x for x, _ in batches], 0.5,
                          loss_fn=lambda model, x: model(x).pow(2).mean())
    assert abs(masks.sparsity() - 0.5) < 0.01
    assert not cnn3.training


def test_custom_scores(mlp):
    masks = ck.MaskSet(mlp)
    original = {n: w.detach().abs().clone() for n, w in masks.targets.items()}
    scores = masks.scores(lambda n, w: -w.abs())  # keep the smallest
    ck.prune_by_scores(mlp, scores, 0.5, scope="layer", masks=masks)
    for name, w in masks.targets.items():
        kept = original[name][masks[name]]
        pruned = original[name][~masks[name]]
        assert kept.max() <= pruned.min() + 1e-7
        assert torch.all(w[~masks[name]] == 0)


def test_attach_keeps_zeros_while_training(cnn, batches):
    masks = ck.prune_magnitude(cnn, 0.5).attach()
    cnn.train()
    opt = torch.optim.SGD(cnn.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-2)
    for x, y in batches * 3:
        loss = nn.functional.cross_entropy(cnn(x), y)
        opt.zero_grad()
        loss.backward()
        for name, w in masks.targets.items():
            assert torch.all(w.grad[~masks[name]] == 0)
        opt.step()
    cnn.eval()
    cnn(batches[0][0])  # the pre-forward hook re-applies the masks
    for name, w in masks.targets.items():
        assert torch.all(w[~masks[name]] == 0)
    assert abs(ck.sparsity_report(cnn)["total"] - 0.5) < 0.01
    masks.detach()
    assert not masks.attached


def test_deepcopy_of_attached_model(cnn, batches):
    masks = ck.prune_magnitude(cnn, 0.5).attach()
    copied = copy.deepcopy(cnn)
    copied(batches[0][0])
    assert abs(ck.weight_sparsity(copied) - 0.5) < 0.01
    masks.detach()


def test_gradual_pruner(mlp, mlp_batches):
    mlp.train()
    opt = torch.optim.SGD(mlp.parameters(), lr=0.05, momentum=0.9)
    total = 60
    pruner = ck.GradualPruner(mlp, final_sparsity=0.8, total_steps=total, update_every=5, end=0.8)
    seen = []
    for step in range(total):
        x, y = mlp_batches[step % len(mlp_batches)]
        loss = nn.functional.cross_entropy(mlp(x), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        pruner.step()
        seen.append(pruner.masks.sparsity())
    masks = pruner.finish()
    assert abs(masks.sparsity() - 0.8) < 0.02
    assert all(b >= a - 1e-9 for a, b in zip(seen, seen[1:]))
    assert 0 < seen[len(seen) // 2] < 0.8
    assert ck.cubic_schedule(0, 100, 0.9) == 0.0 and ck.cubic_schedule(100, 100, 0.9) == 0.9


def test_state_dict_roundtrip(cnn):
    original = clone(cnn)
    masks = ck.prune_magnitude(cnn, 0.6)
    state = masks.state_dict()
    restored = ck.MaskSet(original).load_state_dict(state)
    assert abs(restored.sparsity() - 0.6) < 0.01
    for name in masks:
        assert torch.equal(restored.targets[name] == 0, cnn.get_parameter(name) == 0)


def test_multihead_attention(attention):
    masks = ck.prune_magnitude(attention, 0.5)
    assert set(masks) == {"attn.in_proj_weight", "attn.out_proj.weight", "fc.weight"}
    out = attention(torch.randn(2, 5, 8))
    assert out.shape == (2, 4) and torch.isfinite(out).all()


def test_registry():
    assert {"magnitude", "random", "nm", "filters", "snip", "obd"} <= set(ck.available("prune"))
    assert ck.get("magnitude").fn is ck.prune_magnitude
    with pytest.raises(KeyError, match="Did you mean"):
        ck.get("magnitud")
    assert "magnitude" in ck.describe("prune")


def test_sparsity_report_format(cnn):
    ck.prune_magnitude(cnn, 0.5)
    report = ck.sparsity_report(cnn)
    assert abs(report["total"] - 0.5) < 0.01
    text = ck.format_sparsity(report)
    assert "classifier.weight" in text and "total" in text
