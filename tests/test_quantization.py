import copy

import pytest
import torch
import torch.nn as nn

import compresskit as ck
from compresskit.quantization.layers import QUANT_TYPES
from conftest import SharedConvOut, SmallCNN, clone, rel_error, warm_bn


def has_type(model, types):
    return any(isinstance(m, types) for m in model.modules())


# ----------------------------------------------------------------------------- BatchNorm folding

def test_fold_bn_preserves_output(cnn, batches):
    x = batches[0][0]
    y0 = cnn(x)
    assert ck.fold_bn(cnn, x) == 2
    assert not has_type(cnn, nn.BatchNorm2d)
    assert torch.allclose(cnn(x), y0, atol=1e-4)


def test_fold_bn_skips_shared_tensors(batches):
    m = warm_bn(SharedConvOut(), batches)
    assert ck.fold_bn(m, batches[0][0]) == 0
    assert has_type(m, nn.BatchNorm2d)


def test_fold_bn_linear_bn1d(mlp_batches):
    m = nn.Sequential(nn.Linear(16, 32), nn.BatchNorm1d(32), nn.ReLU(), nn.Linear(32, 4))
    warm_bn(m, mlp_batches)
    x = mlp_batches[0][0]
    y0 = m(x)
    assert ck.fold_bn(m, (x, None)) == 1
    assert torch.allclose(m(x), y0, atol=1e-4)


# ----------------------------------------------------------------------------- PTQ

def test_ptq_close_to_float(cnn, batches):
    x = batches[0][0]
    y = cnn(x)
    q = ck.quantize_ptq(cnn, batches)
    assert not has_type(cnn, QUANT_TYPES)  # inplace=False leaves the original alone
    info = ck.quant_summary(q)
    assert info["quantized_layers"] == 3 and info["weight_bits"] == [8] and info["act_bits"] == [8]
    assert not has_type(q, nn.BatchNorm2d)
    assert rel_error(q(x), y) < 0.1
    assert all(m.aq.mode == "quant" and m.wq.mode == "quant" for _, m in ck.quant_layers(q))


def test_ptq_exclude_and_inplace(cnn, batches):
    q = ck.quantize_ptq(cnn, batches, exclude=("first", "last"), inplace=True)
    assert q is cnn
    assert ck.quant_summary(q)["layers"] == ["features.3"]


def test_ptq_lazy_calibration(cnn, batches):
    x = batches[0][0]
    y = cnn(x)
    q = ck.quantize_ptq(cnn, None, fold_bn=False)
    assert all(m.aq.mode == "init" for _, m in ck.quant_layers(q))
    out = q(x)
    assert all(m.aq.mode == "quant" for _, m in ck.quant_layers(q))
    assert rel_error(out, y) < 0.1


@pytest.mark.parametrize("observer", ["percentile", "mse", "kl", "ema"])
def test_ptq_observers(cnn, batches, observer):
    x = batches[0][0]
    q = ck.quantize_ptq(cnn, batches, a_observer=observer, w_observer="mse" if observer == "mse" else "minmax",
                        a_symmetric=observer == "kl")
    assert rel_error(q(x), cnn(x)) < 0.2


def test_ptq_low_bits_and_weight_only_fake(cnn, batches):
    x = batches[0][0]
    q4 = ck.quantize_ptq(cnn, batches, w_bits=4, a_bits=4)
    assert torch.isfinite(q4(x)).all()
    assert ck.quant_summary(q4)["weight_bits"] == [4]
    qw = ck.quantize_ptq(cnn, batches, w_bits=8, a_bits=32)
    assert all(isinstance(m.aq, ck.NoQuant) for _, m in ck.quant_layers(qw))
    assert rel_error(qw(x), cnn(x)) < 0.05


def test_strip_quantizers(cnn, batches):
    q = ck.quantize_ptq(cnn, batches)
    baked = q.classifier.wq(q.classifier.weight).detach().clone()
    s = ck.strip_quantizers(q)
    assert not has_type(s, QUANT_TYPES)
    assert type(s.classifier) is nn.Linear and type(s.features[0]) is nn.Conv2d
    assert torch.allclose(s.classifier.weight, baked)
    assert set(s.state_dict()) <= set(cnn.state_dict()) | {"features.0.bias", "features.3.bias"}
    assert has_type(q, QUANT_TYPES)  # the quantized model is untouched


def test_export_mode_matches_quantize(cnn, batches):
    x = batches[0][0]
    q = ck.quantize_ptq(cnn, batches)
    y = q(x)
    with ck.export_mode(q):
        assert all(m.wq.export and m.aq.export for _, m in ck.quant_layers(q))
        y_export = q(x)
    assert not any(m.wq.export for _, m in ck.quant_layers(q))
    assert torch.allclose(y, y_export, atol=1e-4)


def test_state_dict_reload(cnn, batches):
    x = batches[0][0]
    fresh = clone(cnn)
    q = ck.quantize_ptq(cnn, batches)
    ck.fold_bn(fresh, x)
    ck.attach_quantizers(fresh, 8, 8)
    fresh.load_state_dict(q.state_dict())
    assert all(m.aq.mode == "quant" for _, m in ck.quant_layers(fresh))
    assert torch.allclose(fresh(x), q(x))


def test_prune_then_quantize_keeps_zeros(cnn, batches):
    masks = ck.prune_magnitude(cnn, 0.5)
    q = ck.quantize_ptq(cnn, batches)
    for name, m in ck.quant_layers(q):
        w_q = m.wq(m.weight)
        assert torch.all(w_q[~masks[f"{name}.weight"]] == 0)
    assert abs(ck.weight_sparsity(q) - 0.5) < 0.01


# ----------------------------------------------------------------------------- QAT

def train_steps(model, batches, params=None, steps=3, lr=0.01):
    model.train()
    opt = torch.optim.SGD(params if params is not None else model.parameters(), lr=lr)
    losses = []
    for x, y in batches[:steps]:
        loss = nn.functional.cross_entropy(model(x), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss))
    return losses


def test_qat_ste(cnn, batches):
    x = batches[0][0]
    q = ck.prepare_qat(cnn, batches, method="ste")
    assert has_type(q, nn.BatchNorm2d)  # kept by default
    before = q.classifier.weight.detach().clone()
    losses = train_steps(q, batches)
    assert all(torch.isfinite(torch.tensor(losses)))
    assert not torch.equal(before, q.classifier.weight)  # gradients flow through the STE
    assert isinstance(q.classifier.wq, ck.DynamicWeightQuantizer)
    f = ck.finalize_qat(q)
    assert not f.training and f is q
    assert all(not m.aq.ema_training for _, m in ck.quant_layers(f))
    assert torch.isfinite(f(x)).all()
    assert rel_error(f(x), cnn(x)) < 1.0


@pytest.mark.parametrize("method,per_layer", [("lsq", 2), ("lsq+", 3)])
def test_qat_lsq(cnn, batches, method, per_layer):
    q = ck.prepare_qat(cnn, batches, method=method, w_bits=4, a_bits=4)
    qparams = ck.quantizer_parameters(q)
    assert len(qparams) == 3 * per_layer
    assert len(ck.model_parameters(q)) + len(qparams) == len(list(q.parameters()))
    groups = [{"params": ck.model_parameters(q)}, {"params": qparams, "weight_decay": 0.0}]
    train_steps(q, batches, params=groups)
    assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in qparams)
    ck.finalize_qat(q)
    aq = q.classifier.aq
    assert torch.allclose(aq.scale, aq.s.detach().abs())
    assert torch.isfinite(q(batches[0][0])).all()


def test_qat_lazy_init(mlp, mlp_batches):
    q = ck.prepare_qat(mlp, None, method="lsq")
    assert ck.quantizer_parameters(q) == [] or all(isinstance(p, nn.Parameter) for p in ck.quantizer_parameters(q))
    q(mlp_batches[0][0])
    assert len(ck.quantizer_parameters(q)) == 4  # 2 layers x (weight step + activation step)
    with pytest.raises(ValueError):
        ck.prepare_qat(mlp, None, method="nope")


# ----------------------------------------------------------------------------- weight-only

@pytest.mark.parametrize("bits", [2, 4, 8])
@pytest.mark.parametrize("symmetric", [True, False])
@pytest.mark.parametrize("group_size", [None, 4])
def test_quantize_matrix_roundtrip(bits, symmetric, group_size):
    w = torch.randn(8, 16)
    w[0, :4] = 0.0
    packed, scale, zp = ck.quantization.quantize_matrix(w, bits, group_size, symmetric)
    assert packed.dtype == torch.uint8 and packed.shape == (8, 16 * bits // 8)
    deq = ck.quantization.dequantize_matrix(packed, scale, zp, bits, 16, group_size)
    step = scale.max()
    assert (deq - w).abs().max() <= step / 2 + 1e-5
    assert torch.all(deq[0, :4] == 0)


def test_weight_only_int8(mlp, mlp_batches):
    x = mlp_batches[0][0]
    y = mlp(x)
    q = ck.quantize_weight_only(mlp, 8)
    assert isinstance(q[0], ck.WeightOnlyLinear) and isinstance(mlp[0], nn.Linear)
    assert rel_error(q(x), y) < 0.05
    assert q[0].qweight.dtype == torch.uint8 and q[0].qweight.shape == (32, 16)
    assert ck.state_dict_bytes(q) < 0.4 * ck.state_dict_bytes(mlp)
    assert q[0].weight.shape == (32, 16)
    assert ck.count_params(q) == ck.count_params(mlp)  # the packed weights still count as parameters
    assert ck.count_params(q, trainable_only=True) == 32 + 10  # only the biases are trainable


def test_weight_only_int4_groups(mlp, mlp_batches):
    x = mlp_batches[0][0]
    q = ck.quantize_weight_only(mlp, 4, group_size=8)
    assert q[0].qweight.shape == (32, 8) and q[0].scale.shape == (32, 2) and q[0].zero_point.shape == (32, 2)
    assert rel_error(q(x), mlp(x)) < 0.3
    q2 = ck.quantize_weight_only(mlp, 2)
    assert torch.isfinite(q2(x)).all()
    with pytest.warns(UserWarning):
        ck.quantize_weight_only(mlp, 4, group_size=5)


def test_weight_only_conv(cnn, batches):
    x = batches[0][0]
    q = ck.quantize_weight_only(cnn, 8, layers=(nn.Linear, nn.Conv2d))
    assert isinstance(q.features[0], ck.WeightOnlyConv)
    assert rel_error(q(x), cnn(x)) < 0.05
    assert q.features[3].weight.shape == (16, 8, 3, 3)
    q = ck.quantize_weight_only(cnn, 8, layers=(nn.Linear, nn.Conv2d), exclude="first")
    assert isinstance(q.features[0], nn.Conv2d)


# ----------------------------------------------------------------------------- torch backend

def test_dynamic_int8(mlp, mlp_batches):
    x = mlp_batches[0][0]
    q = ck.quantize_dynamic_int8(mlp)
    assert "dynamic" in type(q[0]).__module__
    assert rel_error(q(x), mlp(x)) < 0.1
    assert isinstance(mlp[0], nn.Linear)
