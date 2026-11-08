import torch
import torch.nn as nn

import compresskit as ck
from conftest import clone


def test_counts(cnn, batches):
    x = batches[0][0]
    assert ck.count_params(cnn) == sum(p.numel() for p in cnn.parameters())
    # BatchNorm biases start at zero, so they are the only zeros of a fresh model
    assert ck.count_nonzero(cnn) == ck.count_params(cnn) - 8 - 16
    macs = ck.count_macs(cnn, x)
    # conv1: 64 positions x 8 out x 27; conv2: 64 x 16 x 72; classifier: 10 x 16
    assert macs["macs"] == 64 * 8 * 27 + 64 * 16 * 72 + 10 * 16
    assert macs["sparse_macs"] == macs["macs"]
    ck.prune_magnitude(cnn, 0.5)
    sparse = ck.count_macs(cnn, (x, None))
    assert sparse["macs"] == macs["macs"] and sparse["sparse_macs"] < 0.6 * macs["macs"]


def test_macs_attention(attention):
    macs = ck.count_macs(attention, torch.randn(2, 5, 8))
    assert macs["macs"] > 0


def test_sizes(cnn, batches):
    fp32 = ck.state_dict_bytes(cnn)
    assert ck.estimated_bytes(cnn, sparse=False) == fp32
    assert ck.estimated_bytes(cnn) <= fp32  # the zero BatchNorm biases are counted as a 1-bit mask
    pruned = clone(cnn)
    ck.prune_magnitude(pruned, 0.5)
    assert ck.state_dict_bytes(pruned) == fp32
    assert ck.estimated_bytes(pruned) < 0.75 * fp32
    q = ck.quantize_ptq(cnn, batches)
    assert ck.estimated_bytes(q) < 0.5 * fp32
    wo = ck.quantize_weight_only(cnn, 8, layers=(nn.Linear, nn.Conv2d))
    assert ck.state_dict_bytes(wo) < 0.5 * fp32
    assert ck.estimated_bytes(wo, sparse=False) == ck.state_dict_bytes(wo)


def test_latency_summary_compare(cnn, batches):
    x = batches[0][0]
    ms = ck.latency_ms(cnn, x, iters=3, warmup=1)
    assert isinstance(ms, float) and ms > 0
    s = ck.summary(cnn, x)
    assert {"params", "nonzero_params", "weight_sparsity", "state_dict_mb", "estimated_mb", "macs",
            "sparse_macs"} <= set(s)
    q = ck.quantize_ptq(cnn, batches)
    table = ck.compare({"fp32": cnn, "int8": q}, example=x, latency=True, iters=2, warmup=1)
    assert "fp32" in table and "int8" in table and "latency_ms" in table
    ck.print_summary(cnn)


def test_compress_pipeline(cnn, batches):
    result = ck.compress(cnn, [("magnitude", {"sparsity": 0.5}), ("ptq", {"calib_batches": batches})])
    assert abs(result.masks.sparsity() - 0.5) < 0.01
    assert ck.quant_summary(result.model)["quantized_layers"] == 3
    assert result.masks.model is result.model
    assert ck.weight_sparsity(cnn) == 0.0  # the original is untouched
    assert [s["method"] for s in result.steps] == ["magnitude", "ptq"]
    result = ck.compress(cnn, [{"method": "filters", "ratio": 0.5}, ("nm", {})], verbose=True)
    assert result.masks.sparsity() > 0.5


def test_custom_registered_method(mlp):
    @ck.register("keep_largest_rows_test", kind="prune")
    def keep_largest_rows(model, ratio, *, masks=None):
        return ck.prune_filters(model, ratio, criterion="l1", masks=masks)

    result = ck.compress(mlp, [("keep_largest_rows_test", {"ratio": 0.5})])
    assert result.masks.sparsity() > 0.4
    assert "keep_largest_rows_test" in ck.available("prune")
