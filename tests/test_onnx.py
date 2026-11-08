import os

import numpy as np
import pytest
import torch

onnx = pytest.importorskip("onnx")
ort = pytest.importorskip("onnxruntime")

import compresskit as ck  # noqa: E402


def run_ort(path, x):
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    return sess.run(None, {sess.get_inputs()[0].name: x.numpy()})[0]


def test_export_onnx_int8_qdq(cnn, batches, tmp_path):
    x = batches[0][0]
    q = ck.quantize_ptq(cnn, batches)
    path = ck.export_onnx(q, x, str(tmp_path / "int8.onnx"))
    ops = {n.op_type for n in onnx.load(path).graph.node}
    assert {"QuantizeLinear", "DequantizeLinear", "Conv", "Gemm"} <= ops
    with torch.no_grad():
        y_torch = q(x).numpy()
    y_ort = run_ort(path, x)
    assert y_ort.shape == y_torch.shape
    assert np.abs(y_ort - y_torch).max() < 2e-2  # ORT's integer kernels round a little differently
    assert not any(m.wq.export for _, m in ck.quant_layers(q))  # export mode was left again
    assert os.path.getsize(path) > 0


def test_export_onnx_float_model(cnn, batches, tmp_path):
    x = batches[0][0]
    path = ck.export_onnx(cnn, x, str(tmp_path / "fp32.onnx"))
    with torch.no_grad():
        y_torch = cnn(x).numpy()
    assert np.allclose(run_ort(path, x), y_torch, atol=1e-4)


def test_export_onnx_after_qat_and_shrink(cnn, batches, tmp_path):
    x = batches[0][0]
    ck.prune_filters(cnn, 0.5, scope="layer").detach()
    ck.remove_pruned_channels(cnn, x)
    q = ck.prepare_qat(cnn, batches, method="lsq", w_bits=8, a_bits=8)
    ck.finalize_qat(q)  # learned step sizes are synced into scale / zero_point for export
    path = ck.export_onnx(q, x, str(tmp_path / "qat.onnx"))
    with torch.no_grad():
        y_torch = q(x).numpy()
    assert np.abs(run_ort(path, x) - y_torch).max() < 5e-2
