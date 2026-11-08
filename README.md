# compresskit

English | [한국어](README.ko.md)

Pruning and quantization for PyTorch models, as plain functions. Install it from GitHub, import it, call it on any `nn.Module`. There are no configs, no training harness and no datasets: every function takes a model (plus a few calibration batches where the method needs them) and returns the result.

```python
import compresskit as ck

masks  = ck.prune_magnitude(model, sparsity=0.5)       # zero the smallest 50% of the weights, in place
qmodel = ck.quantize_ptq(model, calib_batches)         # int8 post-training quantization (a copy)
print(ck.compare({"fp32": model, "int8": qmodel}, example=calib_batches[0]))
```

## Install

```bash
pip install git+https://github.com/MZUI3/compresskit.git
```

For development:

```bash
git clone https://github.com/MZUI3/compresskit.git
cd compresskit
pip install -e ".[dev]"
pytest
```

Python 3.9+ and PyTorch 2.0+. Everything runs on whatever device the model is on (CPU or CUDA); the one exception is `quantize_dynamic_int8`, which uses PyTorch's CPU int8 kernels.

## Pruning

| Function | What it does | Reference |
|---|---|---|
| `prune_magnitude(model, sparsity, scope="global")` | Zero the weights with the smallest \|w\| | Han et al., 2015 |
| `prune_random(model, sparsity, seed=None)` | Random baseline | |
| `prune_snip(model, batches, sparsity)` | Score \|w · ∂L/∂w\| on a few batches; works at initialisation too | Lee et al., 2019 |
| `prune_obd(model, batches, sparsity)` | Optimal Brain Damage with the empirical Fisher diagonal | LeCun et al., 1990 |
| `prune_filters(model, ratio, criterion="l2")` | Zero whole output channels / filters by L1 or L2 norm | Li et al., 2017 |
| `prune_nm(model, n=2, m=4)` | N:M semi-structured sparsity (2:4 runs on Ampere+ sparse tensor cores) | Mishra et al., 2021 |
| `prune_by_scores(model, scores, sparsity)` | Your own importance scores `{param name: tensor}` | |
| `GradualPruner(model, final_sparsity, total_steps)` | Gradual magnitude pruning on the cubic schedule while you train | Zhu & Gupta, 2017 |

Every function prunes **in place** and returns a `MaskSet`: one boolean mask per Conv / Linear / MultiheadAttention weight.

```python
masks = ck.prune_magnitude(model, 0.8, scope="global", exclude=("first", "last"))
masks.sparsity()            # 0.8
print(masks.report())       # per-layer table

masks.attach()              # keep the zeros while fine-tuning: gradients are masked and the
for x, y in loader:         # masks are re-applied before every forward pass
    loss = criterion(model(x), y); loss.backward(); optimizer.step(); optimizer.zero_grad()
masks.detach()

torch.save({"model": model.state_dict(), "masks": masks.state_dict()}, "pruned.pt")
masks = ck.MaskSet(model).load_state_dict(torch.load("pruned.pt")["masks"])   # later
```

- `scope="global"` ranks all layers with one threshold; `"layer"` prunes every layer to the same sparsity.
- `exclude` / `include` take `"first"`, `"last"`, module names or prefixes (`"backbone.layer1"`).
- Iterative pruning: pass the `MaskSet` back in (`ck.prune_magnitude(model, 0.9, masks=masks)`); pruned weights never come back.
- Gradual pruning during training:

```python
pruner = ck.GradualPruner(model, final_sparsity=0.8, total_steps=epochs * len(loader), update_every=100)
for x, y in loader:
    ...; optimizer.step(); optimizer.zero_grad()
    pruner.step()
masks = pruner.finish()
```

Masked weights keep their shape, so unstructured and filter pruning make the model smaller to store (`estimated_bytes`) but not faster on dense hardware. To physically remove the channels that `prune_filters` zeroed, hand the masked model to a dependency-graph tool such as Torch-Pruning.

## Quantization

| Function | What it does | Result |
|---|---|---|
| `quantize_ptq(model, calib_batches, w_bits=8, a_bits=8)` | Fold BatchNorm, fake-quantize weights and layer inputs, calibrate the ranges on a few batches (`minmax`, `percentile`, `mse`, `kl` observers) | simulated |
| `prepare_qat(model, calib_batches, method="ste" \| "lsq" \| "lsq+")` → train → `finalize_qat(model)` | Quantization-aware training with the straight-through estimator, LSQ or LSQ+ | simulated |
| `quantize_weight_only(model, bits=8 \| 4 \| 2, group_size=None)` | Integer weights in packed `uint8` buffers, dequantized in `forward` | real storage, any device |
| `quantize_dynamic_int8(model)` | PyTorch dynamic int8 quantization of Linear / LSTM / GRU | real int8 kernels, CPU |

*Simulated* means the weights and activations are rounded to the integer grid but stored and computed in float: you get exactly the accuracy an int8 (or int4) deployment would see, on any device, and the fake quantizers export to ONNX as QuantizeLinear/DequantizeLinear nodes (`export_onnx`). Real integer kernels are the job of the runtime you deploy with (ONNX Runtime, TensorRT, ...).

```python
# PTQ: a few batches of training data are enough
q = ck.quantize_ptq(model, calib_batches, w_bits=8, a_bits=8, a_observer="percentile")
ck.quant_summary(q)                      # {'quantized_layers': 20, 'weight_bits': [8], 'act_bits': [8], ...}
ck.export_onnx(q, calib_batches[0], "model_int8.onnx")      # needs the onnx package
plain = ck.strip_quantizers(q)           # plain nn.Linear / nn.Conv again, weights kept at their quantized values

# QAT: LSQ at 4 bits
q = ck.prepare_qat(model, calib_batches, method="lsq", w_bits=4, a_bits=4)
optimizer = torch.optim.SGD([
    {"params": ck.model_parameters(q), "lr": 1e-3},
    {"params": ck.quantizer_parameters(q), "lr": 1e-4, "weight_decay": 0.0},   # the learned step sizes
])
...train...
ck.finalize_qat(q)                       # freeze the quantizers, eval mode

# weight-only int4 with group-wise scales: 8x smaller weights, runs everywhere
q = ck.quantize_weight_only(model, bits=4, group_size=128)
```

Lower-level pieces are available too: `fold_bn`, `attach_quantizers`, `init_weight_quantizers`, `calibrate`, `set_mode`, `quantizers`, `export_mode`, and the `Quantizer`, `LSQQuantizer`, `Observer` classes. Quantized layers keep the original parameters and state_dict keys (`fc.weight` stays `fc.weight`) and add `fc.wq.*` / `fc.aq.*` buffers; to reload a saved quantized model, rebuild the float model, call `fold_bn` and `attach_quantizers` with the same settings, then `load_state_dict`.

## Measuring

```python
ck.summary(model, example)        # params, non-zeros, weight sparsity, state_dict MB, estimated MB, MACs
ck.compare({"fp32": model, "pruned": pruned, "int8": q}, example, latency=True)   # markdown table
ck.count_params(model); ck.weight_sparsity(model); ck.count_macs(model, example)
ck.estimated_bytes(model)         # quantized weights at their bit width, pruned tensors as non-zeros + 1-bit mask
ck.latency_ms(model, example)     # median forward time on the model's device
```

## Pipelines and your own methods

```python
result = ck.compress(model, [
    ("magnitude", {"sparsity": 0.5}),
    ("ptq", {"calib_batches": calib_batches}),
])
result.model, result.masks, result.steps

@ck.register("my_prune", kind="prune")
def prune_mine(model, sparsity, *, masks=None, **kwargs):
    masks = masks or ck.MaskSet(model)
    return masks.prune_by_scores(masks.scores(lambda name, w: w.abs()), sparsity)

ck.available()        # all registered names
print(ck.describe())  # one line per method
```

## Layout

```
src/compresskit/
  pruning/        masks.py (MaskSet), unstructured.py, structured.py, nm.py, gradual.py
  quantization/   observers.py, quantizer.py, layers.py, fold.py, ptq.py, qat.py, weight_only.py, torch_backend.py
  utils/          stats.py (params, MACs, bytes, summary, compare), benchmark.py (latency)
  registry.py     @register / get / available
  pipeline.py     compress()
tests/            pytest suite (small CPU models, a few seconds)
examples/         quickstart.py, train_with_pruning.py, qat_lsq.py
```

## Notes

- Layers handled: `nn.Linear`, `nn.Conv1d/2d/3d`, and for pruning also `nn.MultiheadAttention` (`in_proj_weight`). Other modules are left alone.
- `quantize_ptq` folds BatchNorm into the preceding layer first (a deployed int8 model runs folded weights); `prepare_qat` keeps BatchNorm so it can keep adapting.
- Quantization functions return a deep copy by default (`inplace=False`); pruning functions work in place because the masks reference the model's parameters.
- `quantize_dynamic_int8` wraps `torch.ao.quantization`, which PyTorch has deprecated in favour of `torchao`; it still ships and is the quickest way to real int8 speed-ups on x86 CPUs.

## Tests

```bash
pytest
```
