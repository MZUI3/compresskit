# compresskit

[English](README.md) | 한국어

PyTorch 모델의 pruning과 quantization을 그냥 함수로 쓰는 라이브러리입니다. GitHub에서 `pip install` 하고, import 하고, 아무 `nn.Module`에나 호출하면 됩니다. config 파일도, 학습 하네스도, 데이터셋도 없습니다. 모든 함수는 모델(그리고 기법이 필요로 하면 calibration 배치 몇 개)을 받아 결과를 돌려줍니다.

```python
import compresskit as ck

masks  = ck.prune_magnitude(model, sparsity=0.5)       # 크기가 작은 50% 가중치를 0으로 (in place)
qmodel = ck.quantize_ptq(model, calib_batches)         # int8 post-training quantization (복사본 반환)
print(ck.compare({"fp32": model, "int8": qmodel}, example=calib_batches[0]))
```

## 설치

```bash
pip install git+https://github.com/MZUI3/compresskit.git
```

개발용:

```bash
git clone https://github.com/MZUI3/compresskit.git
cd compresskit
pip install -e ".[dev]"
pytest
```

Python 3.10+, PyTorch 2.0+. 모든 함수는 모델이 올라가 있는 장치(CPU든 CUDA든)에서 그대로 동작합니다. 유일한 예외는 PyTorch의 CPU int8 커널을 쓰는 `quantize_dynamic_int8`입니다. Lightning 콜백을 쓰려면 `pip install lightning`이 필요합니다.

쓰는 방법은 세 가지입니다. 직접 다루는 정도가 큰 순서입니다.

1. **함수**: 내 모델과 내 학습 루프에 `ck.prune_magnitude`, `ck.quantize_ptq` 등을 그대로 호출.
2. **`ck.finetune` / `ck.compress`**: 학습 루프와 기법 체인까지 맡기고 싶을 때.
3. **콜백**(`compresskit.lightning`, PyTorch Lightning에서 차용한 방식): PyTorch Lightning으로 학습한다면 콜백 하나 추가로 끝. 나머지는 그대로.

## Pruning

| 함수 | 하는 일 | 참고 |
|---|---|---|
| `prune_magnitude(model, sparsity, scope="global")` | \|w\|가 가장 작은 가중치를 0으로 | Han et al., 2015 |
| `prune_random(model, sparsity, seed=None)` | 랜덤 baseline | |
| `prune_snip(model, batches, sparsity)` | 배치 몇 개로 \|w · ∂L/∂w\| 점수를 매김, 초기화 직후에도 사용 가능 | Lee et al., 2019 |
| `prune_obd(model, batches, sparsity)` | Optimal Brain Damage, Hessian 대각을 empirical Fisher로 근사 | LeCun et al., 1990 |
| `prune_filters(model, ratio, criterion="l2")` | 출력 채널(필터) 단위로 L1/L2 norm이 작은 것을 통째로 0으로 | Li et al., 2017 |
| `prune_nm(model, n=2, m=4)` | N:M semi-structured sparsity (2:4는 Ampere 이후 GPU의 sparse tensor core에서 가속) | Mishra et al., 2021 |
| `prune_by_scores(model, scores, sparsity)` | 직접 만든 중요도 점수 `{파라미터 이름: tensor}`로 pruning | |
| `GradualPruner(model, final_sparsity, total_steps)` | 학습 중 cubic schedule로 sparsity를 점진적으로 올림 | Zhu & Gupta, 2017 |
| `remove_pruned_channels(model, example)` | `prune_filters`가 0으로 만든 채널을 실제로 삭제: 파라미터, MACs, latency 모두 감소 | |

모든 함수는 **in place**로 pruning하고 `MaskSet`을 돌려줍니다. Conv / Linear / MultiheadAttention 가중치마다 boolean mask 하나씩입니다.

```python
masks = ck.prune_magnitude(model, 0.8, scope="global", exclude=("first", "last"))
masks.sparsity()            # 0.8
print(masks.report())       # 레이어별 표

masks.attach()              # fine-tuning 동안 0을 유지: gradient를 mask하고, 매 forward 직전에
for x, y in loader:         # mask를 다시 적용해 momentum/weight decay가 0을 되살리지 못하게 함
    loss = criterion(model(x), y); loss.backward(); optimizer.step(); optimizer.zero_grad()
masks.detach()

torch.save({"model": model.state_dict(), "masks": masks.state_dict()}, "pruned.pt")
masks = ck.MaskSet(model).load_state_dict(torch.load("pruned.pt")["masks"])   # 나중에 복원
```

- `scope="global"`은 모든 레이어를 하나의 threshold로 순위를 매기고, `"layer"`는 레이어마다 같은 sparsity로 자릅니다.
- `exclude` / `include`에는 `"first"`, `"last"`, 모듈 이름, 접두사(`"backbone.layer1"`)를 쓸 수 있습니다.
- 반복 pruning: `MaskSet`을 다시 넘기면 됩니다(`ck.prune_magnitude(model, 0.9, masks=masks)`). 이미 잘린 가중치는 되살아나지 않습니다.
- 학습 중 점진적 pruning:

```python
pruner = ck.GradualPruner(model, final_sparsity=0.8, total_steps=epochs * len(loader), update_every=100)
for x, y in loader:
    ...; optimizer.step(); optimizer.zero_grad()
    pruner.step()
masks = pruner.finish()
```

mask는 텐서 shape을 그대로 두므로 unstructured pruning은 저장 용량(`estimated_bytes`)만 줄이고 latency는 그대로입니다. filter pruning은 끝까지 갈 수 있습니다.

```python
masks = ck.prune_filters(model, 0.5)                   # 레이어마다 출력 채널 절반을 0으로
ck.finetune(model, train_loader, epochs=3, masks=masks)  # 0을 고정한 채 정확도 회복
removed = ck.remove_pruned_channels(model, example)    # {"features.0": 16, ...}: 이제 모델이 실제로 작아짐
```

`remove_pruned_channels`는 forward 한 번을 추적해서 안전한 곳만 지웁니다. 채널을 만든 레이어의 출력이 채널을 보존하는 모듈(BatchNorm, activation, pooling, dropout, Flatten)만 거쳐 정확히 하나의 Conv(`groups=1`) 또는 Linear로 들어갈 때만 삭제합니다. 출력이 공유되거나(residual 덧셈, concat) grouped conv가 소비하는 경우는 0인 채로 그대로 둡니다. classifier의 출력은 절대 지우지 않습니다. 파라미터 shape이 바뀌므로 학습이 끝난 뒤에 써야 합니다(이전 optimizer와 `MaskSet`은 무효가 됩니다). 0이 된 채널도 bias / BatchNorm shift가 activation을 거친 상수를 다음 레이어로 계속 흘려보냅니다. 이 상수를 `example`에서 측정해 소비 레이어의 bias에 접어 넣으므로, Linear 소비자와 padding 없는 conv에서는 출력이 그대로이고 zero padding이 있는 conv에서는 테두리 위치만 달라집니다(`compensate_bias=False`로 끌 수 있습니다).

## Quantization

| 함수 | 하는 일 | 결과 |
|---|---|---|
| `quantize_ptq(model, calib_batches, w_bits=8, a_bits=8)` | BatchNorm folding 후 가중치와 레이어 입력을 fake-quantize하고, 배치 몇 개로 range를 calibration (`minmax`, `percentile`, `mse`, `kl` observer) | simulated |
| `prepare_qat(model, calib_batches, method="ste" \| "lsq" \| "lsq+")` → 학습 → `finalize_qat(model)` | straight-through estimator, LSQ, LSQ+로 quantization-aware training | simulated |
| `quantize_weight_only(model, bits=8 \| 4 \| 2, group_size=None)` | 정수 가중치를 packed `uint8` 버퍼에 저장하고 `forward`에서 dequantize | 실제 저장 용량 감소, 모든 장치 |
| `quantize_dynamic_int8(model)` | PyTorch dynamic int8 quantization (Linear / LSTM / GRU) | 실제 int8 커널, CPU |

*simulated*는 가중치와 activation을 정수 격자로 반올림하되 float으로 저장·연산한다는 뜻입니다. int8(또는 int4) 배포에서 볼 정확도를 어떤 장치에서든 그대로 얻을 수 있습니다. `export_onnx`는 8-bit fake quantizer를 QuantizeLinear/DequantizeLinear 노드로 내보내고(TorchScript exporter), ONNX Runtime이 fake-quantized 모델을 1e-2 안팎의 오차로 재현합니다. 실제 정수 커널은 배포하는 런타임(ONNX Runtime, TensorRT 등)의 몫입니다.

```python
# PTQ: 학습 데이터 배치 몇 개면 충분
q = ck.quantize_ptq(model, calib_batches, w_bits=8, a_bits=8, a_observer="percentile")
ck.quant_summary(q)                      # {'quantized_layers': 20, 'weight_bits': [8], 'act_bits': [8], ...}
ck.export_onnx(q, calib_batches[0], "model_int8.onnx")      # onnx 패키지 필요
plain = ck.strip_quantizers(q)           # 다시 일반 nn.Linear / nn.Conv, 가중치는 양자화된 값 유지

# QAT: 4-bit LSQ
q = ck.prepare_qat(model, calib_batches, method="lsq", w_bits=4, a_bits=4)
optimizer = torch.optim.SGD([
    {"params": ck.model_parameters(q), "lr": 1e-3},
    {"params": ck.quantizer_parameters(q), "lr": 1e-4, "weight_decay": 0.0},   # 학습되는 step size
])
...학습...
ck.finalize_qat(q)                       # quantizer 고정, eval 모드

# weight-only int4, group 단위 scale: 가중치 8배 축소, 어디서나 실행
q = ck.quantize_weight_only(model, bits=4, group_size=128)
```

저수준 부품도 그대로 쓸 수 있습니다: `fold_bn`, `attach_quantizers`, `init_weight_quantizers`, `calibrate`, `set_mode`, `quantizers`, `export_mode`, 그리고 `Quantizer`, `LSQQuantizer`, `Observer` 클래스. 양자화된 레이어는 원래 파라미터와 state_dict 키를 그대로 유지하고(`fc.weight`는 그대로 `fc.weight`) `fc.wq.*` / `fc.aq.*` 버퍼가 추가됩니다. 저장한 양자화 모델을 다시 불러오려면 float 모델을 만들고 같은 설정으로 `fold_bn`, `attach_quantizers`를 호출한 뒤 `load_state_dict` 하면 됩니다.

## 학습: fine-tune, 점진적 pruning, QAT

학습 루프가 있다면 각 부품은 한 줄씩입니다. 루프 전에 `masks.attach()`, optimizer step마다 `pruner.step()`, QAT 앞뒤로 `prepare_qat` / `finalize_qat`. 루프가 없다면 `finetune`이 합리적인 기본값(momentum SGD 또는 Adam/AdamW, cosine schedule, validation, mask 처리, QAT 파라미터 그룹)을 갖춘 작은 루프이고, `evaluate`는 loss와 accuracy를 돌려줍니다.

```python
masks = ck.prune_magnitude(model, 0.8)
ck.finetune(model, train_loader, epochs=3, lr=1e-3, masks=masks, val_loader=val_loader)

pruner = ck.GradualPruner(model, final_sparsity=0.9, total_steps=5 * len(train_loader))
ck.finetune(model, train_loader, epochs=5, pruner=pruner)

q = ck.prepare_qat(model, method="lsq", w_bits=4, a_bits=4)
ck.finetune(q, train_loader, epochs=2, optimizer="adam", lr=3e-4, quantizer_lr=1e-4)
ck.finalize_qat(q)

ck.evaluate(q, val_loader)       # {'loss': ..., 'accuracy': ...}
```

배치는 `(inputs, targets)`이고 loss는 cross-entropy가 기본이며, `loss_fn(model, batch)`을 넘기면 바뀝니다. `finetune`은 파이프라인 단계로도 쓸 수 있고 `compress`가 mask를 알아서 넘겨줍니다: `compress(model, [("magnitude", {"sparsity": 0.8}), ("finetune", {"train_loader": loader, "epochs": 2}), ("ptq", {"calib_batches": calib})])`.

## PyTorch Lightning에서 차용한 콜백 방식

`LightningModule`은 그대로 둡니다. 압축은 콜백입니다.

```python
import lightning as L
import compresskit.lightning as ckl

trainer = L.Trainer(max_epochs=20, callbacks=[
    ckl.GradualPruning(final_sparsity=0.8, end=0.7, module_attr="model"),  # 학습의 앞 70% 동안 0 -> 80%
    ckl.QAT(w_bits=8, a_bits=8, method="lsq", module_attr="model"),         # fake quantizer 삽입, fit 끝나면 finalize
])
trainer.fit(lit_module, datamodule)
```

- `Pruning(prune, amount, at_epoch=0)`: `"magnitude"`, `"random"`, `"filters"`, `"nm"` 또는 `model -> MaskSet` callable로 한 번 pruning하고 mask를 고정. `Pruning(masks=...)`는 이미 있는 mask를 붙임.
- `GradualPruning(final_sparsity, ...)`: `trainer.estimated_stepping_batches` 위에서 cubic schedule.
- `QAT(w_bits, a_bits, method)`: `setup`에서 quantizer 삽입, train dataloader에 접근할 수 있으면 그것으로 calibration(아니면 첫 배치), 학습되는 step size는 optimizer에 자동 추가, fit이 끝나면 `finalize_qat`.

`module_attr`는 LightningModule 안의 네트워크 이름입니다(`self.model` -> `"model"`). `None`이면 모듈 전체를 압축합니다. sparsity는 epoch마다 log되고 mask는 `callback.masks`로 꺼낼 수 있습니다. `lightning`과 예전 `pytorch_lightning` 패키지 모두에서 동작합니다.

## 측정

```python
ck.summary(model, example)        # 파라미터 수, 0이 아닌 수, weight sparsity, state_dict MB, 추정 MB, MACs
ck.compare({"fp32": model, "pruned": pruned, "int8": q}, example, latency=True)   # markdown 표
ck.count_params(model); ck.weight_sparsity(model); ck.count_macs(model, example)
ck.estimated_bytes(model)         # 양자화 가중치는 bit 수대로, 0이 있는 텐서는 non-zero + 1-bit mask로 계산
ck.latency_ms(model, example)     # 모델이 있는 장치에서 forward 1회의 중앙값(ms)
```

## 파이프라인과 직접 만든 기법

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

ck.available()        # 등록된 이름 전부
print(ck.describe())  # 기법별 한 줄 설명
```

## 구조

```
src/compresskit/
  pruning/        masks.py (MaskSet), unstructured.py, structured.py, nm.py, gradual.py, shrink.py (remove_pruned_channels)
  quantization/   observers.py, quantizer.py, layers.py, fold.py, ptq.py, qat.py, weight_only.py, torch_backend.py
  utils/          stats.py (파라미터, MACs, 바이트, summary, compare), benchmark.py (latency)
  train.py        finetune(), evaluate()
  lightning.py    Pruning, GradualPruning, QAT 콜백 (lightning 필요)
  registry.py     @register / get / available
  pipeline.py     compress()
tests/            pytest 스위트 (작은 CPU 모델, 수 초; lightning이 없으면 Lightning 테스트는 skip)
examples/         quickstart.py, train_with_pruning.py, qat_lsq.py, lightning_callbacks.py
```

## 참고

- 다루는 레이어: `nn.Linear`, `nn.Conv1d/2d/3d`, pruning에서는 `nn.MultiheadAttention`(`in_proj_weight`)까지. 그 외 모듈은 건드리지 않습니다.
- `quantize_ptq`는 먼저 BatchNorm을 앞 레이어에 fold합니다(배포되는 int8 모델은 fold된 가중치로 돕니다). `prepare_qat`는 BatchNorm을 유지해서 학습 중 계속 적응하게 둡니다.
- quantization 함수는 기본적으로 deep copy를 반환하고(`inplace=False`), pruning 함수는 mask가 모델 파라미터를 참조해야 하므로 in place로 동작합니다.
- `quantize_dynamic_int8`은 `torch.ao.quantization`을 감쌉니다. PyTorch가 `torchao` 쪽으로 가면서 deprecated 표시를 했지만 아직 포함되어 있고, x86 CPU에서 실제 int8 속도 향상을 얻는 가장 빠른 길입니다.

## 테스트

```bash
pytest
```
