"""Prune, quantize and compare a small CNN on random data.

    python examples/quickstart.py
"""
import torch
import torch.nn as nn

import compresskit as ck


def small_cnn(num_classes=10):
    return nn.Sequential(
        nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
        nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
        nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        nn.Linear(64, num_classes),
    )


def main():
    torch.manual_seed(0)
    model = small_cnn()
    calib = [torch.randn(16, 3, 32, 32) for _ in range(8)]  # stand-in for a few training batches

    # BatchNorm needs running statistics before quantization: pretend we trained
    model.train()
    with torch.no_grad():
        for x in calib:
            model(x)
    model.eval()
    x = calib[0]

    # 1. unstructured magnitude pruning, 60% global sparsity
    result = ck.compress(model, [("magnitude", {"sparsity": 0.6})])
    pruned = result.model
    print(result.masks.report())

    # 2. int8 PTQ on the float model, and on the pruned one (the zeros survive quantization)
    int8 = ck.quantize_ptq(model, calib)
    pruned_int8 = ck.quantize_ptq(pruned, calib)

    # 3. weight-only int4 with 32-value groups: real storage reduction, any device
    int4 = ck.quantize_weight_only(model, bits=4, group_size=32, layers=(nn.Linear, nn.Conv2d))

    print(ck.compare({"fp32": model, "pruned 60%": pruned, "int8 ptq": int8,
                      "pruned + int8": pruned_int8, "weight-only int4": int4}, example=x, latency=True))
    for name, m in [("int8", int8), ("pruned+int8", pruned_int8), ("int4", int4)]:
        err = (m(x) - model(x)).norm() / model(x).norm()
        print(f"{name:>12}: relative output error vs fp32 = {err:.4f}")


if __name__ == "__main__":
    main()
