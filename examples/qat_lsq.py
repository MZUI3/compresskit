"""4-bit quantization-aware training with LSQ, compared with 4-bit PTQ.

    python examples/qat_lsq.py

Synthetic data again: watch the gap between PTQ and QAT at 4 bits, not the absolute accuracy.
"""
import copy

import torch
import torch.nn as nn

import compresskit as ck


def make_data(n=4096, d=32, classes=5, seed=0):
    g = torch.Generator().manual_seed(seed)
    teacher = nn.Sequential(nn.Linear(d, 64), nn.Tanh(), nn.Linear(64, classes))
    with torch.no_grad():
        x = torch.randn(n, d, generator=g)
        y = teacher(x).argmax(1)
    return torch.utils.data.TensorDataset(x, y)


def accuracy(model, loader):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            correct += int((model(x).argmax(1) == y).sum())
            total += len(y)
    return correct / total


def train(model, loader, optimizer, epochs):
    for _ in range(epochs):
        model.train()
        for x, y in loader:
            loss = nn.functional.cross_entropy(model(x), y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


def main():
    torch.manual_seed(0)
    loader = torch.utils.data.DataLoader(make_data(), batch_size=64, shuffle=True)
    model = nn.Sequential(nn.Linear(32, 128), nn.BatchNorm1d(128), nn.ReLU(),
                          nn.Linear(128, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Linear(128, 5))
    train(model, loader, torch.optim.Adam(model.parameters(), lr=1e-3), epochs=5)
    print(f"fp32 baseline: {accuracy(model, loader):.3f}")

    calib = [x for x, _ in list(loader)[:8]]
    ptq4 = ck.quantize_ptq(model, calib, w_bits=4, a_bits=4)
    print(f"4-bit PTQ:     {accuracy(ptq4, loader):.3f}")

    # QAT: start from the trained model, learn weights and step sizes together
    qat = ck.prepare_qat(copy.deepcopy(model), calib, method="lsq", w_bits=4, a_bits=4)
    optimizer = torch.optim.Adam([
        {"params": ck.model_parameters(qat), "lr": 3e-4},
        {"params": ck.quantizer_parameters(qat), "lr": 1e-4, "weight_decay": 0.0},
    ])
    train(qat, loader, optimizer, epochs=3)
    ck.finalize_qat(qat)
    print(f"4-bit LSQ QAT: {accuracy(qat, loader):.3f}")
    print(ck.quant_summary(qat))
    print(ck.compare({"fp32": model, "ptq w4a4": ptq4, "qat w4a4": qat}, example=calib[0]))


if __name__ == "__main__":
    main()
