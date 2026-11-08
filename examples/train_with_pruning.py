"""Gradual magnitude pruning during training, then fine-tuning with the masks fixed.

    python examples/train_with_pruning.py

The data is synthetic (a random linear teacher), so the point is the mechanics, not the accuracy.
"""
import torch
import torch.nn as nn

import compresskit as ck


def make_data(n=2048, d=64, classes=10, seed=0):
    g = torch.Generator().manual_seed(seed)
    teacher = torch.randn(d, classes, generator=g)
    x = torch.randn(n, d, generator=g)
    y = (x @ teacher).argmax(1)
    return torch.utils.data.TensorDataset(x, y)


def evaluate(model, loader):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            correct += int((model(x).argmax(1) == y).sum())
            total += len(y)
    return correct / total


def main():
    torch.manual_seed(0)
    loader = torch.utils.data.DataLoader(make_data(), batch_size=64, shuffle=True)
    model = nn.Sequential(nn.Linear(64, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 10))
    opt = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9, weight_decay=5e-4)
    epochs = 6

    # Sparsity rises from 0 to 90% over the first 70% of training (cubic schedule), then stays fixed.
    pruner = ck.GradualPruner(model, final_sparsity=0.9, total_steps=epochs * len(loader),
                              update_every=10, end=0.7, exclude="last")
    for epoch in range(epochs):
        model.train()
        for x, y in loader:
            loss = nn.functional.cross_entropy(model(x), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            pruner.step()
        print(f"epoch {epoch + 1}: loss {loss.item():.3f}  acc {evaluate(model, loader):.3f}  {pruner}")
    masks = pruner.finish()
    print(masks.report())

    # The masks are still attached: more fine-tuning keeps the zeros. Detach when done.
    masks.detach()
    print(f"weight sparsity read from the model: {ck.weight_sparsity(model):.3f}")
    print(ck.compare({"pruned": model}, example=next(iter(loader))[0]))
    torch.save({"model": model.state_dict(), "masks": masks.state_dict()}, "pruned_mlp.pt")


if __name__ == "__main__":
    main()
