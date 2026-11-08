"""Prune and run QAT inside a PyTorch Lightning fit, with callbacks only.

    pip install lightning
    python examples/lightning_callbacks.py
"""
import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F

import compresskit as ck
import compresskit.lightning as ckl


class LitMLP(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(nn.Linear(64, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 10))

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = F.cross_entropy(self(x), y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.05, momentum=0.9, weight_decay=5e-4)


def make_loader(n=4096, d=64, classes=10, seed=0):
    g = torch.Generator().manual_seed(seed)
    teacher = torch.randn(d, classes, generator=g)
    x = torch.randn(n, d, generator=g)
    return torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x, (x @ teacher).argmax(1)), batch_size=64)


def main():
    L.seed_everything(0)
    loader = make_loader()
    lit = LitMLP()
    callbacks = [
        ckl.GradualPruning(final_sparsity=0.8, end=0.7, update_every=10, exclude="last", module_attr="model"),
        ckl.QAT(w_bits=8, a_bits=8, method="ste", module_attr="model"),
    ]
    trainer = L.Trainer(max_epochs=6, accelerator="cpu", devices=1, logger=False, enable_checkpointing=False,
                        enable_model_summary=False, callbacks=callbacks)
    trainer.fit(lit, loader)

    print(callbacks[0].masks.report())
    print(ck.quant_summary(lit.model))
    print(ck.evaluate(lit.model, loader))
    print(ck.compare({"pruned 80% + int8 QAT": lit.model}, example=next(iter(loader))[0]))


if __name__ == "__main__":
    main()
