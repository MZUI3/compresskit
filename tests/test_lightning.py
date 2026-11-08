import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

pl = pytest.importorskip("lightning.pytorch")

import compresskit as ck  # noqa: E402
import compresskit.lightning as ckl  # noqa: E402


class Lit(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(16, 64), nn.ReLU(), nn.Linear(64, 10))

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        return F.cross_entropy(self(x), y)

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.05, momentum=0.9, weight_decay=1e-3)


def make_loader(n=256, bs=32):
    g = torch.Generator().manual_seed(0)
    ds = torch.utils.data.TensorDataset(torch.randn(n, 16, generator=g), torch.randint(0, 10, (n,), generator=g))
    return torch.utils.data.DataLoader(ds, batch_size=bs)


def make_trainer(callbacks, epochs=2):
    return pl.Trainer(max_epochs=epochs, accelerator="cpu", devices=1, logger=False, enable_checkpointing=False,
                      enable_progress_bar=False, enable_model_summary=False, callbacks=callbacks)


def test_gradual_pruning_callback():
    model = Lit()
    cb = ckl.GradualPruning(final_sparsity=0.8, update_every=2, module_attr="net")
    make_trainer([cb], epochs=3).fit(model, make_loader())
    assert abs(cb.masks.sparsity() - 0.8) < 0.02
    assert abs(ck.weight_sparsity(model.net) - 0.8) < 0.02
    assert not cb.masks.attached


def test_oneshot_pruning_callback_keeps_zeros():
    model = Lit()
    cb = ckl.Pruning("magnitude", 0.5, module_attr="net")
    make_trainer([cb]).fit(model, make_loader())
    assert abs(ck.weight_sparsity(model.net) - 0.5) < 0.01  # momentum and weight decay did not revive them
    assert not cb.masks.attached

    model = Lit()
    cb = ckl.Pruning(lambda m: ck.prune_filters(m, 0.5, scope="layer"), module_attr="net", at_epoch=1)
    make_trainer([cb]).fit(model, make_loader())
    assert int((model.net[0].weight.flatten(1) == 0).all(1).sum()) == 32
    with pytest.raises(ValueError):
        ckl.Pruning("magnitude")


def test_qat_callback_lazy_lsq_adds_param_group():
    model = Lit()
    cb = ckl.QAT(4, 4, method="lsq", module_attr="net", quantizer_lr=1e-4)
    trainer = make_trainer([cb])
    trainer.fit(model, make_loader())
    assert len(ck.quantizer_parameters(model.net)) == 4
    groups = trainer.optimizers[0].param_groups
    assert len(groups) == 2 and groups[1]["lr"] == 1e-4 and groups[1]["weight_decay"] == 0.0
    aq = model.net[2].aq
    assert torch.allclose(aq.scale, aq.s.detach().abs())  # finalised
    assert not model.net.training


def test_qat_callback_with_calibration_batches():
    model = Lit()
    loader = make_loader()
    cb = ckl.QAT(8, 8, method="lsq", module_attr="net", calib_batches=[x for x, _ in loader][:4])
    trainer = make_trainer([cb], epochs=1)
    trainer.fit(model, loader)
    # the step sizes existed before configure_optimizers, so SGD(self.parameters()) already covered them
    assert len(trainer.optimizers[0].param_groups) == 1
    assert ck.quant_summary(model.net)["quantized_layers"] == 2
    assert torch.isfinite(model(torch.randn(3, 16))).all()
