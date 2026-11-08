import pytest
import torch
import torch.nn as nn

import compresskit as ck


def test_evaluate(mlp, mlp_batches):
    r = ck.evaluate(mlp, mlp_batches)
    assert set(r) == {"loss", "accuracy"} and 0.0 <= r["accuracy"] <= 1.0 and r["loss"] > 0
    r = ck.evaluate(mlp, mlp_batches, loss_fn=lambda m, b: m(b[0]).pow(2).mean(), num_batches=2)
    assert set(r) == {"loss"}
    r = ck.evaluate(mlp, mlp_batches, loss_fn=lambda m, b: m(b[0]).pow(2).mean(),
                    metric=lambda m, b: float((m(b[0]).argmax(1) == b[1]).float().mean()))
    assert set(r) == {"loss", "metric"}


def test_finetune_keeps_masks(mlp, mlp_batches):
    masks = ck.prune_magnitude(mlp, 0.5)
    before = mlp[0].weight.detach().clone()
    history = ck.finetune(mlp, mlp_batches, epochs=2, lr=0.05, masks=masks, val_loader=mlp_batches, log=None)
    assert len(history) == 2
    assert {"epoch", "train_loss", "lr", "time_s", "val_loss", "val_accuracy", "sparsity"} <= set(history[0])
    assert history[-1]["lr"] < 0.05  # cosine schedule ran
    assert not torch.equal(before, mlp[0].weight)
    assert abs(ck.weight_sparsity(mlp) - 0.5) < 0.01
    assert not masks.attached
    assert not mlp.training  # finetune leaves the model in eval mode


def test_finetune_with_gradual_pruner(mlp, mlp_batches):
    pruner = ck.GradualPruner(mlp, final_sparsity=0.7, total_steps=2 * len(mlp_batches), update_every=1, end=1.0)
    ck.finetune(mlp, mlp_batches, epochs=2, pruner=pruner, log=None)
    assert abs(pruner.masks.sparsity() - 0.7) < 0.02
    assert abs(ck.weight_sparsity(mlp) - 0.7) < 0.02


def test_finetune_qat_lazy_lsq(mlp, mlp_batches):
    q = ck.prepare_qat(mlp, None, method="lsq", w_bits=4, a_bits=4)
    assert len(ck.quantizer_parameters(q)) == 2  # weight step sizes exist; activation ones wait for data
    history = ck.finetune(q, mlp_batches, epochs=1, optimizer="adam", lr=1e-3, quantizer_lr=1e-4, log=None)
    qparams = ck.quantizer_parameters(q)
    assert len(qparams) == 4 and all(p.grad is not None for p in qparams)
    assert history[0]["train_loss"] > 0
    ck.finalize_qat(q)
    assert torch.isfinite(q(mlp_batches[0][0])).all()


def test_finetune_custom_optimizer_and_max_steps(mlp, mlp_batches):
    opt = torch.optim.SGD(mlp.parameters(), lr=0.01)
    logged = []
    history = ck.finetune(mlp, mlp_batches, epochs=5, optimizer=opt, max_steps=3, scheduler=None, log=logged.append)
    assert len(history) == 1 and len(logged) == 1 and history[0]["lr"] == 0.01
    with pytest.raises(ValueError):
        ck.finetune(mlp, mlp_batches, epochs=1, optimizer="rmsprop", log=None)
    with pytest.raises(ValueError):
        ck.finetune(mlp, mlp_batches, epochs=1, scheduler="step", log=None)


def test_pipeline_finetune_receives_masks(mlp, mlp_batches):
    result = ck.compress(mlp, [("magnitude", {"sparsity": 0.5}),
                               ("finetune", {"train_loader": mlp_batches, "epochs": 1, "lr": 0.05, "log": None})])
    assert abs(result.masks.sparsity() - 0.5) < 0.01
    assert abs(ck.weight_sparsity(result.model) - 0.5) < 0.01  # the masks were attached during fine-tuning
    assert isinstance(result.steps[1]["output"], list) and len(result.steps[1]["output"]) == 1
    assert ck.weight_sparsity(mlp) == 0.0
