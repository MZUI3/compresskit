"""Wall-clock latency."""
import statistics
import time
from typing import Optional

import torch
import torch.nn as nn

from .._common import batch_input, model_device, to_device


@torch.no_grad()
def latency_ms(model: nn.Module, example, *, iters: int = 50, warmup: int = 10,
               threads: Optional[int] = None) -> float:
    """Median time of one forward pass in milliseconds, at the example's batch size, on the model's device.
    ``threads`` pins the CPU thread count for the measurement (restored afterwards)."""
    was_training = model.training
    model.eval()
    device = model_device(model)
    x = to_device(batch_input(example), device)
    cuda = device.type == "cuda"
    old_threads = torch.get_num_threads()
    if threads:
        torch.set_num_threads(threads)
    try:
        for _ in range(warmup):
            model(x)
        times = []
        for _ in range(iters):
            if cuda:
                torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            model(x)
            if cuda:
                torch.cuda.synchronize(device)
            times.append((time.perf_counter() - t0) * 1000)
    finally:
        torch.set_num_threads(old_threads)
        model.train(was_training)
    return statistics.median(times)
