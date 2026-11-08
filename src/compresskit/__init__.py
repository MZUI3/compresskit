"""compresskit: pruning and quantization for PyTorch models, as plain functions.

    import compresskit as ck

    masks = ck.prune_magnitude(model, sparsity=0.5)        # zero the smallest 50% of the weights
    qmodel = ck.quantize_ptq(model, calib_batches)          # int8 post-training quantization (simulated)
    print(ck.compare({"fp32": model, "int8": qmodel}, example=calib_batches[0]))
"""
__version__ = "0.1.0"

from .pruning import (  # noqa: F401
    GradualPruner,
    MaskSet,
    channel_scores,
    cubic_schedule,
    format_sparsity,
    gradient_scores,
    nm_mask,
    prune_by_scores,
    prune_filters,
    prune_magnitude,
    prune_nm,
    prune_obd,
    prune_random,
    prune_snip,
    remove_pruned_channels,
    sparsity_report,
)
from .quantization import (  # noqa: F401
    DynamicWeightQuantizer,
    LSQQuantizer,
    NoQuant,
    Observer,
    QConv1d,
    QConv2d,
    QConv3d,
    QLinear,
    Quantizer,
    WeightOnlyConv,
    WeightOnlyLinear,
    attach_quantizers,
    calibrate,
    export_mode,
    export_onnx,
    finalize_qat,
    fold_bn,
    init_weight_quantizers,
    model_parameters,
    prepare_qat,
    quant_layers,
    quant_summary,
    quantize_dynamic_int8,
    quantize_ptq,
    quantize_weight_only,
    quantizer_parameters,
    quantizers,
    set_engine,
    set_mode,
    strip_quantizers,
)
from .utils import (  # noqa: F401
    compare,
    count_macs,
    count_nonzero,
    count_params,
    estimated_bytes,
    latency_ms,
    print_summary,
    state_dict_bytes,
    summary,
    weight_sparsity,
)
from .train import evaluate, finetune  # noqa: F401
from .registry import available, describe, get, register  # noqa: F401
from .pipeline import CompressResult, compress  # noqa: F401
