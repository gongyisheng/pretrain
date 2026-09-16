import torch


try:
    from . import _C
except ImportError as error:
    _C = None
    _C_IMPORT_ERROR = str(error)
    print(f"Warning: CUDA kernels were not registered: {_C_IMPORT_ERROR}")
else:
    _C_IMPORT_ERROR = None
    # Register only what was built: an absent spec is how the selector learns
    # this backend is unavailable, so no runtime availability check is needed.
    from . import gemm as _gemm  # noqa: F401

    if hasattr(torch.ops.aot_kernel, "quantize_fp8"):
        from . import fp8 as _fp8  # noqa: F401
    if hasattr(torch.ops.aot_kernel, "quantize_int8"):
        from . import int8 as _int8  # noqa: F401
    if hasattr(torch.ops.aot_kernel, "quantize_mxfp8"):
        from . import mxfp8 as _mxfp8  # noqa: F401
    if hasattr(torch.ops.aot_kernel, "quantize_nvfp4"):
        from . import nvfp4 as _nvfp4  # noqa: F401
