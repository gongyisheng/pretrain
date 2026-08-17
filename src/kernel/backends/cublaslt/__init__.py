try:
    from . import _C
except ImportError as error:
    _C = None
    _C_IMPORT_ERROR = str(error)
else:
    _C_IMPORT_ERROR = None
    # Register only what was built: an absent spec is how the selector learns
    # this backend is unavailable, so no runtime availability check is needed.
    from . import gemm as _gemm  # noqa: F401
