# Import implementation modules to register kernels.
from . import gemm as _gemm  # noqa: F401

try:
    from . import _C
except ImportError as error:
    _C = None
    _C_IMPORT_ERROR = str(error)
else:
    _C_IMPORT_ERROR = None
