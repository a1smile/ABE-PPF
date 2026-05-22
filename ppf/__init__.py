run_registration = None
build_ppf_model = None

try:
    from .registration import run_registration
except ModuleNotFoundError as exc:
    if exc.name != "open3d":
        raise

try:
    from .model_builder import build_ppf_model
except ModuleNotFoundError as exc:
    if exc.name != "open3d":
        raise

__all__ = ["run_registration", "build_ppf_model"]
