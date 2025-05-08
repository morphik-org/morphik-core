"""Enterprise-only FastAPI routers.

This sub-package bundles **all** additional HTTP API routes that are only
available in Morphik Enterprise Edition.  Each module should expose an
``APIRouter`` instance called ``router`` so that it can be conveniently
mounted via :pyfunc:`ee.init_app`.
"""

from importlib import import_module
from typing import List

from fastapi import FastAPI

from .apps import router as _apps_router  # noqa: F401 – imported for side effects

__all__: List[str] = []


def init_app(app: FastAPI) -> None:
    """Mount all enterprise routers onto the given *app* instance."""
    # Discover routers lazily – import sub-modules that register a global
    # ``router`` attribute.  Keep the list here explicit to avoid accidental
    # exposure of unfinished modules.
    for module_path in [
        "ee.routers.cloud_uri",
        "ee.routers.apps",
    ]:
        mod = import_module(module_path)
        if hasattr(mod, "router"):
            app.include_router(mod.router)
