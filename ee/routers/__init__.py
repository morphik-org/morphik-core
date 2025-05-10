"""Enterprise-only FastAPI routers.

This sub-package bundles **all** additional HTTP API routes that are only
available in Morphik Enterprise Edition.  Each module should expose an
``APIRouter`` instance called ``router`` so that it can be conveniently
mounted via :pyfunc:`ee.init_app`.
"""

import logging
from importlib import import_module
from typing import List

from fastapi import FastAPI

from .apps import router as _apps_router  # noqa: F401 – imported for side effects

__all__: List[str] = []


def init_app(app: FastAPI) -> None:
    """Mount all enterprise routers onto the given *app* instance."""
    logger = logging.getLogger(__name__)
    logger.info("EE.ROUTERS.INIT_APP: Initializing enterprise routers...")

    # Discover routers lazily – import sub-modules that register a global
    # ``router`` attribute.  Keep the list here explicit to avoid accidental
    # exposure of unfinished modules.
    for module_path in [
        "ee.routers.cloud_uri",
        "ee.routers.apps",
        "ee.routers.connectors_router",
    ]:
        logger.info(f"EE.ROUTERS.INIT_APP: Processing module: {module_path}")
        try:
            logger.info(f"EE.ROUTERS.INIT_APP: Attempting to import {module_path}...")
            mod = import_module(module_path)
            logger.info(f"EE.ROUTERS.INIT_APP: Successfully imported {module_path}.")

            if hasattr(mod, "router"):
                logger.info(f"EE.ROUTERS.INIT_APP: Found 'router' in {module_path}. Attempting to include it...")
                app.include_router(mod.router)
                logger.info(f"EE.ROUTERS.INIT_APP: Successfully included router from {module_path}.")
            else:
                logger.warning(f"EE.ROUTERS.INIT_APP: Module {module_path} does not have a 'router' attribute.")
        except ImportError as e:
            logger.error(f"EE.ROUTERS.INIT_APP: Failed to import {module_path}: {e}", exc_info=True)
        except Exception as e:
            logger.error(f"EE.ROUTERS.INIT_APP: Unexpected error processing {module_path}: {e}", exc_info=True)
            # Potentially re-raise or handle if a critical router fails, or decide to continue
            # For now, just log and continue to see if other routers load.
    logger.info("EE.ROUTERS.INIT_APP: Finished initializing enterprise routers.")
