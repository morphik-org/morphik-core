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

__all__: List[str] = []


def init_app(app: FastAPI) -> None:
    """Enterprise routers disabled for simplified surface."""
    logger = logging.getLogger(__name__)
    logger.info("EE.ROUTERS.INIT_APP: Skipped (EE endpoints disabled)")
