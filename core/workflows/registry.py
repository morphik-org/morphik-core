from __future__ import annotations

import importlib
import pkgutil
from pathlib import Path
from typing import Dict, List

from core.models.workflows import ActionDefinition

"""Dynamic registry for Workflow Actions (Step-3)."""

_ACTIONS_PACKAGE = "core.workflows.actions"


class _Registry:
    def __init__(self):
        self._defs: Dict[str, ActionDefinition] = {}
        self._modules: Dict[str, object] = {}
        self._discover()

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    def _discover(self):
        package = importlib.import_module(_ACTIONS_PACKAGE)
        package_path = Path(package.__file__).parent

        for _, mod_name, is_pkg in pkgutil.iter_modules([str(package_path)]):
            if is_pkg:
                continue
            full_name = f"{_ACTIONS_PACKAGE}.{mod_name}"
            mod = importlib.import_module(full_name)
            if hasattr(mod, "definition"):
                definition: ActionDefinition = getattr(mod, "definition")
                self._defs[definition.id] = definition
                self._modules[definition.id] = mod

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def list_definitions(self) -> List[ActionDefinition]:
        return list(self._defs.values())

    def get_definition(self, action_id: str) -> ActionDefinition | None:
        return self._defs.get(action_id)

    def get_runner(self, action_id: str):
        mod = self._modules.get(action_id)
        if not mod or not hasattr(mod, "run"):
            return None
        return getattr(mod, "run")


ACTION_REGISTRY = _Registry()
