import importlib
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
from pydantic import BaseModel, ValidationError

INGESTION_MODULE_NAME = "core.services.ingestion_service"
_MISSING = object()


class PydanticSettingsStub(BaseModel):
    MODE: str
    ENABLE_COLPALI: bool
    COLPALI_PDF_DPI: int


def _module_attr(module_name: str, attr_name: str):
    module = sys.modules.get(module_name)
    if module is None:
        return _MISSING
    return getattr(module, attr_name, _MISSING)


def _remove_module(module_name: str):
    module = sys.modules.pop(module_name, _MISSING)
    if module is _MISSING:
        return

    parent_name, _, attr_name = module_name.rpartition(".")
    parent_module = sys.modules.get(parent_name)
    if parent_module is not None and getattr(parent_module, attr_name, _MISSING) is module:
        delattr(parent_module, attr_name)


def _restore_module(module_name: str, previous_module, previous_parent_attr) -> None:
    parent_name, _, attr_name = module_name.rpartition(".")

    if previous_module is _MISSING:
        _remove_module(module_name)
    else:
        sys.modules[module_name] = previous_module

    parent_module = sys.modules.get(parent_name)
    if parent_module is None:
        return

    if previous_parent_attr is _MISSING:
        current_attr = getattr(parent_module, attr_name, _MISSING)
        current_module = sys.modules.get(module_name, _MISSING)
        if current_attr is not _MISSING and current_attr is current_module:
            delattr(parent_module, attr_name)
    else:
        setattr(parent_module, attr_name, previous_parent_attr)


def _remove_module_tree(root_module_name: str) -> None:
    for module_name in sorted(
        [name for name in sys.modules if name == root_module_name or name.startswith(f"{root_module_name}.")],
        key=len,
        reverse=True,
    ):
        _remove_module(module_name)


def _snapshot_module_tree(root_module_name: str):
    snapshot = {}
    for module_name, module in sys.modules.items():
        if module_name == root_module_name or module_name.startswith(f"{root_module_name}."):
            parent_name, _, attr_name = module_name.rpartition(".")
            snapshot[module_name] = (module, _module_attr(parent_name, attr_name))
    return snapshot


def _restore_module_tree(root_module_name: str, previous_modules, previous_root_attr) -> None:
    _remove_module_tree(root_module_name)
    for module_name, (previous_module, previous_parent_attr) in sorted(
        previous_modules.items(), key=lambda item: len(item[0])
    ):
        _restore_module(module_name, previous_module, previous_parent_attr)

    if root_module_name not in previous_modules:
        parent_name, _, attr_name = root_module_name.rpartition(".")
        parent_module = sys.modules.get(parent_name)
        if parent_module is None:
            return
        if previous_root_attr is _MISSING:
            if getattr(parent_module, attr_name, _MISSING) is not _MISSING:
                delattr(parent_module, attr_name)
        else:
            setattr(parent_module, attr_name, previous_root_attr)


@pytest.fixture
def fresh_ingestion_service_import():
    modules_before = set(sys.modules)
    previous_ingestion_module = sys.modules.get(INGESTION_MODULE_NAME, _MISSING)
    previous_ingestion_attr = _module_attr("core.services", "ingestion_service")
    previous_embedding_modules = _snapshot_module_tree("core.embedding")
    previous_embedding_attr = _module_attr("core", "embedding")

    import core

    # Avoid unrelated embedding import side effects while testing ingestion_service imports.
    embedding_stub = ModuleType("core.embedding")
    embedding_stub.__path__ = [str(Path(__file__).resolve().parents[2] / "embedding")]
    sys.modules["core.embedding"] = embedding_stub
    setattr(core, "embedding", embedding_stub)

    _remove_module(INGESTION_MODULE_NAME)

    try:
        yield lambda: importlib.import_module(INGESTION_MODULE_NAME)
    finally:
        for module_name in sorted(set(sys.modules) - modules_before, key=len, reverse=True):
            if module_name == INGESTION_MODULE_NAME or module_name.startswith("core.embedding"):
                _remove_module(module_name)

        _restore_module(INGESTION_MODULE_NAME, previous_ingestion_module, previous_ingestion_attr)
        _restore_module_tree("core.embedding", previous_embedding_modules, previous_embedding_attr)


def test_importing_ingestion_service_does_not_resolve_settings(monkeypatch, fresh_ingestion_service_import):
    from core import config as config_module

    calls = []

    def fail_if_resolved():
        calls.append(True)
        raise AssertionError("settings should not resolve during import")

    monkeypatch.setattr(config_module, "get_settings", fail_if_resolved)

    ingestion_module = fresh_ingestion_service_import()

    assert calls == []
    assert ingestion_module.IngestionService.__name__ == "IngestionService"
    with pytest.raises(AssertionError, match="settings should not resolve during import"):
        ingestion_module.settings.MODE


def test_injected_settings_bypass_constructor_resolution(monkeypatch):
    from core.services import ingestion_service as ingestion_module

    sentinel_settings = SimpleNamespace(MODE="cloud", ENABLE_COLPALI=True, COLPALI_PDF_DPI=150)

    def fail_if_resolved():
        raise AssertionError("injected settings should bypass get_settings")

    monkeypatch.setattr(ingestion_module, "get_settings", fail_if_resolved)

    service = ingestion_module.IngestionService(None, None, None, None, None, settings=sentinel_settings)

    assert service.settings is sentinel_settings


def test_constructor_resolves_settings_when_not_injected(monkeypatch):
    from core.services import ingestion_service as ingestion_module

    sentinel_settings = SimpleNamespace(MODE="self_hosted", ENABLE_COLPALI=False, COLPALI_PDF_DPI=96)
    calls = []

    def get_test_settings():
        calls.append(True)
        return sentinel_settings

    monkeypatch.setattr(ingestion_module, "settings", ingestion_module._SettingsProxy())
    monkeypatch.setattr(ingestion_module, "get_settings", get_test_settings)

    service = ingestion_module.IngestionService(None, None, None, None, None)

    assert calls == [True]
    assert service.settings is sentinel_settings


def test_constructor_uses_config_module_get_settings_after_import(monkeypatch):
    from core import config as config_module
    from core.services import ingestion_service as ingestion_module

    sentinel_settings = SimpleNamespace(MODE="self_hosted", ENABLE_COLPALI=False, COLPALI_PDF_DPI=96)

    monkeypatch.setattr(ingestion_module, "settings", ingestion_module._SettingsProxy())
    monkeypatch.setattr(config_module, "get_settings", lambda: sentinel_settings)

    service = ingestion_module.IngestionService(None, None, None, None, None)

    assert service.settings is sentinel_settings


def test_constructor_resolves_settings_before_applying_proxy_overrides(monkeypatch):
    from core.services import ingestion_service as ingestion_module

    proxy = ingestion_module._SettingsProxy()
    sentinel_settings = SimpleNamespace(MODE="self_hosted", ENABLE_COLPALI=False, COLPALI_PDF_DPI=96)
    calls = []

    def get_test_settings():
        calls.append(True)
        return sentinel_settings

    monkeypatch.setattr(ingestion_module, "settings", proxy)
    monkeypatch.setattr(ingestion_module, "get_settings", get_test_settings)

    ingestion_module.settings.MODE = "cloud"
    service = ingestion_module.IngestionService(None, None, None, None, None)

    assert calls == [True]
    assert service.settings is not proxy
    assert service.settings is not sentinel_settings
    assert service.settings.MODE == "cloud"
    assert service.settings.ENABLE_COLPALI is False


def test_deleted_proxy_overrides_restore_constructor_fallback(monkeypatch):
    from core.services import ingestion_service as ingestion_module

    proxy = ingestion_module._SettingsProxy()
    sentinel_settings = SimpleNamespace(MODE="self_hosted", ENABLE_COLPALI=False, COLPALI_PDF_DPI=96)
    calls = []

    def get_test_settings():
        calls.append(True)
        return sentinel_settings

    monkeypatch.setattr(ingestion_module, "settings", proxy)
    monkeypatch.setattr(ingestion_module, "get_settings", get_test_settings)

    ingestion_module.settings.MODE = "cloud"
    del ingestion_module.settings.MODE
    service = ingestion_module.IngestionService(None, None, None, None, None)

    assert calls == [True]
    assert service.settings is sentinel_settings
    assert service.settings.MODE == "self_hosted"


def test_proxy_overrides_are_validated_during_constructor_resolution(monkeypatch):
    from core.services import ingestion_service as ingestion_module

    proxy = ingestion_module._SettingsProxy()
    sentinel_settings = PydanticSettingsStub(MODE="self_hosted", ENABLE_COLPALI=False, COLPALI_PDF_DPI=96)

    monkeypatch.setattr(ingestion_module, "settings", proxy)
    monkeypatch.setattr(ingestion_module, "get_settings", lambda: sentinel_settings)

    ingestion_module.settings.COLPALI_PDF_DPI = "not-an-int"

    with pytest.raises(ValidationError):
        ingestion_module.IngestionService(None, None, None, None, None)


def test_module_settings_proxy_resolves_lazily(monkeypatch):
    from core.services import ingestion_service as ingestion_module

    sentinel_settings = SimpleNamespace(UNIQUE_TEST_VALUE="resolved")

    monkeypatch.setattr(ingestion_module, "settings", ingestion_module._SettingsProxy())
    monkeypatch.setattr(ingestion_module, "get_settings", lambda: sentinel_settings)

    assert ingestion_module.settings.UNIQUE_TEST_VALUE == "resolved"
