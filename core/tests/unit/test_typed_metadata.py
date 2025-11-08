"""Unit tests for typed metadata normalization and coercion."""

from datetime import UTC, date, datetime
from decimal import Decimal

import pytest

from core.utils.typed_metadata import (
    ALL_METADATA_TYPES,
    SCALAR_METADATA_TYPES,
    TypedMetadataError,
    canonicalize_type_name,
    merge_metadata,
    normalize_metadata,
)


class TestCanonicalizeTypeName:
    """Test type name canonicalization."""

    def test_canonical_names(self):
        """Test that canonical names map to themselves."""
        assert canonicalize_type_name("string") == "string"
        assert canonicalize_type_name("number") == "number"
        assert canonicalize_type_name("decimal") == "decimal"
        assert canonicalize_type_name("boolean") == "boolean"
        assert canonicalize_type_name("datetime") == "datetime"
        assert canonicalize_type_name("date") == "date"
        assert canonicalize_type_name("array") == "array"
        assert canonicalize_type_name("object") == "object"

    def test_aliases(self):
        """Test that aliases are properly canonicalized."""
        assert canonicalize_type_name("str") == "string"
        assert canonicalize_type_name("text") == "string"
        assert canonicalize_type_name("int") == "number"
        assert canonicalize_type_name("integer") == "number"
        assert canonicalize_type_name("float") == "number"
        assert canonicalize_type_name("double") == "number"
        assert canonicalize_type_name("numeric") == "number"
        assert canonicalize_type_name("bool") == "boolean"
        assert canonicalize_type_name("timestamp") == "datetime"
        assert canonicalize_type_name("list") == "array"
        assert canonicalize_type_name("dict") == "object"
        assert canonicalize_type_name("map") == "object"

    def test_case_insensitive(self):
        """Test that type names are case insensitive."""
        assert canonicalize_type_name("STRING") == "string"
        assert canonicalize_type_name("Number") == "number"
        assert canonicalize_type_name("BOOLEAN") == "boolean"

    def test_invalid_type(self):
        """Test that invalid types raise TypedMetadataError."""
        with pytest.raises(TypedMetadataError, match="Unsupported metadata type 'invalid'"):
            canonicalize_type_name("invalid")

    def test_invalid_type_with_field(self):
        """Test error message includes field name when provided."""
        with pytest.raises(TypedMetadataError, match="for field 'my_field'"):
            canonicalize_type_name("invalid", field="my_field")


class TestNormalizeMetadata:
    """Test metadata normalization."""

    def test_empty_metadata(self):
        """Test normalization of empty metadata."""
        normalized, types = normalize_metadata({})
        assert normalized == {}
        assert types == {}

    def test_inferred_types(self):
        """Test that types are correctly inferred when no hints provided."""
        metadata = {
            "name": "test",
            "count": 42,
            "price": 19.99,
            "active": True,
            "created": datetime(2024, 1, 15, 12, 30, 0, tzinfo=UTC),
            "birthdate": date(1990, 5, 20),
            "amount": Decimal("1234.56"),
            "tags": ["a", "b"],
            "config": {"key": "value"},
            "empty": None,
        }

        normalized, types = normalize_metadata(metadata)

        assert types["name"] == "string"
        assert types["count"] == "number"
        assert types["price"] == "number"
        assert types["active"] == "boolean"
        assert types["created"] == "datetime"
        assert types["birthdate"] == "date"
        assert types["amount"] == "decimal"
        assert types["tags"] == "array"
        assert types["config"] == "object"
        assert types["empty"] == "null"

    def test_explicit_type_hints_override_inference(self):
        """Test that explicit type hints override type inference."""
        metadata = {"value": "123"}
        type_hints = {"value": "number"}

        normalized, types = normalize_metadata(metadata, type_hints)

        assert normalized["value"] == 123
        assert types["value"] == "number"

    def test_explicit_type_hint_with_none_preserves_null(self):
        """Explicit hints should not turn None into the string 'None'."""
        metadata = {"Filename": None}
        type_hints = {"Filename": "string"}

        normalized, types = normalize_metadata(metadata, type_hints)

        assert normalized["Filename"] is None
        assert types["Filename"] == "null"

    def test_number_coercion_from_string(self):
        """Test number coercion from string."""
        metadata = {"int_val": "42", "float_val": "3.14", "negative": "-99"}
        type_hints = {"int_val": "number", "float_val": "number", "negative": "number"}

        normalized, types = normalize_metadata(metadata, type_hints)

        assert normalized["int_val"] == 42
        assert normalized["float_val"] == 3.14
        assert normalized["negative"] == -99

    def test_number_coercion_rejects_boolean(self):
        """Test that boolean cannot be coerced to number."""
        metadata = {"value": True}
        type_hints = {"value": "number"}

        with pytest.raises(TypedMetadataError, match="cannot coerce boolean"):
            normalize_metadata(metadata, type_hints)

    def test_number_coercion_rejects_nan_and_infinity(self):
        """Test that NaN and infinity are rejected."""
        with pytest.raises(TypedMetadataError, match="cannot store NaN or infinite"):
            normalize_metadata({"value": float("nan")}, {"value": "number"})

        with pytest.raises(TypedMetadataError, match="cannot store NaN or infinite"):
            normalize_metadata({"value": float("inf")}, {"value": "number"})

    def test_decimal_coercion(self):
        """Test decimal coercion from various types."""
        metadata = {
            "from_string": "1234.56",
            "from_int": 100,
            "from_float": 99.99,
            "from_decimal": Decimal("777.77"),
        }
        type_hints = {k: "decimal" for k in metadata.keys()}

        normalized, types = normalize_metadata(metadata, type_hints)

        assert normalized["from_string"] == "1234.56"
        assert normalized["from_int"] == "100"  # Fixed: now preserves whole numbers correctly
        assert normalized["from_float"] == "99.99"
        assert normalized["from_decimal"] == "777.77"
        assert all(t == "decimal" for t in types.values())

    def test_decimal_normalization(self):
        """Test that decimals are normalized (trailing zeros removed)."""
        metadata = {"value": "1234.5600"}
        type_hints = {"value": "decimal"}

        normalized, _ = normalize_metadata(metadata, type_hints)

        assert normalized["value"] == "1234.56"

    def test_datetime_coercion_from_string(self):
        """Test datetime coercion from ISO8601 string."""
        metadata = {
            "utc": "2024-01-15T12:30:00Z",
            "with_tz": "2024-01-15T12:30:00+05:00",
            "no_tz": "2024-01-15T12:30:00",
        }
        type_hints = {k: "datetime" for k in metadata.keys()}

        normalized, types = normalize_metadata(metadata, type_hints)

        # All should be converted to ISO format with timezone
        assert "2024-01-15T12:30:00" in normalized["utc"]
        assert "2024-01-15T12:30:00+05:00" == normalized["with_tz"]
        # Naive datetime should get UTC
        assert "+00:00" in normalized["no_tz"] or "Z" in normalized["no_tz"]

    def test_datetime_coercion_from_datetime_object(self):
        """Test datetime coercion from datetime object."""
        dt = datetime(2024, 1, 15, 12, 30, 0, tzinfo=UTC)
        metadata = {"value": dt}
        type_hints = {"value": "datetime"}

        normalized, _ = normalize_metadata(metadata, type_hints)

        assert "2024-01-15T12:30:00" in normalized["value"]

    def test_datetime_coercion_from_date_object(self):
        """Test datetime coercion from date object."""
        d = date(2024, 1, 15)
        metadata = {"value": d}
        type_hints = {"value": "datetime"}

        normalized, _ = normalize_metadata(metadata, type_hints)

        assert "2024-01-15" in normalized["value"]

    def test_date_coercion_from_string(self):
        """Test date coercion from ISO8601 string."""
        metadata = {"value": "2024-01-15"}
        type_hints = {"value": "date"}

        normalized, _ = normalize_metadata(metadata, type_hints)

        assert normalized["value"] == "2024-01-15"

    def test_date_coercion_from_datetime_string(self):
        """Test date coercion from datetime string (extracts date part)."""
        metadata = {"value": "2024-01-15T12:30:00Z"}
        type_hints = {"value": "date"}

        normalized, _ = normalize_metadata(metadata, type_hints)

        assert normalized["value"] == "2024-01-15"

    def test_date_coercion_from_date_object(self):
        """Test date coercion from date object."""
        d = date(2024, 1, 15)
        metadata = {"value": d}
        type_hints = {"value": "date"}

        normalized, _ = normalize_metadata(metadata, type_hints)

        assert normalized["value"] == "2024-01-15"

    def test_boolean_coercion_from_string(self):
        """Test boolean coercion from various string values."""
        true_values = ["true", "TRUE", "1", "yes", "YES", "y", "Y", "on", "ON"]
        false_values = ["false", "FALSE", "0", "no", "NO", "n", "N", "off", "OFF"]

        for val in true_values:
            normalized, _ = normalize_metadata({"value": val}, {"value": "boolean"})
            assert normalized["value"] is True, f"Failed for: {val}"

        for val in false_values:
            normalized, _ = normalize_metadata({"value": val}, {"value": "boolean"})
            assert normalized["value"] is False, f"Failed for: {val}"

    def test_boolean_coercion_from_number(self):
        """Test boolean coercion from numbers."""
        normalized, _ = normalize_metadata(
            {"zero": 0, "one": 1, "neg": -1}, {k: "boolean" for k in ["zero", "one", "neg"]}
        )

        assert normalized["zero"] is False
        assert normalized["one"] is True
        assert normalized["neg"] is True

    def test_boolean_invalid_string(self):
        """Test that invalid boolean strings raise error."""
        with pytest.raises(TypedMetadataError, match="expects 'true' or 'false'"):
            normalize_metadata({"value": "maybe"}, {"value": "boolean"})

    def test_array_preservation(self):
        """Test that arrays are preserved and nested values are sanitized."""
        metadata = {
            "tags": ["a", "b", "c"],
            "mixed": [1, "two", True, None],
            "nested": [{"key": "value"}],
        }

        normalized, types = normalize_metadata(metadata)

        assert normalized["tags"] == ["a", "b", "c"]
        assert normalized["mixed"] == [1, "two", True, None]
        assert normalized["nested"] == [{"key": "value"}]
        assert all(types[k] == "array" for k in metadata.keys())

    def test_object_preservation(self):
        """Test that objects are preserved and nested values are sanitized."""
        metadata = {
            "config": {"host": "localhost", "port": 8080, "enabled": True},
        }

        normalized, types = normalize_metadata(metadata)

        assert normalized["config"]["host"] == "localhost"
        assert normalized["config"]["port"] == 8080
        assert normalized["config"]["enabled"] is True
        assert types["config"] == "object"

    def test_nested_datetime_sanitization(self):
        """Test that datetime objects in nested structures are converted to ISO strings."""
        dt = datetime(2024, 1, 15, 12, 30, 0, tzinfo=UTC)
        metadata = {"config": {"created_at": dt}}

        normalized, _ = normalize_metadata(metadata)

        assert isinstance(normalized["config"]["created_at"], str)
        assert "2024-01-15T12:30:00" in normalized["config"]["created_at"]

    def test_null_value(self):
        """Test that null values are preserved."""
        metadata = {"value": None}

        normalized, types = normalize_metadata(metadata)

        assert normalized["value"] is None
        assert types["value"] == "null"


class TestMergeMetadata:
    """Test metadata merging."""

    def test_merge_empty(self):
        """Test merging with empty existing metadata."""
        merged, types = merge_metadata(None, None, {"key": "value"})

        assert merged["key"] == "value"
        assert types["key"] == "string"

    def test_merge_overwrites_existing(self):
        """Test that updates overwrite existing values."""
        existing = {"a": "old", "b": "keep"}
        existing_types = {"a": "string", "b": "string"}
        updates = {"a": "new"}

        merged, types = merge_metadata(existing, existing_types, updates)

        assert merged["a"] == "new"
        assert merged["b"] == "keep"
        assert types["a"] == "string"
        assert types["b"] == "string"

    def test_merge_adds_new_fields(self):
        """Test that new fields are added."""
        existing = {"a": "old"}
        existing_types = {"a": "string"}
        updates = {"b": "new"}

        merged, types = merge_metadata(existing, existing_types, updates)

        assert merged["a"] == "old"
        assert merged["b"] == "new"
        assert types["a"] == "string"
        assert types["b"] == "string"

    def test_merge_preserves_external_id(self):
        """Test that external_id is preserved."""
        existing = {"external_id": "doc-123"}
        existing_types = {"external_id": "string"}
        updates = {"key": "value"}

        merged, types = merge_metadata(existing, existing_types, updates, external_id="doc-123")

        assert merged["external_id"] == "doc-123"

    def test_merge_sets_external_id_if_missing(self):
        """Test that external_id is set if not present."""
        existing = {"key": "value"}
        existing_types = {"key": "string"}
        updates = {}

        merged, types = merge_metadata(existing, existing_types, updates, external_id="doc-456")

        assert merged["external_id"] == "doc-456"
        assert types["external_id"] == "string"

    def test_merge_with_type_change(self):
        """Test merging with explicit type change."""
        existing = {"count": "42"}
        existing_types = {"count": "string"}
        updates = {"count": "99"}
        update_types = {"count": "number"}

        merged, types = merge_metadata(existing, existing_types, updates, update_types)

        assert merged["count"] == 99
        assert types["count"] == "number"

    def test_merge_coerces_updates(self):
        """Test that updates are coerced according to type hints."""
        existing = {}
        existing_types = {}
        updates = {"value": "123"}
        update_types = {"value": "number"}

        merged, types = merge_metadata(existing, existing_types, updates, update_types)

        assert merged["value"] == 123
        assert types["value"] == "number"

    def test_merge_allows_clearing_string_field(self):
        """Typed metadata should allow clearing values back to null."""
        existing = {"Filename": "file.pdf"}
        existing_types = {"Filename": "string"}
        updates = {"Filename": None}
        update_types = {"Filename": "string"}

        merged, types = merge_metadata(existing, existing_types, updates, update_types)

        assert merged["Filename"] is None
        assert types["Filename"] == "null"


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_string_number_coercion(self):
        """Test that empty string cannot be coerced to number."""
        with pytest.raises(TypedMetadataError, match="cannot coerce empty string to number"):
            normalize_metadata({"value": ""}, {"value": "number"})

    def test_empty_string_datetime_coercion(self):
        """Test that empty string cannot be coerced to datetime."""
        with pytest.raises(TypedMetadataError, match="expects a datetime value"):
            normalize_metadata({"value": ""}, {"value": "datetime"})

    def test_empty_string_date_coercion(self):
        """Test that empty string cannot be coerced to date."""
        with pytest.raises(TypedMetadataError, match="expects a date value"):
            normalize_metadata({"value": ""}, {"value": "date"})

    def test_invalid_decimal_string(self):
        """Test that invalid decimal strings raise error."""
        with pytest.raises(TypedMetadataError, match="expects a decimal-compatible value"):
            normalize_metadata({"value": "not-a-number"}, {"value": "decimal"})

    def test_invalid_datetime_string(self):
        """Test that invalid datetime strings raise error."""
        with pytest.raises(TypedMetadataError, match="expects an ISO8601 datetime"):
            normalize_metadata({"value": "not-a-datetime"}, {"value": "datetime"})

    def test_invalid_date_string(self):
        """Test that invalid date strings raise error."""
        with pytest.raises(TypedMetadataError, match="expects an ISO8601 date"):
            normalize_metadata({"value": "not-a-date"}, {"value": "date"})

    def test_type_mismatch_array(self):
        """Test that non-array values cannot be coerced to array."""
        with pytest.raises(TypedMetadataError, match="expects an array"):
            normalize_metadata({"value": "not-an-array"}, {"value": "array"})

    def test_type_mismatch_object(self):
        """Test that non-object values cannot be coerced to object."""
        with pytest.raises(TypedMetadataError, match="expects an object"):
            normalize_metadata({"value": "not-an-object"}, {"value": "object"})

    def test_underscore_in_numbers(self):
        """Test that underscores in number strings are handled (Python-style)."""
        metadata = {"value": "1_000_000"}
        type_hints = {"value": "number"}

        normalized, _ = normalize_metadata(metadata, type_hints)

        assert normalized["value"] == 1000000

    def test_scientific_notation(self):
        """Test that scientific notation is supported."""
        metadata = {"value": "1.23e10"}
        type_hints = {"value": "number"}

        normalized, _ = normalize_metadata(metadata, type_hints)

        assert normalized["value"] == 1.23e10

    def test_constants_are_defined(self):
        """Test that expected constants are defined."""
        assert "string" in SCALAR_METADATA_TYPES
        assert "number" in SCALAR_METADATA_TYPES
        assert "decimal" in SCALAR_METADATA_TYPES
        assert "boolean" in SCALAR_METADATA_TYPES
        assert "datetime" in SCALAR_METADATA_TYPES
        assert "date" in SCALAR_METADATA_TYPES
        assert "null" in SCALAR_METADATA_TYPES

        assert "array" in ALL_METADATA_TYPES
        assert "object" in ALL_METADATA_TYPES
        assert SCALAR_METADATA_TYPES.issubset(ALL_METADATA_TYPES)
