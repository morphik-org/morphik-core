"""
Metadata filter SQL generation with typed comparison support.

Translates JSON-style filter expressions into PostgreSQL WHERE clauses with
special handling for typed metadata (number, decimal, datetime, date).

**Implicit equality** (backwards compatible, JSONB containment):
    {"field": value}

**Explicit operators** (typed comparisons with safe casting):
    {"field": {"$eq": value}}

Supported operators: $and, $or, $nor, $not, $eq, $ne, $gt, $gte, $lt, $lte,
$in, $nin, $exists, $type, $regex, $contains.
"""

import json
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from typing import Any, Dict, List, Optional

from core.utils.typed_metadata import TypedMetadataError, canonicalize_type_name


class InvalidMetadataFilterError(ValueError):
    """Raised when metadata filters are malformed or unsupported."""


class MetadataFilterBuilder:
    """Translate JSON-style metadata filters into SQL, covering arrays, regex, and substring operators."""

    _COLUMN_FIELDS = {
        "filename": "filename",
    }

    def build(self, filters: Optional[Dict[str, Any]]) -> str:
        """Construct a SQL WHERE clause from a metadata filter dictionary."""
        if filters is None:
            return ""

        if not isinstance(filters, dict):
            raise InvalidMetadataFilterError("Metadata filters must be provided as a JSON object.")

        if not filters:
            return ""

        clause = self._parse_metadata_filter(filters, context="metadata filter")
        if not clause:
            raise InvalidMetadataFilterError("Metadata filter produced no valid conditions.")
        return clause

    def _parse_metadata_filter(self, expression: Any, context: str) -> str:
        """Recursively parse a document-operator metadata filter into SQL."""
        if isinstance(expression, dict):
            if not expression:
                raise InvalidMetadataFilterError(f"{context.capitalize()} cannot be empty.")

            clauses: List[str] = []
            for key, value in expression.items():
                if key == "$and":
                    if not isinstance(value, list):
                        raise InvalidMetadataFilterError("$and operator expects a non-empty list of conditions.")
                    clauses.append(
                        self._combine_clauses(
                            [self._parse_metadata_filter(item, context="$and condition") for item in value],
                            "AND",
                            'operator "$and"',
                        )
                    )
                elif key == "$or":
                    if not isinstance(value, list):
                        raise InvalidMetadataFilterError("$or operator expects a non-empty list of conditions.")
                    clauses.append(
                        self._combine_clauses(
                            [self._parse_metadata_filter(item, context="$or condition") for item in value],
                            "OR",
                            'operator "$or"',
                        )
                    )
                elif key == "$nor":
                    if not isinstance(value, list):
                        raise InvalidMetadataFilterError("$nor operator expects a non-empty list of conditions.")
                    inner = self._combine_clauses(
                        [self._parse_metadata_filter(item, context="$nor condition") for item in value],
                        "OR",
                        'operator "$nor"',
                    )
                    clauses.append(f"(NOT {inner})")
                elif key == "$not":
                    sub_context = 'operator "$not"'
                    clauses.append(f"(NOT {self._parse_metadata_filter(value, context=sub_context)})")
                else:
                    clauses.append(self._build_field_metadata_clause(key, value))

            return self._combine_clauses(clauses, "AND", context)

        if isinstance(expression, list):
            if not expression:
                raise InvalidMetadataFilterError(f"{context.capitalize()} cannot be an empty list.")
            subclauses = [self._parse_metadata_filter(item, context="nested condition") for item in expression]
            return self._combine_clauses(subclauses, "OR", context)

        raise InvalidMetadataFilterError(f"{context.capitalize()} must be expressed as a JSON object.")

    def _combine_clauses(self, clauses: List[str], operator: str, context: str) -> str:
        """Combine multiple SQL clauses with a logical operator."""
        cleaned = [clause for clause in clauses if clause]
        if not cleaned:
            raise InvalidMetadataFilterError(f"No valid conditions supplied for {context}.")
        if len(cleaned) == 1:
            return cleaned[0]
        return "(" + f" {operator} ".join(cleaned) + ")"

    def _build_field_metadata_clause(self, field: str, value: Any) -> str:
        """Build SQL clause for a single metadata field."""
        if field in self._COLUMN_FIELDS:
            return self._build_column_field_clause(field, value)

        if isinstance(value, dict) and not any(key.startswith("$") for key in value):
            # Treat as literal JSON sub-document match
            return self._jsonb_contains_clause(field, value)

        if isinstance(value, dict):
            return self._build_operator_clause(field, value)

        if isinstance(value, list):
            return self._build_list_clause(field, value)

        return self._build_single_value_clause(field, value)

    def _build_operator_clause(self, field: str, operators: Dict[str, Any]) -> str:
        """Build SQL clause for operator-based metadata filters."""
        if not isinstance(operators, dict) or not operators:
            raise InvalidMetadataFilterError(f"Operator block for field '{field}' must be a non-empty object.")

        clauses: List[str] = []
        for operator, operand in operators.items():
            if operator in {"$eq", "$ne", "$gt", "$gte", "$lt", "$lte"}:
                # All comparison operators support typed metadata (number, decimal, date, datetime)
                comparison_clause = self._build_comparison_clause(field, operator, operand)
                if operator == "$ne":
                    clauses.append(f"(NOT {comparison_clause})")
                else:
                    clauses.append(comparison_clause)
            elif operator == "$in":
                if not isinstance(operand, list):
                    raise InvalidMetadataFilterError(f"$in operator for field '{field}' expects a list of values.")
                clauses.append(self._build_list_clause(field, operand))
            elif operator == "$nin":
                if not isinstance(operand, list):
                    raise InvalidMetadataFilterError(f"$nin operator for field '{field}' expects a list of values.")
                clauses.append(f"(NOT {self._build_list_clause(field, operand)})")
            elif operator == "$exists":
                clauses.append(self._build_exists_clause(field, operand))
            elif operator == "$not":
                clauses.append(f"(NOT {self._build_field_metadata_clause(field, operand)})")
            elif operator == "$type":
                clauses.append(self._build_type_clause(field, operand))
            elif operator == "$regex":
                clauses.append(self._build_regex_clause(field, operand))
            elif operator == "$contains":
                clauses.append(self._build_contains_clause(field, operand))
            else:
                raise InvalidMetadataFilterError(
                    f"Unsupported metadata filter operator '{operator}' for field '{field}'."
                )

        return self._combine_clauses(clauses, "AND", f"field '{field}' operator block")

    def _build_list_clause(self, field: str, values: List[Any]) -> str:
        """Build clause matching any of the provided values."""
        if not isinstance(values, list) or not values:
            raise InvalidMetadataFilterError(f"Filter list for field '{field}' must contain at least one value.")

        clauses = []
        for item in values:
            if isinstance(item, dict) and any(key.startswith("$") for key in item):
                clauses.append(self._build_operator_clause(field, item))
            else:
                clauses.append(self._build_single_value_clause(field, item))

        return self._combine_clauses(clauses, "OR", f"list of values for field '{field}'")

    def _build_single_value_clause(self, field: str, value: Any) -> str:
        """Build clause matching a single value."""
        if isinstance(value, dict):
            if any(key.startswith("$") for key in value):
                return self._build_operator_clause(field, value)
            return self._jsonb_contains_clause(field, value)

        return self._jsonb_contains_clause(field, value)

    def _build_column_field_clause(self, field: str, value: Any) -> str:
        """Build SQL clause for a reserved column field (e.g., filename)."""
        column = self._COLUMN_FIELDS[field]
        builder = TextColumnFilterBuilder(column)

        if isinstance(value, dict):
            if not value:
                raise InvalidMetadataFilterError(f"{field} filter cannot be empty.")
            if any(key.startswith("$") for key in value):
                return builder.build(value)
            raise InvalidMetadataFilterError(
                f"{field} filter must use operators (e.g., {{'{field}': {{'$eq': 'example.pdf'}}}})."
            )

        if isinstance(value, list):
            return builder._build_in_clause(value, negate=False)

        return builder._build_comparison_clause("$eq", value)

    def _build_exists_clause(self, field: str, operand: Any) -> str:
        """Build clause handling $exists operator."""
        expected = operand
        if isinstance(expected, str):
            expected = expected.lower() in {"1", "true", "yes"}
        elif isinstance(expected, (int, float)):
            expected = bool(expected)
        elif not isinstance(expected, bool):
            raise InvalidMetadataFilterError(f"$exists operator for field '{field}' expects a boolean value.")

        field_key = self._escape_single_quotes(field)
        clause = f"(doc_metadata ? '{field_key}')"
        return clause if expected else f"(NOT {clause})"

    def _build_comparison_clause(self, field: str, operator: str, operand: Any) -> str:
        """Build typed comparison clauses (number, decimal, date, datetime, string)."""
        sql_operator = self._map_comparison_operator(operator)
        clauses = []

        # Try numeric types
        numeric_clause = self._build_numeric_comparison_clause(field, sql_operator, operand)
        if numeric_clause:
            clauses.append(numeric_clause)

        decimal_clause = self._build_decimal_comparison_clause(field, sql_operator, operand)
        if decimal_clause:
            clauses.append(decimal_clause)

        # Try datetime types
        datetime_clause = self._build_datetime_comparison_clause(field, sql_operator, operand)
        if datetime_clause:
            clauses.append(datetime_clause)

        date_clause = self._build_date_comparison_clause(field, sql_operator, operand)
        if date_clause:
            clauses.append(date_clause)

        # Try string type (only for $eq/$ne)
        if operator in {"$eq", "$ne"} and isinstance(operand, str):
            string_clause = self._build_string_comparison_clause(field, sql_operator, operand)
            if string_clause:
                clauses.append(string_clause)

        if not clauses:
            raise InvalidMetadataFilterError(
                f"Operator '{operator}' for field '{field}' requires a numeric, decimal, ISO8601 date/datetime, or string value."
            )

        if len(clauses) == 1:
            return clauses[0]
        return "(" + " OR ".join(clauses) + ")"

    def _build_numeric_comparison_clause(self, field: str, sql_operator: str, operand: Any) -> str:
        """Build comparison clause for 'number' typed metadata."""
        try:
            literal = self._format_numeric_literal(operand)
        except InvalidMetadataFilterError:
            return ""

        field_key = self._escape_single_quotes(field)
        type_expr = self._metadata_type_expr(field_key)
        # Use CASE to ensure casting only happens when type is correct
        value_expr = (
            f"(CASE WHEN {type_expr} = 'number' THEN (doc_metadata ->> '{field_key}')::double precision ELSE NULL END)"
        )
        return f"({value_expr} {sql_operator} {literal})"

    def _build_decimal_comparison_clause(self, field: str, sql_operator: str, operand: Any) -> str:
        """Build comparison clause for 'decimal' typed metadata."""
        try:
            literal = self._format_numeric_literal(operand)
        except InvalidMetadataFilterError:
            return ""

        field_key = self._escape_single_quotes(field)
        type_expr = self._metadata_type_expr(field_key)
        # Use CASE to ensure casting only happens when type is correct
        value_expr = f"(CASE WHEN {type_expr} = 'decimal' THEN (doc_metadata ->> '{field_key}')::numeric ELSE NULL END)"
        return f"({value_expr} {sql_operator} {literal}::numeric)"

    def _build_datetime_comparison_clause(self, field: str, sql_operator: str, operand: Any) -> str:
        """Build comparison clause for 'datetime' typed metadata."""
        try:
            literal = self._format_datetime_literal(operand, field)
        except InvalidMetadataFilterError:
            return ""

        field_key = self._escape_single_quotes(field)
        type_expr = self._metadata_type_expr(field_key)
        # Use CASE to ensure casting only happens when type is correct
        value_expr = (
            f"(CASE WHEN {type_expr} = 'datetime' THEN (doc_metadata ->> '{field_key}')::timestamptz ELSE NULL END)"
        )
        return f"({value_expr} {sql_operator} {literal})"

    def _build_date_comparison_clause(self, field: str, sql_operator: str, operand: Any) -> str:
        """Build comparison clause for 'date' typed metadata."""
        try:
            literal = self._format_date_literal(operand, field)
        except InvalidMetadataFilterError:
            return ""

        field_key = self._escape_single_quotes(field)
        type_expr = self._metadata_type_expr(field_key)
        # Use CASE to ensure casting only happens when type is correct
        value_expr = f"(CASE WHEN {type_expr} = 'date' THEN (doc_metadata ->> '{field_key}')::date ELSE NULL END)"
        return f"({value_expr} {sql_operator} {literal})"

    def _build_string_comparison_clause(self, field: str, sql_operator: str, operand: str) -> str:
        """Build comparison clause for 'string' typed metadata (only for $eq/$ne)."""
        field_key = self._escape_single_quotes(field)
        escaped_value = self._escape_single_quotes(operand)
        type_expr = self._metadata_type_expr(field_key)
        value_expr = f"(doc_metadata ->> '{field_key}')"
        # For strings without explicit type, assume string type (COALESCE handles missing metadata_types)
        return f"((COALESCE({type_expr}, 'string') = 'string') AND {value_expr} {sql_operator} '{escaped_value}')"

    def _build_type_clause(self, field: str, operand: Any) -> str:
        """Build clause enforcing metadata type."""
        if isinstance(operand, str):
            type_names = [operand]
        elif isinstance(operand, list) and operand:
            if not all(isinstance(item, str) for item in operand):
                raise InvalidMetadataFilterError(f"$type operator for field '{field}' expects string entries.")
            type_names = operand
        else:
            raise InvalidMetadataFilterError(f"$type operator for field '{field}' expects a string or list of strings.")

        canonical_types: List[str] = []
        for type_name in type_names:
            try:
                canonical_types.append(canonicalize_type_name(type_name, field=field))
            except TypedMetadataError as exc:
                raise InvalidMetadataFilterError(str(exc)) from exc

        field_key = self._escape_single_quotes(field)
        type_expr = f"COALESCE(metadata_types ->> '{field_key}', 'string')"
        clauses = [f"({type_expr} = '{type_name}')" for type_name in canonical_types]
        if len(clauses) == 1:
            return clauses[0]
        return "(" + " OR ".join(clauses) + ")"

    def _jsonb_contains_clause(self, field: str, value: Any) -> str:
        """Build JSONB containment clause for a field/value pairing.

        This is used for implicit equality (e.g., {"field": "value"}) and only supports
        JSON-serializable types. For typed comparisons with date/datetime/Decimal objects,
        use explicit operators like $eq, $gt, etc.
        """
        try:
            json_payload = json.dumps({field: value})
        except (TypeError, ValueError) as exc:  # noqa: BLE001
            raise InvalidMetadataFilterError(
                f"Metadata filter for field '{field}' contains a non-serializable value: {exc}. "
                f"Use explicit operators like {{'$eq': value}} for typed comparisons with date, datetime, or Decimal objects."
            ) from exc

        escaped_payload = json_payload.replace("'", "''")
        base_clause = f"(doc_metadata @> '{escaped_payload}'::jsonb)"

        array_clause = self._build_array_membership_clause(field, value)
        if array_clause:
            return f"({base_clause} OR {array_clause})"
        return base_clause

    def _build_array_membership_clause(self, field: str, value: Any) -> str:
        """Match scalar comparisons against array-valued metadata fields.

        Only supports JSON-serializable primitives (str, int, float, bool, None).
        """
        if not isinstance(value, (str, int, float, bool)) and value is not None:
            return ""

        try:
            array_payload = json.dumps([value])
        except (TypeError, ValueError):
            return ""

        escaped_array_payload = array_payload.replace("'", "''")
        field_key = self._escape_single_quotes(field)

        return (
            f"((jsonb_typeof(doc_metadata -> '{field_key}') = 'array') "
            f"AND ((doc_metadata -> '{field_key}') @> '{escaped_array_payload}'::jsonb))"
        )

    def _build_regex_clause(self, field: str, operand: Any) -> str:
        """Apply PostgreSQL regex operators to strings/arrays, honoring the optional 'i' flag."""
        pattern, case_insensitive = self._normalize_regex_operand(operand, field)
        regex_operator = "~*" if case_insensitive else "~"

        escaped_pattern = pattern.replace("\\", "\\\\").replace("'", "''")
        field_key = self._escape_single_quotes(field)

        base_clause = f"((doc_metadata ->> '{field_key}') {regex_operator} '{escaped_pattern}')"
        array_clause = self._build_array_regex_clause(field, regex_operator, escaped_pattern)
        if array_clause:
            return f"({base_clause} OR {array_clause})"
        return base_clause

    def _normalize_regex_operand(self, operand: Any, field: str) -> tuple[str, bool]:
        """Validate regex operands; accept strings or {'pattern','flags'} with only the 'i' flag."""
        if isinstance(operand, str):
            return operand, False

        if isinstance(operand, dict):
            pattern = operand.get("pattern")
            if not isinstance(pattern, str) or not pattern:
                raise InvalidMetadataFilterError(f"$regex operator for field '{field}' expects a non-empty pattern.")

            flags = operand.get("flags", "")
            if not isinstance(flags, str):
                raise InvalidMetadataFilterError(f"$regex operator for field '{field}' expects flags to be a string.")

            unsupported_flags = {flag for flag in flags if flag not in {"", "i"}}
            if unsupported_flags:
                raise InvalidMetadataFilterError(
                    f"$regex operator for field '{field}' does not support flags: {', '.join(sorted(unsupported_flags))}."
                )

            return pattern, "i" in flags

        raise InvalidMetadataFilterError(
            f"$regex operator for field '{field}' expects a string or object with 'pattern'."
        )

    def _build_array_regex_clause(self, field: str, regex_operator: str, escaped_pattern: str) -> str:
        """Run regex comparisons against each JSON array element."""
        field_key = self._escape_single_quotes(field)
        array_value_expr = "trim('\"' FROM arr.value::text)"
        return (
            f"((jsonb_typeof(doc_metadata -> '{field_key}') = 'array') AND EXISTS ("
            f"SELECT 1 FROM jsonb_array_elements(doc_metadata -> '{field_key}') AS arr(value) "
            f"WHERE jsonb_typeof(arr.value) = 'string' AND {array_value_expr} {regex_operator} '{escaped_pattern}'))"
        )

    def _build_contains_clause(self, field: str, operand: Any) -> str:
        """Perform substring matching with LIKE/ILIKE, defaulting to case-insensitive array-aware checks."""
        value, case_sensitive = self._normalize_contains_operand(operand, field)
        like_operator = "LIKE" if case_sensitive else "ILIKE"

        escaped_pattern = self._escape_like_pattern(value)
        field_key = self._escape_single_quotes(field)

        base_clause = f"((doc_metadata ->> '{field_key}') {like_operator} '%{escaped_pattern}%')"
        array_clause = self._build_array_like_clause(field, like_operator, escaped_pattern)
        if array_clause:
            return f"({base_clause} OR {array_clause})"
        return base_clause

    def _normalize_contains_operand(self, operand: Any, field: str) -> tuple[str, bool]:
        """Validate substring operands; accept strings or {'value','case_sensitive'}."""
        if isinstance(operand, str):
            return operand, False

        if isinstance(operand, dict):
            value = operand.get("value")
            if not isinstance(value, str) or not value:
                raise InvalidMetadataFilterError(
                    f"$contains operator for field '{field}' expects a non-empty string value."
                )
            case_sensitive = operand.get("case_sensitive", False)
            if not isinstance(case_sensitive, bool):
                raise InvalidMetadataFilterError(
                    f"$contains operator for field '{field}' expects 'case_sensitive' to be a boolean."
                )
            return value, case_sensitive

        raise InvalidMetadataFilterError(
            f"$contains operator for field '{field}' expects a string or object with 'value'."
        )

    def _escape_like_pattern(self, value: str) -> str:
        """Escape wildcard characters for SQL LIKE/ILIKE patterns."""
        escaped = value.replace("\\", "\\\\")
        escaped = escaped.replace("%", "\\%").replace("_", "\\_")
        return escaped.replace("'", "''")

    def _build_array_like_clause(self, field: str, like_operator: str, escaped_pattern: str) -> str:
        """Apply LIKE/ILIKE matching to each string element in JSON arrays."""
        field_key = self._escape_single_quotes(field)
        array_value_expr = "trim('\"' FROM arr.value::text)"
        return (
            f"((jsonb_typeof(doc_metadata -> '{field_key}') = 'array') AND EXISTS ("
            f"SELECT 1 FROM jsonb_array_elements(doc_metadata -> '{field_key}') AS arr(value) "
            f"WHERE jsonb_typeof(arr.value) = 'string' AND "
            f"{array_value_expr} {like_operator} '%{escaped_pattern}%'))"
        )

    def _metadata_type_expr(self, field_key: str) -> str:
        """Return SQL expression fetching the stored metadata type for a field."""
        return f"(metadata_types ->> '{field_key}')"

    def _map_comparison_operator(self, operator: str) -> str:
        """Map comparison operators to SQL symbols."""
        mapping = {"$eq": "=", "$ne": "=", "$gt": ">", "$gte": ">=", "$lt": "<", "$lte": "<="}
        return mapping[operator]

    def _format_numeric_literal(self, operand: Any) -> str:
        """Serialize numeric operands into SQL-safe literals."""
        if isinstance(operand, bool) or operand is None:
            raise InvalidMetadataFilterError("Numeric comparisons require a numeric operand.")

        if isinstance(operand, (int, float)):
            text = str(operand)
        elif isinstance(operand, str):
            text = operand.strip()
            if not text:
                raise InvalidMetadataFilterError("Numeric comparisons require a numeric operand.")
        else:
            raise InvalidMetadataFilterError("Numeric comparisons require a numeric operand.")

        try:
            value = Decimal(text)
        except (InvalidOperation, ValueError) as exc:  # noqa: BLE001
            raise InvalidMetadataFilterError(f"'{operand}' is not a valid numeric literal.") from exc

        normalized = format(value.normalize(), "f")
        if "." in normalized:
            normalized = normalized.rstrip("0").rstrip(".")
        return normalized or "0"

    def _format_datetime_literal(self, operand: Any, field: str) -> str:
        """Serialize datetime operands into SQL timestamptz literals."""
        iso_value = self._coerce_datetime_string(operand, field, is_date=False)
        escaped = self._escape_single_quotes(iso_value)
        return f"'{escaped}'::timestamptz"

    def _format_date_literal(self, operand: Any, field: str) -> str:
        """Serialize date operands into SQL date literals."""
        iso_value = self._coerce_datetime_string(operand, field, is_date=True)
        escaped = self._escape_single_quotes(iso_value)
        return f"'{escaped}'::date"

    def _coerce_datetime_string(self, operand: Any, field: str, is_date: bool) -> str:
        """Convert supported operand types into ISO date/datetime strings."""
        if isinstance(operand, datetime):
            if is_date:
                return operand.date().isoformat()
            dt_value = operand
        elif isinstance(operand, date):
            if is_date:
                return operand.isoformat()
            dt_value = datetime(operand.year, operand.month, operand.day)
        elif isinstance(operand, str):
            text = operand.strip()
            if not text:
                raise InvalidMetadataFilterError(
                    f"Comparison operator for field '{field}' expects a non-empty ISO8601 string."
                )
            try:
                if text.endswith("Z"):
                    text = text[:-1] + "+00:00"
                if is_date:
                    return date.fromisoformat(text.split("T", 1)[0]).isoformat()
                dt_value = datetime.fromisoformat(text)
            except ValueError as exc:
                raise InvalidMetadataFilterError(
                    f"Value '{operand}' for field '{field}' is not a valid ISO8601 date/datetime."
                ) from exc
        else:
            raise InvalidMetadataFilterError(
                f"Comparison operator for field '{field}' expects a string or datetime/date object."
            )

        return dt_value.isoformat()

    @staticmethod
    def _escape_single_quotes(value: str) -> str:
        """Escape single quotes for SQL literals."""
        return value.replace("'", "''")


class TextColumnFilterBuilder:
    """Translate filter expressions into SQL for a single text column."""

    def __init__(self, column: str):
        self._column = column

    def build(self, filters: Optional[Dict[str, Any]]) -> str:
        """Construct a SQL WHERE clause from a filter dictionary."""
        if filters is None:
            return ""

        if not isinstance(filters, dict):
            raise InvalidMetadataFilterError("Filename filters must be provided as a JSON object.")

        if not filters:
            return ""

        clause = self._parse_filter(filters, context="filename filter")
        if not clause:
            raise InvalidMetadataFilterError("Filename filter produced no valid conditions.")
        return clause

    def _parse_filter(self, expression: Any, context: str) -> str:
        """Recursively parse an operator filter into SQL."""
        if isinstance(expression, dict):
            if not expression:
                raise InvalidMetadataFilterError(f"{context.capitalize()} cannot be empty.")

            clauses: List[str] = []
            for key, value in expression.items():
                if key == "$and":
                    if not isinstance(value, list):
                        raise InvalidMetadataFilterError("$and operator expects a non-empty list of conditions.")
                    clauses.append(
                        self._combine_clauses(
                            [self._parse_filter(item, context="$and condition") for item in value],
                            "AND",
                            'operator "$and"',
                        )
                    )
                elif key == "$or":
                    if not isinstance(value, list):
                        raise InvalidMetadataFilterError("$or operator expects a non-empty list of conditions.")
                    clauses.append(
                        self._combine_clauses(
                            [self._parse_filter(item, context="$or condition") for item in value],
                            "OR",
                            'operator "$or"',
                        )
                    )
                elif key == "$nor":
                    if not isinstance(value, list):
                        raise InvalidMetadataFilterError("$nor operator expects a non-empty list of conditions.")
                    inner = self._combine_clauses(
                        [self._parse_filter(item, context="$nor condition") for item in value],
                        "OR",
                        'operator "$nor"',
                    )
                    clauses.append(f"(NOT {inner})")
                elif key == "$not":
                    sub_context = 'operator "$not"'
                    clauses.append(f"(NOT {self._parse_filter(value, context=sub_context)})")
                else:
                    clauses.append(self._build_operator_clause(key, value))

            return self._combine_clauses(clauses, "AND", context)

        if isinstance(expression, list):
            if not expression:
                raise InvalidMetadataFilterError(f"{context.capitalize()} cannot be an empty list.")
            subclauses = [self._parse_filter(item, context="nested condition") for item in expression]
            return self._combine_clauses(subclauses, "OR", context)

        raise InvalidMetadataFilterError(f"{context.capitalize()} must be expressed as a JSON object.")

    def _combine_clauses(self, clauses: List[str], operator: str, context: str) -> str:
        """Combine multiple SQL clauses with a logical operator."""
        cleaned = [clause for clause in clauses if clause]
        if not cleaned:
            raise InvalidMetadataFilterError(f"No valid conditions supplied for {context}.")
        if len(cleaned) == 1:
            return cleaned[0]
        return "(" + f" {operator} ".join(cleaned) + ")"

    def _build_operator_clause(self, operator: str, operand: Any) -> str:
        """Build SQL clause for operator-based filename filters."""
        if operator in {"$eq", "$ne", "$gt", "$gte", "$lt", "$lte"}:
            return self._build_comparison_clause(operator, operand)
        if operator == "$in":
            return self._build_in_clause(operand, negate=False)
        if operator == "$nin":
            return self._build_in_clause(operand, negate=True)
        if operator == "$exists":
            return self._build_exists_clause(operand)
        if operator == "$regex":
            return self._build_regex_clause(operand)
        if operator == "$contains":
            return self._build_contains_clause(operand)
        raise InvalidMetadataFilterError(f"Unsupported filename filter operator '{operator}'.")

    def _build_comparison_clause(self, operator: str, operand: Any) -> str:
        """Build comparison clauses for the filename column."""
        column = self._column

        if operator == "$eq":
            if operand is None:
                return f"({column} IS NULL)"
            if not isinstance(operand, str):
                raise InvalidMetadataFilterError("Filename $eq operator expects a string value.")
            escaped = self._escape_single_quotes(operand)
            return f"({column} = '{escaped}')"

        if operator == "$ne":
            if operand is None:
                return f"({column} IS NOT NULL)"
            if not isinstance(operand, str):
                raise InvalidMetadataFilterError("Filename $ne operator expects a string value.")
            escaped = self._escape_single_quotes(operand)
            return f"({column} IS DISTINCT FROM '{escaped}')"

        if operand is None or not isinstance(operand, str):
            raise InvalidMetadataFilterError(f"Filename {operator} operator expects a string value.")

        escaped = self._escape_single_quotes(operand)
        sql_operator = self._map_comparison_operator(operator)
        return f"({column} {sql_operator} '{escaped}')"

    def _build_in_clause(self, operand: Any, negate: bool) -> str:
        """Build IN/NOT IN clauses with NULL handling."""
        if not isinstance(operand, list) or not operand:
            raise InvalidMetadataFilterError("Filename $in/$nin operator expects a non-empty list of values.")

        has_null = any(item is None for item in operand)
        values = [item for item in operand if item is not None]

        escaped_values: List[str] = []
        for item in values:
            if not isinstance(item, str):
                raise InvalidMetadataFilterError("Filename $in/$nin operator expects string values.")
            escaped_values.append(f"'{self._escape_single_quotes(item)}'")

        column = self._column
        in_list = ", ".join(escaped_values)

        if not negate:
            clauses: List[str] = []
            if escaped_values:
                clauses.append(f"({column} IN ({in_list}))")
            if has_null:
                clauses.append(f"({column} IS NULL)")
            return self._combine_clauses(clauses, "OR", "filename $in operator")

        if has_null:
            if escaped_values:
                return f"(({column} IS NOT NULL) AND ({column} NOT IN ({in_list})))"
            return f"({column} IS NOT NULL)"

        return f"(({column} IS NULL) OR ({column} NOT IN ({in_list})))"

    def _build_exists_clause(self, operand: Any) -> str:
        """Build clause handling $exists operator."""
        expected = operand
        if isinstance(expected, str):
            expected = expected.lower() in {"1", "true", "yes"}
        elif isinstance(expected, (int, float)):
            expected = bool(expected)
        elif not isinstance(expected, bool):
            raise InvalidMetadataFilterError("Filename $exists operator expects a boolean value.")

        column = self._column
        return f"({column} IS NOT NULL)" if expected else f"({column} IS NULL)"

    def _build_regex_clause(self, operand: Any) -> str:
        """Apply PostgreSQL regex operators to the filename column."""
        pattern, case_insensitive = self._normalize_regex_operand(operand)
        regex_operator = "~*" if case_insensitive else "~"

        escaped_pattern = pattern.replace("\\", "\\\\").replace("'", "''")
        return f"({self._column} {regex_operator} '{escaped_pattern}')"

    def _normalize_regex_operand(self, operand: Any) -> tuple[str, bool]:
        """Validate regex operands; accept strings or {'pattern','flags'} with only the 'i' flag."""
        if isinstance(operand, str):
            return operand, False

        if isinstance(operand, dict):
            pattern = operand.get("pattern")
            if not isinstance(pattern, str) or not pattern:
                raise InvalidMetadataFilterError("Filename $regex operator expects a non-empty pattern.")

            flags = operand.get("flags", "")
            if not isinstance(flags, str):
                raise InvalidMetadataFilterError("Filename $regex operator expects flags to be a string.")

            unsupported_flags = {flag for flag in flags if flag not in {"", "i"}}
            if unsupported_flags:
                raise InvalidMetadataFilterError(
                    f"Filename $regex operator does not support flags: {', '.join(sorted(unsupported_flags))}."
                )

            return pattern, "i" in flags

        raise InvalidMetadataFilterError("Filename $regex operator expects a string or object with 'pattern'.")

    def _build_contains_clause(self, operand: Any) -> str:
        """Perform substring matching with LIKE/ILIKE."""
        value, case_sensitive = self._normalize_contains_operand(operand)
        like_operator = "LIKE" if case_sensitive else "ILIKE"

        escaped_pattern = self._escape_like_pattern(value)
        return f"({self._column} {like_operator} '%{escaped_pattern}%')"

    def _normalize_contains_operand(self, operand: Any) -> tuple[str, bool]:
        """Validate substring operands; accept strings or {'value','case_sensitive'}."""
        if isinstance(operand, str):
            return operand, False

        if isinstance(operand, dict):
            value = operand.get("value")
            if not isinstance(value, str) or not value:
                raise InvalidMetadataFilterError("Filename $contains operator expects a non-empty string value.")
            case_sensitive = operand.get("case_sensitive", False)
            if not isinstance(case_sensitive, bool):
                raise InvalidMetadataFilterError(
                    "Filename $contains operator expects 'case_sensitive' to be a boolean."
                )
            return value, case_sensitive

        raise InvalidMetadataFilterError("Filename $contains operator expects a string or object with 'value'.")

    def _escape_like_pattern(self, value: str) -> str:
        """Escape wildcard characters for SQL LIKE/ILIKE patterns."""
        escaped = value.replace("\\", "\\\\")
        escaped = escaped.replace("%", "\\%").replace("_", "\\_")
        return escaped.replace("'", "''")

    def _map_comparison_operator(self, operator: str) -> str:
        """Map comparison operators to SQL symbols."""
        mapping = {"$gt": ">", "$gte": ">=", "$lt": "<", "$lte": "<="}
        return mapping[operator]

    @staticmethod
    def _escape_single_quotes(value: str) -> str:
        """Escape single quotes for SQL literals."""
        return value.replace("'", "''")
