import json
from typing import Any, Dict, List, Optional


class InvalidMetadataFilterError(ValueError):
    """Raised when metadata filters are malformed or unsupported."""


class MetadataFilterBuilder:
    """Translate JSON-style metadata filters into SQL, covering arrays, regex, and substring operators."""

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
            if operator == "$eq":
                clauses.append(self._build_single_value_clause(field, operand))
            elif operator == "$ne":
                clauses.append(f"(NOT {self._build_single_value_clause(field, operand)})")
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

    def _jsonb_contains_clause(self, field: str, value: Any) -> str:
        """Build JSONB containment clause for a field/value pairing."""
        try:
            json_payload = json.dumps({field: value})
        except (TypeError, ValueError) as exc:  # noqa: BLE001
            raise InvalidMetadataFilterError(
                f"Metadata filter for field '{field}' contains a non-serializable value: {exc}"
            ) from exc

        escaped_payload = json_payload.replace("'", "''")
        base_clause = f"(doc_metadata @> '{escaped_payload}'::jsonb)"

        array_clause = self._build_array_membership_clause(field, value)
        if array_clause:
            return f"({base_clause} OR {array_clause})"
        return base_clause

    def _build_array_membership_clause(self, field: str, value: Any) -> str:
        """Match scalar comparisons against array-valued metadata fields."""
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

    @staticmethod
    def _escape_single_quotes(value: str) -> str:
        """Escape single quotes for SQL literals."""
        return value.replace("'", "''")
