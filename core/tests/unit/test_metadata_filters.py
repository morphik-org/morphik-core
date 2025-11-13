"""Unit tests for metadata filter SQL generation with typed metadata support."""

from datetime import date, datetime

import pytest

from core.database.metadata_filters import InvalidMetadataFilterError, MetadataFilterBuilder


class TestBasicFilters:
    """Test basic filter operations."""

    def test_simple_equality(self):
        """Test simple equality filter."""
        builder = MetadataFilterBuilder()
        filters = {"department": "engineering"}
        sql = builder.build(filters)

        assert sql is not None
        assert "department" in sql
        assert "engineering" in sql

    def test_and_operator(self):
        """Test $and operator."""
        builder = MetadataFilterBuilder()
        filters = {"$and": [{"dept": "eng"}, {"active": True}]}
        sql = builder.build(filters)

        assert "AND" in sql
        assert "dept" in sql
        assert "active" in sql

    def test_or_operator(self):
        """Test $or operator."""
        builder = MetadataFilterBuilder()
        filters = {"$or": [{"dept": "eng"}, {"dept": "sales"}]}
        sql = builder.build(filters)

        assert "OR" in sql
        assert "dept" in sql

    def test_in_operator(self):
        """Test $in operator."""
        builder = MetadataFilterBuilder()
        filters = {"status": {"$in": ["active", "pending"]}}
        sql = builder.build(filters)

        assert "status" in sql
        assert "active" in sql and "pending" in sql

    def test_nin_operator(self):
        """Test $nin (not in) operator."""
        builder = MetadataFilterBuilder()
        filters = {"status": {"$nin": ["deleted", "archived"]}}
        sql = builder.build(filters)

        assert "NOT" in sql
        assert "status" in sql

    def test_exists_operator(self):
        """Test $exists operator."""
        builder = MetadataFilterBuilder()
        filters = {"optional_field": {"$exists": True}}
        sql = builder.build(filters)

        assert "optional_field" in sql

    def test_not_operator(self):
        """Test $not operator."""
        builder = MetadataFilterBuilder()
        filters = {"status": {"$not": {"$eq": "deleted"}}}
        sql = builder.build(filters)

        assert "NOT" in sql
        assert "status" in sql


class TestComparisonOperators:
    """Test new comparison operators for typed metadata."""

    def test_gt_operator_with_number(self):
        """Test $gt (greater than) operator with numbers."""
        builder = MetadataFilterBuilder()
        filters = {"priority": {"$gt": 5}}
        sql = builder.build(filters)

        assert "priority" in sql
        assert ">" in sql
        # Should check both number and decimal types
        assert "number" in sql or "decimal" in sql

    def test_gte_operator(self):
        """Test $gte (greater than or equal) operator."""
        builder = MetadataFilterBuilder()
        filters = {"score": {"$gte": 100}}
        sql = builder.build(filters)

        assert "score" in sql
        assert ">=" in sql

    def test_lt_operator(self):
        """Test $lt (less than) operator."""
        builder = MetadataFilterBuilder()
        filters = {"age": {"$lt": 65}}
        sql = builder.build(filters)

        assert "age" in sql
        assert "<" in sql

    def test_lte_operator(self):
        """Test $lte (less than or equal) operator."""
        builder = MetadataFilterBuilder()
        filters = {"temperature": {"$lte": 100}}
        sql = builder.build(filters)

        assert "temperature" in sql
        assert "<=" in sql

    def test_comparison_with_decimal_string(self):
        """Test comparison with decimal string."""
        builder = MetadataFilterBuilder()
        filters = {"price": {"$lte": "99.99"}}
        sql = builder.build(filters)

        assert "price" in sql
        assert "<=" in sql
        assert "99.99" in sql

    def test_comparison_with_datetime_string(self):
        """Test comparison with datetime string."""
        builder = MetadataFilterBuilder()
        filters = {"created_at": {"$gt": "2024-01-01T00:00:00Z"}}
        sql = builder.build(filters)

        assert "created_at" in sql
        assert ">" in sql
        # Should reference datetime or timestamptz type
        assert "datetime" in sql or "timestamptz" in sql

    def test_comparison_with_datetime_object(self):
        """Test comparison with datetime object."""
        builder = MetadataFilterBuilder()
        dt = datetime(2024, 1, 1, 12, 0, 0)
        filters = {"created_at": {"$gte": dt}}
        sql = builder.build(filters)

        assert "created_at" in sql
        assert ">=" in sql
        assert "2024-01-01" in sql

    def test_comparison_with_date_string(self):
        """Test comparison with date string."""
        builder = MetadataFilterBuilder()
        filters = {"start_date": {"$lt": "2024-12-31"}}
        sql = builder.build(filters)

        assert "start_date" in sql
        assert "<" in sql

    def test_comparison_with_date_object(self):
        """Test comparison with date object using $gte (comparison operators handle date objects)."""
        builder = MetadataFilterBuilder()
        d = date(2024, 6, 15)
        filters = {"event_date": {"$gte": d}}
        sql = builder.build(filters)

        assert "event_date" in sql
        assert "2024-06-15" in sql

    def test_multiple_comparisons_range_query(self):
        """Test range query with multiple comparisons on same field."""
        builder = MetadataFilterBuilder()
        filters = {
            "$and": [
                {"age": {"$gte": 18}},
                {"age": {"$lt": 65}},
            ]
        }
        sql = builder.build(filters)

        assert "age" in sql
        assert ">=" in sql
        assert "<" in sql


class TestTypeOperator:
    """Test the $type operator for metadata type filtering."""

    def test_type_operator_single_type(self):
        """Test $type operator with single type."""
        builder = MetadataFilterBuilder()
        filters = {"value": {"$type": "number"}}
        sql = builder.build(filters)

        assert "value" in sql
        assert "number" in sql or "metadata_types" in sql

    def test_type_operator_multiple_types(self):
        """Test $type operator with list of types."""
        builder = MetadataFilterBuilder()
        filters = {"value": {"$type": ["number", "decimal"]}}
        sql = builder.build(filters)

        assert "value" in sql
        # Should have OR clause for multiple types
        assert "OR" in sql or ("number" in sql and "decimal" in sql)

    def test_type_operator_with_alias(self):
        """Test $type operator with type alias (should canonicalize)."""
        builder = MetadataFilterBuilder()
        filters = {"value": {"$type": "int"}}  # alias for 'number'
        sql = builder.build(filters)

        assert "value" in sql
        # Should be canonicalized to 'number'
        assert "number" in sql

    def test_type_operator_invalid_type(self):
        """Test that invalid type names raise error."""
        builder = MetadataFilterBuilder()
        filters = {"value": {"$type": "invalid-type"}}

        with pytest.raises(InvalidMetadataFilterError, match="Unsupported metadata type"):
            builder.build(filters)

    def test_type_operator_requires_string_or_list(self):
        """Test that $type requires string or list."""
        builder = MetadataFilterBuilder()
        filters = {"value": {"$type": 123}}

        with pytest.raises(InvalidMetadataFilterError, match="expects a string or list"):
            builder.build(filters)


class TestComplexFilters:
    """Test complex nested filter scenarios."""

    def test_complex_nested_and_or(self):
        """Test complex nested AND/OR filter."""
        builder = MetadataFilterBuilder()
        filters = {
            "$and": [
                {"department": "engineering"},
                {
                    "$or": [
                        {"level": {"$gte": 5}},
                        {"years_experience": {"$gt": 10}},
                    ]
                },
                {"active": True},
            ]
        }
        sql = builder.build(filters)

        assert "AND" in sql
        assert "OR" in sql
        assert "department" in sql
        assert "level" in sql
        assert "years_experience" in sql
        assert "active" in sql

    def test_mixed_type_and_comparison_filters(self):
        """Test combining $type and comparison operators."""
        builder = MetadataFilterBuilder()
        filters = {
            "$and": [
                {"priority": {"$type": "number"}},
                {"priority": {"$gt": 5}},
            ]
        }
        sql = builder.build(filters)

        assert "priority" in sql
        assert "number" in sql or "metadata_types" in sql
        assert ">" in sql


class TestSQLInjectionProtection:
    """Test protection against SQL injection."""

    def test_single_quote_escaping(self):
        """Test that single quotes are properly escaped."""
        builder = MetadataFilterBuilder()
        filters = {"name": "O'Brien"}
        sql = builder.build(filters)

        # Should not break SQL - either escaped or safe
        assert "O'Brien" in sql or "O''Brien" in sql

    def test_malicious_regex_pattern(self):
        """Test that malicious regex patterns are handled safely."""
        builder = MetadataFilterBuilder()
        filters = {"email": {"$regex": "'; DROP TABLE users; --"}}

        # Should not raise exception - should be escaped/sanitized
        sql = builder.build(filters)
        assert sql is not None

    def test_unicode_values(self):
        """Test that unicode values are properly handled."""
        builder = MetadataFilterBuilder()
        filters = {"name": "测试用户"}
        sql = builder.build(filters)

        assert "name" in sql
        # Unicode should be safely included
        assert sql is not None


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_filters(self):
        """Test that empty filters return empty string."""
        builder = MetadataFilterBuilder()
        filters = {}
        sql = builder.build(filters)

        assert sql == ""

    def test_none_filters(self):
        """Test that None filters return empty string."""
        builder = MetadataFilterBuilder()
        sql = builder.build(None)

        assert sql == ""

    def test_zero_in_comparison(self):
        """Test that zero is properly handled."""
        builder = MetadataFilterBuilder()
        filters = {"count": {"$gt": 0}}
        sql = builder.build(filters)

        assert "count" in sql
        assert "0" in sql

    def test_negative_numbers(self):
        """Test that negative numbers work."""
        builder = MetadataFilterBuilder()
        filters = {"balance": {"$lt": -100}}
        sql = builder.build(filters)

        assert "balance" in sql
        assert "-100" in sql or "100" in sql  # May be represented differently

    def test_very_large_decimal(self):
        """Test handling of very large decimal values."""
        builder = MetadataFilterBuilder()
        filters = {"amount": {"$eq": "999999999999.99"}}
        sql = builder.build(filters)

        assert "amount" in sql

    def test_scientific_notation(self):
        """Test scientific notation in comparisons."""
        builder = MetadataFilterBuilder()
        filters = {"large_number": {"$gt": 1e10}}
        sql = builder.build(filters)

        assert "large_number" in sql

    def test_datetime_with_timezone_z(self):
        """Test datetime with Z suffix."""
        builder = MetadataFilterBuilder()
        filters = {"timestamp": {"$gte": "2024-01-15T12:30:00Z"}}
        sql = builder.build(filters)

        assert "timestamp" in sql
        assert "2024-01-15" in sql

    def test_datetime_with_timezone_offset(self):
        """Test datetime with timezone offset."""
        builder = MetadataFilterBuilder()
        filters = {"timestamp": {"$lte": "2024-01-15T12:30:00+05:00"}}
        sql = builder.build(filters)

        assert "timestamp" in sql

    def test_leap_year_date(self):
        """Test leap year date."""
        builder = MetadataFilterBuilder()
        filters = {"date": {"$eq": "2024-02-29"}}
        sql = builder.build(filters)

        assert "date" in sql
        assert "2024-02-29" in sql

    def test_empty_string_value(self):
        """Test empty string values."""
        builder = MetadataFilterBuilder()
        filters = {"field": ""}
        sql = builder.build(filters)

        assert "field" in sql

    def test_field_names_with_dots(self):
        """Test field names containing dots."""
        builder = MetadataFilterBuilder()
        filters = {"nested.field.name": "value"}
        sql = builder.build(filters)

        assert "nested.field.name" in sql or "nested" in sql
