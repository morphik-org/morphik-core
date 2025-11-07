# Changelog

All notable changes to the Morphik Python SDK will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2024-01-15

### 🎉 Major Release

First stable 1.0 release of the Morphik Python SDK.

### ⚠️ BREAKING CHANGES

#### `list_documents()` Return Type Changed

The `list_documents()` method now returns `ListDocsResponse` instead of `List[Document]`.

**Migration:**
```python
# Before (v0.x)
docs = db.list_documents(limit=10)
for doc in docs:
    print(doc.filename)

# After (v1.0)
response = db.list_documents(limit=10)
for doc in response.documents:  # Access via .documents
    print(doc.filename)
```

**Why:** The new response includes pagination metadata, aggregates, and more control over results.

### ✨ Added

**Advanced Pagination:**
- `total_count` - Total matching documents (with `include_total_count=True`)
- `has_more` - Boolean indicating more results exist
- `next_skip` - Skip value for next page
- `returned_count` - Number of documents in current response

**Aggregates & Counts:**
- `include_status_counts` - Document counts by processing status
- `include_folder_counts` - Document counts by folder

**Sorting:**
- `sort_by` - Sort by `created_at`, `updated_at`, `filename`, or `external_id`
- `sort_direction` - Sort `asc` or `desc`

**Filtering:**
- `completed_only` - Filter to only completed documents

### 📝 Examples

```python
# Pagination with total count
response = db.list_documents(limit=10, include_total_count=True)
print(f"Showing {response.returned_count} of {response.total_count}")

if response.has_more:
    next_page = db.list_documents(skip=response.next_skip, limit=10)

# Sorting
response = db.list_documents(sort_by="created_at", sort_direction="asc")

# Status aggregates
response = db.list_documents(include_status_counts=True)
print(response.status_counts)  # {"completed": 42, "processing": 3, ...}
```

### 🐛 Fixed

- Fixed `UserScope.list_documents()` filter handling - filters now properly applied across all scopes
- Consolidated list_documents implementation to use shared internal helpers

---

## [0.2.15] - 2024-01-10

Previous release. See git history for older changes.
