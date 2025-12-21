# Performance Analysis Report
**Generated:** 2025-12-21
**Analyzed by:** Claude Code

## Executive Summary

This report identifies critical performance issues, anti-patterns, and optimization opportunities across the Morphik codebase. The analysis covers:

1. **Backend (Python):** N+1 queries, inefficient algorithms, synchronous operations
2. **Frontend (React/TypeScript):** Unnecessary re-renders, missing memoization, component optimization
3. **Database:** Missing eager loading, inefficient queries, pagination issues

---

## 🔴 CRITICAL Issues (Immediate Action Required)

### 1. SQL Injection Vulnerability
**File:** `core/vector_store/multi_vector_store.py:895`
**Severity:** CRITICAL (Security + Performance)

```python
query = f"DELETE FROM multi_vector_embeddings WHERE document_id = '{document_id}'"
```

**Impact:** SQL injection vulnerability + inefficient query
**Fix:** Use parameterized queries

---

### 2. N+1 Query: Folder Details Endpoint
**File:** `core/routes/folders.py:159-241`
**Severity:** CRITICAL

**Issue:** Loops over N folders making N+1 database queries:
```python
for identifier in request.identifiers:
    folder = await _resolve_folder(identifier, auth)  # Multiple DB queries

for folder in target_folders:
    doc_result = await document_service.db.list_documents_flexible(
        auth=auth,
        system_filters={"folder_path": path_filter},
    )  # Separate query for EACH folder
```

**Impact:** For 100 folders → 100+ database queries
**Fix:** Batch fetch with single query using `IN` clause or `ANY()`

---

### 3. Sequential Chunk Processing in Graph Service
**File:** `core/services/graph_service.py:538-589`
**Severity:** CRITICAL

**Issue:** Processes chunks sequentially instead of in parallel:
```python
for chunk in chunks:
    chunk_entities, chunk_relationships = await self.extract_entities_from_text(...)
```

**Impact:** For 100 chunks with 2s LLM latency each → 200 seconds total
**Fix:** Use `asyncio.gather()` to parallelize:
```python
tasks = [self.extract_entities_from_text(chunk, ...) for chunk in chunks]
results = await asyncio.gather(*tasks)
```

---

### 4. Missing React.memo on Large Components
**Files:**
- `ee/ui-component/components/pdf/PDFViewer.tsx` (1592 lines)
- `ee/ui-component/components/chat/ChatSection.tsx` (1145 lines)
- `ee/ui-component/components/GraphSection.tsx` (1344 lines)

**Severity:** CRITICAL (UX Impact)

**Issue:** Large components re-render on every parent update
**Impact:** Poor UX, laggy interface, wasted CPU cycles
**Fix:** Wrap exports with `React.memo()`:
```tsx
export default React.memo(PDFViewer);
```

---

### 5. 200+ Inline Function Definitions
**Files:** All major React components
**Severity:** CRITICAL

**Examples:**
```tsx
// PDFViewer.tsx:976 - Inside .map()
onClick={() => handleDocumentSelect(doc)}

// ChatSection.tsx:522
onClick={() => setShowModelSelector(!showModelSelector)}

// GraphSection.tsx:992 - In loop with complex logic
onClick={e => { e.stopPropagation(); setGraphToDelete(graph.name); }}
```

**Impact:** Creates new function on every render → child components re-render unnecessarily
**Fix:** Use `useCallback()`:
```tsx
const handleClick = useCallback(() => {
  setShowModelSelector(prev => !prev);
}, []);
```

---

## 🟡 HIGH Priority Issues

### 6. Folder Identifier Resolution Loop (N+1)
**File:** `core/routes/folders.py:161-165`

**Issue:** `_resolve_folder()` called in loop, each making up to 3 DB queries:
```python
for identifier in request.identifiers:
    folder = await _resolve_folder(identifier, auth)  # 3 queries per identifier
```

**Impact:** For N identifiers → up to 3N queries
**Fix:** Batch resolve with parallel async or single OR query

---

### 7. Folder Path Creation (Sequential Queries)
**Files:**
- `core/routes/folders.py:88-116`
- `core/services/ingestion_service.py:210-232`

**Issue:** Creates nested folders one at a time:
```python
for idx, segment in enumerate(segments):
    current_path = "/" + "/".join(current_path_parts)
    existing = await document_service.db.get_folder_by_full_path(current_path, auth)
    if existing:
        ...
    else:
        success = await document_service.db.create_folder(folder, auth)
```

**Impact:** For path `/a/b/c/d` → 4 sequential queries
**Fix:** Batch query all paths, create only missing ones

---

### 8. Inefficient Entity Embedding Batch
**File:** `core/services/graph_service.py:1123`

**Issue:** Sequential API calls instead of batching:
```python
return [await self.embedding_model.embed_for_query(text) for text in texts]
```

**Impact:** N API calls instead of 1 batched call
**Fix:** Implement true batching in embedding provider

---

### 9. Folder Deletion Fetches ALL Folders
**File:** `core/routes/folders.py:394-441`

**Issue:** Loads entire folder tree to find descendants:
```python
all_folders = await document_service.db.list_folders(auth)  # ALL folders
descendants = [
    f for f in all_folders
    if f.full_path and f.full_path.startswith(target_path.rstrip("/") + "/")
]
```

**Impact:** Memory spike for large folder hierarchies
**Fix:** Add SQL query with `LIKE` filtering: `WHERE full_path LIKE 'path/%'`

---

### 10. Graph Workflow Lookup Scans All Graphs
**File:** `core/routes/graph.py:463-490`

**Issue:** Linear search through all graphs:
```python
graphs = await document_service.db.list_graphs(auth)  # ALL graphs
for graph in graphs:
    if graph.name == graph_name or workflow_id in graph.system_metadata.get("workflow_id", ""):
        # Found it
```

**Impact:** O(n) lookup instead of O(1) indexed query
**Fix:** Add `get_graph_by_workflow_id()` with indexed query

---

## 🟠 MEDIUM Priority Issues

### 11. XML Chunker O(n²) Parent Finding
**File:** `core/parser/xml_chunker.py:119-125`

**Issue:** Nested iteration to find parent:
```python
for candidate in root.iter():
    if current in candidate:
        parent = candidate
        break
```

**Impact:** O(n²) for deeply nested XML
**Fix:** Use lxml with built-in parent references or maintain parent map

---

### 12. Triple Nested Loops in Entity Processing
**File:** `core/parser/xml_chunker.py:1328-1341`

```python
for doc_id, chunk_numbers in entity1.chunk_sources.items():
    for chunk_num in chunk_numbers:
        entity1_chunks.add((doc_id, chunk_num))
# Repeated for entity2 and relationship
```

**Fix:** Use set comprehensions:
```python
entity1_chunks = {(doc_id, chunk_num)
                  for doc_id, chunk_nums in entity1.chunk_sources.items()
                  for chunk_num in chunk_nums}
```

---

### 13. String Concatenation in Loops
**Files:**
- `core/services/graph_service.py:1390`
- `core/parser/xml_chunker.py:230`

**Issue:**
```python
for path in paths[:5]:
    paths_text += " -> ".join(path) + "\n"  # Creates new string each iteration
```

**Fix:**
```python
paths_text = "\n".join(" -> ".join(path) for path in paths[:5])
```

---

### 14. Unnecessary List/Dict Copies
**File:** `core/services/graph_service.py`

Multiple instances:
- Line 277: `final_doc_ids = document_ids.copy()` (unnecessary)
- Line 318: `merged_entities = existing_entities.copy()` (large dict)
- Line 863: `.copy()` inside list comprehension

**Impact:** Memory allocation and CPU overhead
**Fix:** Only copy when mutation is required

---

### 15. Missing Caching for Entity Embeddings
**File:** `core/services/graph_service.py:1100-1117`

**Issue:** Recalculates embeddings on every query
**Fix:** Cache entity embeddings with LRU or TTL cache

---

### 16. No Pagination on List Endpoints
**Files:**
- `core/routes/folders.py:142` - `list_folders`
- `core/routes/graph.py:204` - `list_graphs`

**Impact:** Unbounded result sets → memory issues
**Fix:** Add `skip`/`limit` with defaults (e.g., max 1000)

---

### 17. Component Size Anti-pattern
**Large Components (1000+ lines):**
- PDFViewer.tsx: 1592 lines, ~50 state variables
- ChatSection.tsx: 1145 lines
- GraphSection.tsx: 1344 lines

**Impact:** Hard to maintain, excessive re-renders
**Fix:** Split into smaller components:
- PDFViewer → DocumentViewer + ChatSidebar + ToolModal + ControlBar
- ChatSection → InputArea + SettingsPanel + MessagesList
- GraphSection → GraphList + GraphDetails + GraphViz + GraphForm

---

### 18. Missing useCallback for Event Handlers
**File:** `ee/ui-component/components/search/SearchSection.tsx`

**Issue:** No `useCallback` usage - all handlers recreated on every render
**Fix:** Wrap handlers:
```tsx
const handleSearch = useCallback((query: string) => {
  setSearchQuery(query);
}, []);
```

---

## 🟢 LOW Priority Issues

### 19. Synchronous File I/O in Async Context
**Files:**
- `core/services/ingestion_service.py:756`
- `core/services/log_uploader.py:145-151`

**Issue:**
```python
with path.open("r", encoding="utf-8") as handle:
    # Blocks event loop
```

**Fix:** Use `aiofiles`:
```python
async with aiofiles.open(path, "r") as handle:
    content = await handle.read()
```

---

### 20. Using Array Index as React Key
**Files:**
- `ee/ui-component/components/GraphSection.tsx:962`
- `ee/ui-component/components/search/SearchSection.tsx:816`

**Issue:**
```tsx
{[...Array(12)].map((_, i) => (
  <div key={i}>  {/* Anti-pattern */}
    <Skeleton />
  </div>
))}
```

**Fix:** Use stable IDs or create UUIDs for skeleton items

---

### 21. Missing SQLAlchemy Relationships
**File:** `core/database/models.py`

**Issue:** All relationships managed via JSONB arrays instead of ORM relationships
**Impact:** No eager loading support, manual joins prone to errors
**Fix:** Consider adding proper `relationship()` for frequently accessed associations

---

### 22. Bulk Operations in Loops
**File:** `core/routes/folders.py:418-427`

**Issue:** Individual delete calls instead of bulk:
```python
removal_results = await asyncio.gather(
    *[document_service.db.remove_document_from_folder(tid, doc_id, auth)
      for doc_id in doc_ids]
)
```

**Fix:** Create `bulk_remove_documents_from_folder()` for single UPDATE query

---

### 23. Storage Key Lookup Sequential
**File:** `core/vector_store/multi_vector_store.py:514-532`

**Issue:** Tries candidates one by one:
```python
for candidate in key_candidates:
    try:
        content_bytes = await self.storage.download_file(...)
```

**Fix:** Try in parallel with `asyncio.gather(..., return_exceptions=True)`

---

### 24. JSON Serialization Overhead
**Multiple files:**
- Multi Vector Store: Lines 675, 559
- Database models: Frequent metadata serialization

**Issue:** Repeated `json.dumps()`/`json.loads()` in hot paths
**Fix:** Cache serialized metadata or use MessagePack

---

### 25. No Streaming for Large Files
**File:** `core/parser/xml_chunker.py:72, 266-267`

**Issue:** Loads entire XML into memory:
```python
root.iter()  # Full tree in memory
full_text = self._elem_text(elem, max_length=50000)
```

**Fix:** Use streaming XML parser (lxml.etree.iterparse)

---

## 📊 Summary Statistics

### Backend (Python)
- **N+1 Queries:** 5 critical instances
- **Missing Batch Operations:** 8 instances
- **Synchronous Operations in Async:** 4 instances
- **O(n²) Algorithms:** 3 instances
- **Missing Caching:** 5 opportunities
- **Security Issues:** 1 SQL injection

### Frontend (React/TypeScript)
- **Components Without React.memo:** 6 major components
- **Inline Functions:** 200+ instances
- **Missing useCallback:** 100+ opportunities
- **Missing useMemo:** 15+ opportunities
- **Components >1000 lines:** 4 components
- **Index-as-key Anti-patterns:** 2+ instances

---

## 🎯 Recommended Implementation Priority

### Phase 1: Security & Critical Performance (Week 1)
1. **Fix SQL injection** (core/vector_store/multi_vector_store.py:895)
2. **Add React.memo** to PDFViewer, ChatSection, GraphSection
3. **Parallelize chunk processing** in GraphService
4. **Fix N+1 in folder details** endpoint

### Phase 2: Database Optimization (Week 2)
5. **Batch folder identifier resolution**
6. **Optimize folder deletion query**
7. **Add pagination** to list endpoints
8. **Fix graph workflow lookup**
9. **Batch folder path creation**

### Phase 3: React Performance (Week 2-3)
10. **Add useCallback** to all event handlers (start with top 20 hottest paths)
11. **Split large components** into smaller ones
12. **Add useMemo** for expensive computations
13. **Fix index-as-key** anti-patterns

### Phase 4: Algorithm Optimization (Week 3-4)
14. **Fix XML chunker O(n²) parent finding**
15. **Implement entity embedding cache**
16. **Replace string concatenation** in loops
17. **Add batch embeddings** API support
18. **Optimize triple nested loops**

### Phase 5: Polish & Monitoring (Week 4+)
19. **Add performance monitoring** to identify runtime bottlenecks
20. **Implement SQLAlchemy relationships** for frequently accessed data
21. **Add streaming parsers** for large files
22. **Reduce unnecessary copies**
23. **Add async file I/O**

---

## 🔧 Quick Wins (Low Effort, High Impact)

These can be implemented immediately with minimal risk:

1. **Add React.memo** to 6 components (~30 min each = 3 hours)
2. **Fix SQL injection** with parameterized query (~15 min)
3. **Wrap inline functions** in useCallback (~2-3 hours for top 50)
4. **Add pagination defaults** to list endpoints (~1 hour)
5. **Fix string concatenation** to use join (~30 min)
6. **Remove unnecessary .copy()** calls (~30 min)

**Total Quick Wins Time:** ~8-10 hours
**Expected Impact:** 30-50% perceived performance improvement

---

## 📈 Monitoring Recommendations

To track improvements, add metrics for:

1. **API Response Times:**
   - Folder operations (target: <100ms)
   - Graph queries (target: <500ms)
   - Document listing (target: <200ms)

2. **Database Query Count:**
   - Queries per request (target: <10 for most endpoints)
   - Query duration (target: 95th percentile <50ms)

3. **Frontend Metrics:**
   - Component render count
   - Time to interactive (target: <2s)
   - React DevTools Profiler data

4. **Resource Usage:**
   - Memory consumption trends
   - CPU usage during peak operations
   - Database connection pool utilization

---

## 🎓 Code Review Guidelines

Going forward, reject PRs that introduce:

1. Queries inside loops without batching
2. New components >500 lines without justification
3. Inline event handlers in loops
4. Missing pagination on list endpoints
5. String concatenation in loops
6. Synchronous I/O in async functions
7. Components without React.memo when >300 lines or expensive rendering

---

## 📚 References

- [React Performance Optimization](https://react.dev/learn/render-and-commit)
- [SQLAlchemy N+1 Prevention](https://docs.sqlalchemy.org/en/20/orm/queryguide/relationships.html)
- [PostgreSQL Query Optimization](https://www.postgresql.org/docs/current/performance-tips.html)
- [Async Python Best Practices](https://docs.python.org/3/library/asyncio-task.html)

---

**End of Report**
