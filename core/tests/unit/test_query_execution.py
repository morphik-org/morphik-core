import pytest
import uuid
import json
from datetime import datetime, UTC

from core.smart_query_engine.query_execution import QueryExecutor
from core.models.request import SmartQueryRequest, FilterOperation, SortOperation
from core.models.documents import Document

# Custom JSON encoder to handle datetime objects
class DateTimeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)


def create_test_document(
    external_id=None,
    filename=None,
    content_type="text/plain",
    content="",
    metadata=None,
    additional_metadata=None,
    created_at=None,
    system_metadata=None,
):
    """Helper function to create test documents with different properties"""
    if external_id is None:
        external_id = str(uuid.uuid4())
    
    if metadata is None:
        metadata = {}
    
    if additional_metadata is None:
        additional_metadata = {}
    
    if created_at is None:
        created_at = datetime.now(UTC)
        # Convert datetime to string to avoid JSON serialization issues
        created_at_str = created_at.isoformat()
    else:
        created_at_str = created_at.isoformat() if isinstance(created_at, datetime) else created_at
    
    if system_metadata is None:
        system_metadata = {
            "created_at": created_at_str,
            "updated_at": created_at_str,
            "content": content,
            "version": 1,
            "folder_name": None,
            "end_user_id": None,
        }
    
    # Ensure datetimes are serialized
    if "created_at" in system_metadata and isinstance(system_metadata["created_at"], datetime):
        system_metadata["created_at"] = system_metadata["created_at"].isoformat()
    if "updated_at" in system_metadata and isinstance(system_metadata["updated_at"], datetime):
        system_metadata["updated_at"] = system_metadata["updated_at"].isoformat()
    
    return Document(
        external_id=external_id,
        filename=filename or f"doc_{external_id}.txt",
        owner={"id": "test_user", "name": "Test User"},
        content_type=content_type,
        metadata=metadata,
        additional_metadata=additional_metadata,
        system_metadata=system_metadata,
    )


@pytest.fixture
def test_documents():
    """Create a set of documents with varied metadata and content for testing"""
    # Document with explicit date in metadata
    doc1 = create_test_document(
        external_id="doc1",
        filename="report_2024_01_15.txt",
        content="This is a quarterly financial report for Q1 2024. Revenue: $10M, Profit: $2M.",
        metadata={
            "date": "2024-01-15",
            "author": "Alice Smith",
            "department": "Finance",
            "type": "quarterly_report",
            "quarter": "Q1",
            "year": 2024,
        },
        additional_metadata={
            "confidentiality": "internal",
            "reviewed": True,
        }
    )
    
    # Document with date hidden in filename
    doc2 = create_test_document(
        external_id="doc2",
        filename="sales_2024_02_20.txt",
        content="Monthly sales report for February 2024. Total sales: $5.2M across all regions.",
        metadata={
            "author": "Bob Johnson",
            "department": "Sales",
            "type": "monthly_report",
            "regions": ["North America", "Europe"],
        },
        additional_metadata={
            "confidentiality": "internal",
            "reviewed": False,
        }
    )
    
    # Document with date in content but not metadata
    doc3 = create_test_document(
        external_id="doc3",
        filename="project_roadmap.txt",
        content="Project roadmap created on March 10, 2024. Key milestones: Alpha release in April, Beta in June, Release in September.",
        metadata={
            "author": "Charlie Davis",
            "department": "Product",
            "type": "roadmap",
            "project": "Phoenix",
        }
    )
    
    # Document with nested metadata and structured data
    doc4 = create_test_document(
        external_id="doc4",
        filename="customer_survey.json",
        content_type="application/json",
        content=json.dumps({
            "survey_results": {
                "satisfaction": 4.2,
                "nps": 65,
                "respondents": 230,
                "date": "2024-03-25"
            }
        }),
        metadata={
            "author": "Diana Lee",
            "department": "Marketing",
            "type": "survey_results",
            "survey_id": "CS2024-Q1",
        },
        additional_metadata={
            "survey_metadata": {
                "distribution_channel": "email",
                "response_rate": 0.42,
                "completion_rate": 0.89
            }
        }
    )
    
    # Document with numerical data in both metadata and content
    doc5 = create_test_document(
        external_id="doc5",
        filename="inventory_status.txt",
        content="Warehouse inventory as of April 5, 2024: 2500 units of Product A, 1200 units of Product B, 850 units of Product C.",
        metadata={
            "author": "Edward Wilson",
            "department": "Operations",
            "type": "inventory",
            "total_items": 4550,
            "warehouse_id": "WH-005",
        }
    )
    
    return [doc1, doc2, doc3, doc4, doc5]


@pytest.fixture
def query_executor():
    """Create a QueryExecutor instance for testing"""
    return QueryExecutor()


@pytest.mark.asyncio
async def test_simple_metadata_filtering(query_executor, test_documents):
    """Test filtering documents by metadata only criteria"""
    
    # Create filter request for finance department
    request = SmartQueryRequest(
        filter=FilterOperation(predicate="department is Finance")
    )
    
    # Execute query
    result = await query_executor.execute_query(request, test_documents)
    
    # Verify only doc1 is returned (Finance department)
    assert len(result) == 1
    assert "doc1" in result


@pytest.mark.asyncio
async def test_content_based_filtering(query_executor, test_documents):
    """Test filtering documents based on content"""
    
    # Create filter request for documents containing roadmap
    request = SmartQueryRequest(
        filter=FilterOperation(predicate="document contains project roadmap")
    )
    
    # Execute query
    result = await query_executor.execute_query(request, test_documents)
    
    # Verify doc3 is in the results (contains "project roadmap")
    assert "doc3" in result


@pytest.mark.asyncio
async def test_filtering_with_multiple_conditions(query_executor, test_documents):
    """Test filtering with complex conditions spanning metadata and content"""
    
    # Filter for quarterly reports with revenue over $5M
    request = SmartQueryRequest(
        filter=FilterOperation(predicate="document is a quarterly report with revenue over $5 million")
    )
    
    # Execute query
    result = await query_executor.execute_query(request, test_documents)
    
    # Verify doc1 is in the results (quarterly report with $10M revenue)
    assert "doc1" in result


@pytest.mark.asyncio
async def test_date_filtering_hidden_in_different_locations(query_executor, test_documents):
    """Test filtering documents by date that could be in metadata, filename, or content"""
    
    # Create filter request for Q1 documents
    request = SmartQueryRequest(
        filter=FilterOperation(predicate="documents from January or February 2024")
    )
    
    # Execute query
    result = await query_executor.execute_query(request, test_documents)
    
    # Verify doc1 and doc2 are returned (January and February docs)
    assert len(result) >= 1
    assert "doc1" in result  # Has January 2024 in metadata
    # doc2 should be included if the LLM correctly identifies the date in filename/content


@pytest.mark.asyncio
async def test_simple_sorting(query_executor, test_documents):
    """Test sorting documents by author name"""
    
    # Create sort request by author name
    request = SmartQueryRequest(
        sort_by=[SortOperation(comparator="sort by author name alphabetically", order="ASC")]
    )
    
    # Execute query
    result = await query_executor.execute_query(request, test_documents)
    
    # Verify first document is doc1 (Alice Smith comes first alphabetically)
    assert result[0] == "doc1"


@pytest.mark.asyncio
async def test_reverse_sorting(query_executor, test_documents):
    """Test sorting documents in descending order by author name"""
    
    # Create sort request by author in descending order
    request = SmartQueryRequest(
        sort_by=[SortOperation(comparator="sort by author name alphabetically", order="DESC")]
    )
    
    # Execute query
    result = await query_executor.execute_query(request, test_documents)
    
    # Verify first document is doc5 (Edward Wilson comes last alphabetically)
    assert result[0] == "doc5"


@pytest.mark.asyncio
async def test_numerical_sorting(query_executor, test_documents):
    """Test sorting documents by numerical values in content or metadata"""
    
    # Create sort request to sort by total number of items
    request = SmartQueryRequest(
        sort_by=[SortOperation(comparator="sort by total number of items or total value", order="ASC")]
    )
    
    # Execute query
    result = await query_executor.execute_query(request, test_documents)
    
    # Verify doc5 with 4550 items is last when sorting ascending
    assert result[-1] == "doc5"


@pytest.mark.asyncio
async def test_department_based_sorting(query_executor, test_documents):
    """Test sorting documents by department name"""
    
    # Create sort request by department
    request = SmartQueryRequest(
        sort_by=[SortOperation(comparator="sort by department name alphabetically", order="ASC")]
    )
    
    # Execute query
    result = await query_executor.execute_query(request, test_documents)
    
    # Verify order has Finance (doc1) first alphabetically
    assert result[0] == "doc1"


@pytest.mark.asyncio
async def test_date_based_sorting(query_executor, test_documents):
    """Test sorting documents by date (from metadata, filename or content)"""
    
    # Create sort request by date, newest first
    request = SmartQueryRequest(
        sort_by=[SortOperation(comparator="sort by document date", order="DESC")]
    )
    
    # Execute query
    result = await query_executor.execute_query(request, test_documents)
    
    # Verify doc5 (April 5) is earlier in the list than doc1 (January 15)
    doc5_index = result.index("doc5") if "doc5" in result else -1
    doc1_index = result.index("doc1") if "doc1" in result else -1
    
    assert doc5_index != -1 and doc1_index != -1
    assert doc5_index < doc1_index  # DESC order, so newer dates come first


@pytest.mark.asyncio
async def test_filter_and_sort_together(query_executor, test_documents):
    """Test filtering and sorting together"""
    
    # Create request to filter for reports and sort by date
    request = SmartQueryRequest(
        filter=FilterOperation(predicate="document type contains 'report'"),
        sort_by=[SortOperation(comparator="sort by date", order="ASC")]
    )
    
    # Execute query
    result = await query_executor.execute_query(request, test_documents)
    
    # Verify doc1 and doc2 are in results and doc1 (Jan) comes before doc2 (Feb)
    assert "doc1" in result and "doc2" in result
    assert result.index("doc1") < result.index("doc2")


@pytest.mark.asyncio
async def test_limit_results(query_executor, test_documents):
    """Test limiting the number of results"""
    
    # Create request to sort by department and limit to 2 results
    request = SmartQueryRequest(
        sort_by=[SortOperation(comparator="sort by department alphabetically", order="ASC")],
        limit=2
    )
    
    # Execute query
    result = await query_executor.execute_query(request, test_documents)
    
    # Verify only 2 results are returned
    assert len(result) == 2
    
    # Finance (doc1) should be first when sorting alphabetically by department
    assert result[0] == "doc1"


@pytest.mark.asyncio
async def test_no_filter_no_sort(query_executor, test_documents):
    """Test query with no filter and no sort criteria"""
    
    # Create empty request
    request = SmartQueryRequest()
    
    # Execute query
    result = await query_executor.execute_query(request, test_documents)
    
    # Verify all documents are returned
    assert len(result) == 5
    assert set(result) == {"doc1", "doc2", "doc3", "doc4", "doc5"}


@pytest.mark.asyncio
async def test_empty_documents_list(query_executor):
    """Test query with empty documents list"""
    
    # Create request with filter and sort
    request = SmartQueryRequest(
        filter=FilterOperation(predicate="any filter"),
        sort_by=[SortOperation(comparator="any sort", order="ASC")]
    )
    
    # Execute query with empty list
    result = await query_executor.execute_query(request, [])
    
    # Verify empty list is returned
    assert result == []


@pytest.mark.asyncio
async def test_complex_nested_metadata_filtering(query_executor):
    """Test filtering based on complex nested metadata structures"""
    
    # Create document with deeply nested metadata
    doc_nested = create_test_document(
        external_id="doc_nested",
        filename="complex_data.json",
        content_type="application/json",
        content=json.dumps({"data": "This is complex structured data"}),
        metadata={
            "author": "Nested Data Specialist",
            "department": "Data Science",
            "type": "complex_data",
        },
        additional_metadata={
            "stats": {
                "depth": 3,
                "nodes": 42,
                "analysis": {
                    "quality": "high",
                    "confidence": 0.95,
                    "methods": ["clustering", "classification", "regression"]
                }
            }
        }
    )
    
    # Regular document without the nested quality field
    doc_regular = create_test_document(
        external_id="doc_regular",
        filename="simple_data.txt",
        content="This is simple data",
        metadata={
            "author": "Regular Author",
            "department": "IT",
            "type": "simple_data",
        }
    )
    
    test_docs = [doc_nested, doc_regular]
    
    # Filter for documents with quality=high in the nested metadata
    request = SmartQueryRequest(
        filter=FilterOperation(predicate="document has high quality in analysis metadata")
    )
    
    # Execute query
    result = await query_executor.execute_query(request, test_docs)
    
    # Verify only the nested document is returned
    assert result == ["doc_nested"]


@pytest.mark.asyncio
async def test_multiple_sort_criteria(query_executor):
    """Test sorting documents with multiple criteria"""
    
    # Create documents in same department with different types
    doc_hr1 = create_test_document(
        external_id="doc_hr1",
        filename="hr_report_2024.txt",
        content="HR annual report. Headcount: 120",
        metadata={
            "author": "Frank Miller",
            "department": "HR",
            "type": "annual_report",
            "year": 2024,
        }
    )
    
    doc_hr2 = create_test_document(
        external_id="doc_hr2",
        filename="hr_budget_2024.txt",
        content="HR budget plan for 2024. Total budget: $1.2M",
        metadata={
            "author": "Grace Wang",
            "department": "HR",
            "type": "budget",
            "year": 2024,
        }
    )
    
    test_docs = [doc_hr1, doc_hr2]
    
    # Create request with multiple sort criteria
    request = SmartQueryRequest(
        sort_by=[
            SortOperation(comparator="sort by department", order="ASC"),
            SortOperation(comparator="sort by document type", order="ASC")
        ]
    )
    
    # Execute query
    result = await query_executor.execute_query(request, test_docs)
    
    # Both are in HR department, but "annual_report" comes before "budget" alphabetically
    assert len(result) == 2
    assert result[0] == "doc_hr1"
    assert result[1] == "doc_hr2"
