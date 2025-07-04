#!/usr/bin/env python3
"""
Script to check completeness between Supabase and TurboPuffer for document migration.
Compares multivector embeddings table in Supabase with TurboPuffer namespace data.
"""

import asyncio

# Add the project root to the path so we can import from core
import os
import sys
from collections import defaultdict
from typing import Dict

import psycopg
from turbopuffer import AsyncTurbopuffer

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)
from core.config import get_settings  # noqa: E402


async def get_supabase_documents(supabase_uri: str) -> Dict[str, Dict]:
    """
    Get document and chunk information from Supabase.

    Returns:
        Dict mapping document_id to {
            'chunks': set of chunk_numbers,
            'app_id': app_id from documents table,
            'total_chunks': count of chunks
        }
    """
    documents = defaultdict(lambda: {"chunks": set(), "app_id": None, "total_chunks": 0})

    # Convert SQLAlchemy URI to psycopg format
    if supabase_uri.startswith("postgresql+asyncpg://"):
        supabase_uri = supabase_uri.replace("postgresql+asyncpg://", "postgresql://")

    with psycopg.connect(supabase_uri) as conn:
        # Get all chunks from multi_vector_embeddings
        query = """
            SELECT document_id, chunk_number
            FROM multi_vector_embeddings
            ORDER BY document_id, chunk_number
        """

        with conn.cursor() as cur:
            cur.execute(query)
            for row in cur:
                document_id, chunk_number = row
                documents[document_id]["chunks"].add(chunk_number)
                documents[document_id]["total_chunks"] += 1

        # Also get total document count from documents table for comparison
        query = """
            SELECT COUNT(DISTINCT external_id) as total_docs_in_documents_table
            FROM documents
        """

        with conn.cursor() as cur:
            cur.execute(query)
            total_docs_in_documents_table = cur.fetchone()[0]
            print(f"Total documents in documents table: {total_docs_in_documents_table}")
            print(f"Documents with embeddings: {len(documents)}")
            if total_docs_in_documents_table != len(documents):
                print(f"WARNING: {total_docs_in_documents_table - len(documents)} documents have no embeddings!")

        # Get app_ids from documents table
        query = """
            SELECT external_id, system_metadata->>'app_id' as app_id
            FROM documents
            WHERE external_id = ANY(%s)
        """

        doc_ids = list(documents.keys())
        if doc_ids:
            with conn.cursor() as cur:
                cur.execute(query, (doc_ids,))
                for row in cur:
                    external_id, app_id = row
                    if external_id in documents:
                        documents[external_id]["app_id"] = app_id or "default"

    return dict(documents)


async def get_turbopuffer_documents(api_key: str, namespace: str, region: str = "aws-us-west-2") -> Dict[str, Dict]:
    """
    Get document and chunk information from TurboPuffer using pagination.

    Returns:
        Dict mapping document_id to {
            'chunks': set of chunk_numbers,
            'total_chunks': count of chunks
        }
    """
    documents = defaultdict(lambda: {"chunks": set(), "total_chunks": 0})

    tpuf = AsyncTurbopuffer(api_key=api_key, region=region)
    ns = tpuf.namespace(namespace)

    # Use pagination to get all documents
    last_id = None
    total_records = 0

    while True:
        # Query with pagination
        filters = ("id", "Gt", last_id) if last_id is not None else None

        result = await ns.query(
            rank_by=("id", "asc"),
            top_k=1000,  # Maximum allowed per query
            filters=filters,
            include_attributes=["document_id", "chunk_number"],
        )

        # Process results
        for row in result.rows:
            document_id = row["document_id"]
            chunk_number = row["chunk_number"]
            documents[document_id]["chunks"].add(chunk_number)
            documents[document_id]["total_chunks"] += 1

        total_records += len(result.rows)
        print(f"Fetched {total_records} records from TurboPuffer...")

        # Check if we've reached the end
        if len(result.rows) < 1000:
            break

        # Update last_id for next iteration
        last_id = result.rows[-1].id

    return dict(documents)


def analyze_completeness(supabase_docs: Dict, turbopuffer_docs: Dict) -> Dict:
    """
    Analyze completeness between Supabase and TurboPuffer data.

    Returns:
        Dict with analysis results
    """
    all_doc_ids = set(supabase_docs.keys()) | set(turbopuffer_docs.keys())

    missing_from_turbopuffer = []
    missing_from_supabase = []
    incomplete_migrations = []
    complete_migrations = []

    # Get all unique app_ids from Supabase
    app_ids = set()
    for doc_data in supabase_docs.values():
        if doc_data["app_id"]:
            app_ids.add(doc_data["app_id"])

    # Analyze each document
    for doc_id in all_doc_ids:
        supabase_data = supabase_docs.get(doc_id)
        turbopuffer_data = turbopuffer_docs.get(doc_id)

        if supabase_data and not turbopuffer_data:
            # Document exists in Supabase but not in TurboPuffer
            missing_from_turbopuffer.append(
                {
                    "document_id": doc_id,
                    "app_id": supabase_data["app_id"],
                    "total_chunks": supabase_data["total_chunks"],
                    "chunks": sorted(list(supabase_data["chunks"])),
                }
            )
        elif turbopuffer_data and not supabase_data:
            # Document exists in TurboPuffer but not in Supabase (shouldn't happen in migration)
            missing_from_supabase.append(
                {
                    "document_id": doc_id,
                    "total_chunks": turbopuffer_data["total_chunks"],
                    "chunks": sorted(list(turbopuffer_data["chunks"])),
                }
            )
        elif supabase_data and turbopuffer_data:
            # Document exists in both - check chunk completeness
            supabase_chunks = supabase_data["chunks"]
            turbopuffer_chunks = turbopuffer_data["chunks"]

            missing_chunks = supabase_chunks - turbopuffer_chunks
            extra_chunks = turbopuffer_chunks - supabase_chunks

            if missing_chunks or extra_chunks:
                incomplete_migrations.append(
                    {
                        "document_id": doc_id,
                        "app_id": supabase_data["app_id"],
                        "supabase_chunks": supabase_data["total_chunks"],
                        "turbopuffer_chunks": turbopuffer_data["total_chunks"],
                        "missing_chunks": sorted(list(missing_chunks)),
                        "extra_chunks": sorted(list(extra_chunks)),
                    }
                )
            else:
                complete_migrations.append(
                    {
                        "document_id": doc_id,
                        "app_id": supabase_data["app_id"],
                        "total_chunks": supabase_data["total_chunks"],
                    }
                )

    # Group missing documents by app_id
    missing_by_app_id = defaultdict(list)
    for doc in missing_from_turbopuffer:
        app_id = doc["app_id"] or "default"
        missing_by_app_id[app_id].append(doc)

    return {
        "app_ids": sorted(list(app_ids)),
        "missing_from_turbopuffer": missing_from_turbopuffer,
        "missing_from_supabase": missing_from_supabase,
        "incomplete_migrations": incomplete_migrations,
        "complete_migrations": complete_migrations,
        "missing_by_app_id": dict(missing_by_app_id),
        "summary": {
            "total_documents_supabase": len(supabase_docs),
            "total_documents_turbopuffer": len(turbopuffer_docs),
            "documents_missing_from_turbopuffer": len(missing_from_turbopuffer),
            "documents_missing_from_supabase": len(missing_from_supabase),
            "incomplete_migrations": len(incomplete_migrations),
            "complete_migrations": len(complete_migrations),
            "total_chunks_supabase": sum(doc["total_chunks"] for doc in supabase_docs.values()),
            "total_chunks_turbopuffer": sum(doc["total_chunks"] for doc in turbopuffer_docs.values()),
        },
    }


def generate_report(analysis: Dict) -> str:
    """Generate a text report from the analysis results."""
    report = []

    report.append("=== MIGRATION COMPLETENESS REPORT ===\n")

    # Summary
    summary = analysis["summary"]
    report.append("SUMMARY:")
    report.append(f"  Total documents in Supabase: {summary['total_documents_supabase']}")
    report.append(f"  Total documents in TurboPuffer: {summary['total_documents_turbopuffer']}")
    report.append(f"  Total chunks in Supabase: {summary['total_chunks_supabase']}")
    report.append(f"  Total chunks in TurboPuffer: {summary['total_chunks_turbopuffer']}")
    report.append(f"  Documents missing from TurboPuffer: {summary['documents_missing_from_turbopuffer']}")
    report.append(f"  Documents missing from Supabase: {summary['documents_missing_from_supabase']}")
    report.append(f"  Incomplete migrations: {summary['incomplete_migrations']}")
    report.append(f"  Complete migrations: {summary['complete_migrations']}")
    report.append("")

    # App IDs found
    report.append("APP IDs FOUND:")
    for app_id in analysis["app_ids"]:
        report.append(f"  - {app_id}")
    report.append("")

    # Documents missing from TurboPuffer by app_id
    report.append("DOCUMENTS MISSING FROM TURBOPUFFER BY APP_ID:")
    if analysis["missing_by_app_id"]:
        for app_id, docs in analysis["missing_by_app_id"].items():
            report.append(f"  App ID: {app_id} ({len(docs)} documents)")
            for doc in docs:
                report.append(f"    - {doc['document_id']} ({doc['total_chunks']} chunks)")
    else:
        report.append("  None")
    report.append("")

    # All missing document IDs
    report.append("ALL DOCUMENT IDs MISSING FROM TURBOPUFFER:")
    if analysis["missing_from_turbopuffer"]:
        for doc in analysis["missing_from_turbopuffer"]:
            report.append(f"  {doc['document_id']}")
    else:
        report.append("  None")
    report.append("")

    # Incomplete migrations
    report.append("INCOMPLETE MIGRATIONS (partial chunk coverage):")
    if analysis["incomplete_migrations"]:
        for doc in analysis["incomplete_migrations"]:
            report.append(f"  {doc['document_id']} (App: {doc['app_id']})")
            report.append(
                f"    Supabase chunks: {doc['supabase_chunks']}, TurboPuffer chunks: {doc['turbopuffer_chunks']}"
            )
            if doc["missing_chunks"]:
                report.append(f"    Missing chunks: {doc['missing_chunks']}")
            if doc["extra_chunks"]:
                report.append(f"    Extra chunks: {doc['extra_chunks']}")
    else:
        report.append("  None")
    report.append("")

    # Documents missing from Supabase (shouldn't happen in migration)
    if analysis["missing_from_supabase"]:
        report.append("DOCUMENTS MISSING FROM SUPABASE (unexpected):")
        for doc in analysis["missing_from_supabase"]:
            report.append(f"  {doc['document_id']} ({doc['total_chunks']} chunks)")
        report.append("")

    return "\n".join(report)


async def main():
    """Main function to run the completeness check."""
    # Load settings from config
    try:
        settings = get_settings()
    except Exception as e:
        print(f"Error loading settings: {e}")
        sys.exit(1)

    # Check if required settings are available
    if not settings.POSTGRES_URI:
        print("Error: POSTGRES_URI not found in settings")
        sys.exit(1)

    if not settings.TURBOPUFFER_API_KEY:
        print("Error: TURBOPUFFER_API_KEY not found in settings")
        sys.exit(1)

    # Allow namespace to be overridden via command line argument
    turbopuffer_namespace = "public"  # Default namespace used in ingestion worker
    if len(sys.argv) > 1:
        turbopuffer_namespace = sys.argv[1]
        print(f"Using namespace: {turbopuffer_namespace}")
    else:
        print(f"Using default namespace: {turbopuffer_namespace}")
        print("(You can specify a different namespace as: python check_completeness.py <namespace>)")

    print(f"Using Supabase URI: {settings.POSTGRES_URI}")
    print(f"Using TurboPuffer API key: {settings.TURBOPUFFER_API_KEY[:10]}...")

    print("Fetching data from Supabase...")
    try:
        supabase_docs = await get_supabase_documents(settings.POSTGRES_URI)
        print(f"Found {len(supabase_docs)} documents in Supabase")
    except Exception as e:
        print(f"Error connecting to Supabase: {e}")
        sys.exit(1)

    print("Fetching data from TurboPuffer...")
    try:
        turbopuffer_docs = await get_turbopuffer_documents(settings.TURBOPUFFER_API_KEY, turbopuffer_namespace)
        print(f"Found {len(turbopuffer_docs)} documents in TurboPuffer")
    except Exception as e:
        print(f"Error connecting to TurboPuffer: {e}")
        sys.exit(1)

    print("Analyzing completeness...")
    analysis = analyze_completeness(supabase_docs, turbopuffer_docs)

    print("Generating report...")
    report = generate_report(analysis)

    # Write report to file
    report_filename = "migration_completeness_report.txt"
    with open(report_filename, "w") as f:
        f.write(report)

    print(f"Report saved to {report_filename}")
    print("\n" + "=" * 50)
    print(report)


if __name__ == "__main__":
    asyncio.run(main())
