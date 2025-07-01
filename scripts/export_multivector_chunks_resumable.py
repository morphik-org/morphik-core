#!/usr/bin/env python3
"""
Export multi_vector_embeddings table from Supabase to local files with resume capability.
This script exports the data in multiple formats for maximum flexibility:
- SQL dump file (.sql)
- SQLite database file (.db)
- CSV file (.csv)

Features:
- Batch processing for large datasets
- Checkpoint/resume functionality
- Progress tracking
- Error recovery
"""

import argparse
import asyncio
import csv
import json
import pickle
import sqlite3
import sys
from datetime import datetime
from decimal import Decimal
from pathlib import Path

import asyncpg

BATCH_SIZE = 1000  # Process 1000 rows at a time
CHECKPOINT_FREQUENCY = 10  # Save checkpoint every 10 batches


class CustomJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle PostgreSQL-specific types."""

    def default(self, obj):
        # Handle BitString objects
        if hasattr(obj, "__class__") and "BitString" in str(obj.__class__):
            try:
                if hasattr(obj, "tobytes"):
                    return obj.tobytes().hex()
                else:
                    return str(obj)
            except Exception:
                return str(obj)

        # Handle Decimal objects
        if isinstance(obj, Decimal):
            return float(obj)

        # Handle bytes
        if isinstance(obj, bytes):
            return obj.hex()

        # Handle datetime
        if isinstance(obj, datetime):
            return obj.isoformat()

        # Default behavior
        return super().default(obj)


def safe_json_dumps(obj):
    """Safely serialize objects to JSON with custom encoder."""
    try:
        return json.dumps(obj, cls=CustomJSONEncoder)
    except Exception:
        # If all else fails, convert to string
        return json.dumps(str(obj))


class ExportCheckpoint:
    """Class to manage export checkpoints for resume functionality."""

    def __init__(self, checkpoint_file):
        self.checkpoint_file = checkpoint_file
        self.data = {
            "timestamp": None,
            "total_rows": 0,
            "processed_rows": 0,
            "last_offset": 0,
            "batch_count": 0,
            "schema": None,
            "files": {"sql": None, "sqlite": None, "csv": None, "metadata": None},
            "completed": False,
        }
        self.load_checkpoint()

    def load_checkpoint(self):
        """Load checkpoint from file if it exists."""
        if Path(self.checkpoint_file).exists():
            try:
                with open(self.checkpoint_file, "rb") as f:
                    self.data = pickle.load(f)
                print(f"📂 Loaded checkpoint: {self.processed_rows}/{self.total_rows} rows completed")
                return True
            except Exception as e:
                print(f"⚠️  Could not load checkpoint: {e}")
                return False
        return False

    def save_checkpoint(self):
        """Save current progress to checkpoint file."""
        try:
            with open(self.checkpoint_file, "wb") as f:
                pickle.dump(self.data, f)
        except Exception as e:
            print(f"⚠️  Could not save checkpoint: {e}")

    def update_progress(self, processed_rows, last_offset, batch_count):
        """Update progress information."""
        self.data["processed_rows"] = processed_rows
        self.data["last_offset"] = last_offset
        self.data["batch_count"] = batch_count

    def mark_completed(self):
        """Mark export as completed."""
        self.data["completed"] = True
        self.save_checkpoint()

    def cleanup(self):
        """Remove checkpoint file after successful completion."""
        if Path(self.checkpoint_file).exists():
            Path(self.checkpoint_file).unlink()
            print("🧹 Cleaned up checkpoint file")

    @property
    def total_rows(self):
        return self.data["total_rows"]

    @property
    def processed_rows(self):
        return self.data["processed_rows"]

    @property
    def last_offset(self):
        return self.data["last_offset"]

    @property
    def batch_count(self):
        return self.data["batch_count"]

    @property
    def schema(self):
        return self.data["schema"]

    @property
    def files(self):
        return self.data["files"]

    @property
    def timestamp(self):
        return self.data["timestamp"]

    @property
    def is_completed(self):
        return self.data["completed"]


async def export_multi_vector_embeddings(resume=False, force_restart=False):
    """Export multi_vector_embeddings table from Supabase in batches with resume capability."""

    # Database connection string
    DATABASE_URL = (
        "postgresql+asyncpg://postgres:xafzaq-8cogni-Dadsyw@db.sjziybcvpdghcaqkssnm.supabase.co:5432/postgres"
    )

    # Clean the URL for asyncpg (remove the +asyncpg part)
    ASYNCPG_URL = DATABASE_URL.replace("postgresql+asyncpg://", "postgresql://")

    # Initialize checkpoint
    checkpoint_file = "export_checkpoint.pkl"
    checkpoint = ExportCheckpoint(checkpoint_file)

    # Handle force restart
    if force_restart and Path(checkpoint_file).exists():
        Path(checkpoint_file).unlink()
        checkpoint = ExportCheckpoint(checkpoint_file)
        print("🔄 Force restart: Removed existing checkpoint")

    # Check if already completed
    if checkpoint.is_completed:
        print("✅ Export already completed! Use --force-restart to start over.")
        return

    print("🔗 Connecting to Supabase database...")

    try:
        # Connect to the database
        conn = await asyncpg.connect(ASYNCPG_URL)

        # Initialize or resume export
        if checkpoint.processed_rows == 0:
            # Fresh start
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            await initialize_export(conn, checkpoint, timestamp)
        else:
            # Resume from checkpoint
            print(f"🔄 Resuming export from {checkpoint.processed_rows}/{checkpoint.total_rows} rows")
            timestamp = checkpoint.timestamp

        # Open files for writing (append mode for resume)
        sql_file, sqlite_conn, csv_file, csv_writer = open_export_files(
            checkpoint, resume=(checkpoint.processed_rows > 0)
        )

        # Process remaining data in batches
        column_names = [col["column_name"] for col in checkpoint.schema]

        print(
            f"📥 Processing remaining {checkpoint.total_rows - checkpoint.processed_rows} rows in batches of {BATCH_SIZE}..."
        )

        batch_count = checkpoint.batch_count

        for offset in range(checkpoint.last_offset, checkpoint.total_rows, BATCH_SIZE):
            batch_query = f"""
            SELECT * FROM multi_vector_embeddings
            ORDER BY id
            LIMIT {BATCH_SIZE} OFFSET {offset};
            """

            try:
                batch_rows = await conn.fetch(batch_query)

                if not batch_rows:
                    break

                # Write batch to all formats
                write_batch_to_sql(sql_file, batch_rows, column_names)
                write_batch_to_sqlite(sqlite_conn, batch_rows, column_names)
                write_batch_to_csv(csv_writer, batch_rows, column_names)

                # Update progress
                processed_rows = checkpoint.processed_rows + len(batch_rows)
                batch_count += 1

                checkpoint.update_progress(processed_rows, offset + BATCH_SIZE, batch_count)

                progress = (processed_rows / checkpoint.total_rows) * 100
                print(f"📊 Progress: {processed_rows}/{checkpoint.total_rows} ({progress:.1f}%) - Batch {batch_count}")

                # Save checkpoint periodically
                if batch_count % CHECKPOINT_FREQUENCY == 0:
                    checkpoint.save_checkpoint()
                    print(f"💾 Checkpoint saved at batch {batch_count}")

            except Exception as e:
                print(f"❌ Error processing batch at offset {offset}: {e}")
                checkpoint.save_checkpoint()
                print("💾 Progress saved. You can resume with --resume flag")
                raise

        # Finalize files
        finalize_sql_dump(sql_file)
        finalize_sqlite_db(sqlite_conn, checkpoint.files["sqlite"])
        finalize_csv_file(csv_file)

        # Export metadata (update if resuming)
        export_metadata(checkpoint.schema, checkpoint.total_rows, timestamp)

        # Mark as completed
        checkpoint.mark_completed()

        await conn.close()
        print("🎉 Export completed successfully!")

        # Show file sizes
        show_file_info(checkpoint.files["sql"], checkpoint.files["sqlite"], checkpoint.files["csv"])

        # Cleanup checkpoint file
        checkpoint.cleanup()

    except Exception as e:
        print(f"❌ Error: {e}")
        print("💾 Progress has been saved. Resume with: python export_multivector_chunks_resumable.py --resume")
        sys.exit(1)


async def initialize_export(conn, checkpoint, timestamp):
    """Initialize a fresh export."""
    print("🆕 Starting fresh export...")

    # Get table schema
    print("📋 Getting table schema...")
    schema_query = """
    SELECT column_name, data_type, is_nullable, column_default
    FROM information_schema.columns
    WHERE table_name = 'multi_vector_embeddings'
    ORDER BY ordinal_position;
    """

    schema_rows = await conn.fetch(schema_query)
    if not schema_rows:
        raise Exception("Table 'multi_vector_embeddings' not found!")

    print(f"📊 Found table with {len(schema_rows)} columns:")
    for row in schema_rows:
        print(f"  - {row['column_name']}: {row['data_type']}")

    # Get total row count
    print("🔢 Getting total row count...")
    count_query = "SELECT COUNT(*) as count FROM multi_vector_embeddings;"
    count_result = await conn.fetchrow(count_query)
    total_rows = count_result["count"]
    print(f"✅ Found {total_rows} rows to export")

    if total_rows == 0:
        raise Exception("No data found in the table")

    # Update checkpoint with initial data
    checkpoint.data["timestamp"] = timestamp
    checkpoint.data["total_rows"] = total_rows
    checkpoint.data["schema"] = [dict(row) for row in schema_rows]
    checkpoint.data["files"] = {
        "sql": f"multi_vector_embeddings_dump_{timestamp}.sql",
        "sqlite": f"multi_vector_embeddings_{timestamp}.db",
        "csv": f"multi_vector_embeddings_{timestamp}.csv",
        "metadata": f"multi_vector_embeddings_metadata_{timestamp}.json",
    }

    checkpoint.save_checkpoint()


def open_export_files(checkpoint, resume=False):
    """Open export files for writing."""
    files = checkpoint.files

    if resume:
        print("📂 Opening files in append mode for resume...")
        # SQL file - append mode
        sql_file = open(files["sql"], "a", encoding="utf-8")

        # SQLite - just connect to existing file
        sqlite_conn = sqlite3.connect(files["sqlite"])

        # CSV file - append mode
        csv_file = open(files["csv"], "a", newline="", encoding="utf-8")
        csv_writer = csv.writer(csv_file)

    else:
        print("📂 Creating new export files...")
        # Initialize SQLite database
        sqlite_conn = init_sqlite_db(files["sqlite"], checkpoint.schema)

        # Initialize SQL dump file
        sql_file = init_sql_dump(files["sql"], checkpoint.schema, checkpoint.total_rows)

        # Initialize CSV file
        csv_file, csv_writer = init_csv_file(files["csv"], checkpoint.schema)

    return sql_file, sqlite_conn, csv_file, csv_writer


def init_sqlite_db(filename, schema_rows):
    """Initialize SQLite database file."""
    print(f"💾 Initializing SQLite database: {filename}")

    # Remove file if it exists
    if Path(filename).exists():
        Path(filename).unlink()

    conn = sqlite3.connect(filename)
    cursor = conn.cursor()

    # Create table
    column_defs = []
    for schema_row in schema_rows:
        col_name = schema_row["column_name"]
        data_type = schema_row["data_type"]

        # Map PostgreSQL types to SQLite types
        if data_type in ["uuid", "character varying", "text", "jsonb"]:
            sqlite_type = "TEXT"
        elif data_type in ["integer", "bigint"]:
            sqlite_type = "INTEGER"
        elif data_type in ["double precision", "real"]:
            sqlite_type = "REAL"
        elif data_type == "bytea":
            sqlite_type = "BLOB"
        elif data_type == "timestamp with time zone":
            sqlite_type = "TEXT"
        elif data_type == "ARRAY":
            sqlite_type = "TEXT"  # Store arrays as JSON strings
        else:
            sqlite_type = "TEXT"

        column_defs.append(f"{col_name} {sqlite_type}")

    create_sql = f"CREATE TABLE multi_vector_embeddings ({', '.join(column_defs)})"
    cursor.execute(create_sql)

    return conn


def init_sql_dump(filename, schema_rows, total_rows):
    """Initialize SQL dump file."""
    print(f"💾 Initializing SQL dump: {filename}")

    sql_file = open(filename, "w", encoding="utf-8")

    # Write header
    sql_file.write("-- Multi Vector Embeddings Table Export\n")
    sql_file.write(f"-- Exported on: {datetime.now().isoformat()}\n")
    sql_file.write(f"-- Total rows: {total_rows}\n\n")

    # Create table statement
    sql_file.write("-- Create table statement\n")
    sql_file.write("CREATE TABLE IF NOT EXISTS multi_vector_embeddings (\n")

    column_defs = []
    for schema_row in schema_rows:
        col_name = schema_row["column_name"]
        data_type = schema_row["data_type"]
        is_nullable = schema_row["is_nullable"]

        # Map PostgreSQL types to more generic SQL types
        if data_type == "uuid":
            sql_type = "TEXT"
        elif data_type == "timestamp with time zone":
            sql_type = "TIMESTAMP"
        elif data_type == "jsonb":
            sql_type = "TEXT"
        elif data_type == "bytea":
            sql_type = "BLOB"
        elif data_type.startswith("character varying"):
            sql_type = "TEXT"
        elif data_type == "integer":
            sql_type = "INTEGER"
        elif data_type == "bigint":
            sql_type = "BIGINT"
        elif data_type == "double precision":
            sql_type = "REAL"
        elif data_type == "ARRAY":
            sql_type = "TEXT"
        else:
            sql_type = "TEXT"

        nullable_clause = "" if is_nullable == "YES" else " NOT NULL"
        column_defs.append(f"    {col_name} {sql_type}{nullable_clause}")

    sql_file.write(",\n".join(column_defs))
    sql_file.write("\n);\n\n")

    # Start transaction
    sql_file.write("-- Data inserts\n")
    sql_file.write("BEGIN TRANSACTION;\n\n")

    return sql_file


def init_csv_file(filename, schema_rows):
    """Initialize CSV file."""
    print(f"💾 Initializing CSV file: {filename}")

    csv_file = open(filename, "w", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_file)

    # Write header
    column_names = [row["column_name"] for row in schema_rows]
    csv_writer.writerow(column_names)

    return csv_file, csv_writer


def write_batch_to_sql(sql_file, batch_rows, column_names):
    """Write a batch of rows to SQL dump file."""
    for row in batch_rows:
        values = []
        for col_name in column_names:
            value = row[col_name]
            if value is None:
                values.append("NULL")
            elif isinstance(value, str):
                # Escape single quotes
                escaped_value = value.replace("'", "''")
                values.append(f"'{escaped_value}'")
            elif isinstance(value, (dict, list)):
                # JSON data or arrays
                json_str = safe_json_dumps(value).replace("'", "''")
                values.append(f"'{json_str}'")
            elif isinstance(value, bytes):
                # Binary data - convert to hex
                hex_str = value.hex()
                values.append(f"decode('{hex_str}', 'hex')")
            elif hasattr(value, "__class__") and "BitString" in str(value.__class__):
                # Handle PostgreSQL BitString objects
                try:
                    # Convert BitString to bytes first, then to hex
                    if hasattr(value, "tobytes"):
                        hex_str = value.tobytes().hex()
                    else:
                        # Fallback: convert to string representation
                        hex_str = str(value).encode().hex()
                    values.append(f"decode('{hex_str}', 'hex')")
                except Exception:
                    # Last resort: convert to string
                    escaped_value = str(value).replace("'", "''")
                    values.append(f"'{escaped_value}'")
            elif isinstance(value, datetime):
                values.append(f"'{value.isoformat()}'")
            else:
                values.append(str(value))

        insert_sql = f"INSERT INTO multi_vector_embeddings ({', '.join(column_names)}) VALUES ({', '.join(values)});\n"
        sql_file.write(insert_sql)


def write_batch_to_sqlite(sqlite_conn, batch_rows, column_names):
    """Write a batch of rows to SQLite database."""
    cursor = sqlite_conn.cursor()

    placeholders = ", ".join(["?" for _ in column_names])
    insert_sql = f"INSERT INTO multi_vector_embeddings ({', '.join(column_names)}) VALUES ({placeholders})"

    batch_data = []
    for row in batch_rows:
        values = []
        for col_name in column_names:
            value = row[col_name]
            if isinstance(value, (dict, list)):
                # Convert JSON/arrays to string
                values.append(safe_json_dumps(value))
            elif hasattr(value, "__class__") and "BitString" in str(value.__class__):
                # Handle PostgreSQL BitString objects
                try:
                    # Convert BitString to bytes for SQLite BLOB storage
                    if hasattr(value, "tobytes"):
                        values.append(value.tobytes())
                    else:
                        # Fallback: convert to string
                        values.append(str(value))
                except Exception:
                    # Last resort: convert to string
                    values.append(str(value))
            elif isinstance(value, datetime):
                values.append(value.isoformat())
            else:
                values.append(value)
        batch_data.append(values)

    cursor.executemany(insert_sql, batch_data)
    sqlite_conn.commit()


def write_batch_to_csv(csv_writer, batch_rows, column_names):
    """Write a batch of rows to CSV file."""
    for row in batch_rows:
        csv_row = []
        for col_name in column_names:
            value = row[col_name]
            if value is None:
                csv_row.append("")
            elif isinstance(value, (dict, list)):
                csv_row.append(safe_json_dumps(value))
            elif isinstance(value, bytes):
                csv_row.append(value.hex())
            elif hasattr(value, "__class__") and "BitString" in str(value.__class__):
                # Handle PostgreSQL BitString objects
                try:
                    # Convert BitString to hex string for CSV
                    if hasattr(value, "tobytes"):
                        csv_row.append(value.tobytes().hex())
                    else:
                        # Fallback: convert to string
                        csv_row.append(str(value))
                except Exception:
                    # Last resort: convert to string
                    csv_row.append(str(value))
            elif isinstance(value, datetime):
                csv_row.append(value.isoformat())
            else:
                csv_row.append(str(value))

        csv_writer.writerow(csv_row)


def finalize_sql_dump(sql_file):
    """Finalize SQL dump file."""
    sql_file.write("\nCOMMIT;\n")
    sql_file.close()
    print("✅ SQL dump completed")


def finalize_sqlite_db(sqlite_conn, filename):
    """Finalize SQLite database."""
    sqlite_conn.close()
    print(f"✅ SQLite database completed: {filename}")


def finalize_csv_file(csv_file):
    """Finalize CSV file."""
    csv_file.close()
    print("✅ CSV file completed")


def export_metadata(schema_rows, row_count, timestamp):
    """Export metadata about the table."""
    filename = f"multi_vector_embeddings_metadata_{timestamp}.json"

    print(f"💾 Creating metadata file: {filename}")

    metadata = {
        "export_timestamp": datetime.now().isoformat(),
        "table_name": "multi_vector_embeddings",
        "row_count": row_count,
        "batch_size": BATCH_SIZE,
        "checkpoint_frequency": CHECKPOINT_FREQUENCY,
        "schema": schema_rows,
    }

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"✅ Metadata saved to: {filename}")


def show_file_info(sql_filename, sqlite_filename, csv_filename):
    """Show information about the exported files."""
    print("\n📁 Export Summary:")

    files = [(sql_filename, "SQL Dump"), (sqlite_filename, "SQLite DB"), (csv_filename, "CSV File")]

    for filename, file_type in files:
        if Path(filename).exists():
            size_mb = Path(filename).stat().st_size / (1024 * 1024)
            print(f"  - {file_type}: {filename} ({size_mb:.1f} MB)")
        else:
            print(f"  - {file_type}: {filename} (not found)")


def show_checkpoint_status():
    """Show current checkpoint status."""
    checkpoint_file = "export_checkpoint.pkl"
    if not Path(checkpoint_file).exists():
        print("📋 No checkpoint found. Starting fresh export.")
        return

    checkpoint = ExportCheckpoint(checkpoint_file)
    if checkpoint.is_completed:
        print("✅ Export already completed!")
        print(f"   Files created with timestamp: {checkpoint.timestamp}")
        return

    progress = (checkpoint.processed_rows / checkpoint.total_rows) * 100 if checkpoint.total_rows > 0 else 0
    print("📋 Checkpoint Status:")
    print(f"   Progress: {checkpoint.processed_rows}/{checkpoint.total_rows} ({progress:.1f}%)")
    print(f"   Batches completed: {checkpoint.batch_count}")
    print(f"   Next offset: {checkpoint.last_offset}")
    print(f"   Export started: {checkpoint.timestamp}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export multi_vector_embeddings table with resume capability")
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
    parser.add_argument("--force-restart", action="store_true", help="Force restart, ignoring any existing checkpoint")
    parser.add_argument("--status", action="store_true", help="Show checkpoint status and exit")

    args = parser.parse_args()

    if args.status:
        show_checkpoint_status()
        sys.exit(0)

    print("🚀 Starting multi_vector_embeddings table export (resumable)...")

    if args.resume:
        print("🔄 Resume mode enabled")
    elif args.force_restart:
        print("🔄 Force restart mode enabled")

    asyncio.run(export_multi_vector_embeddings(resume=args.resume, force_restart=args.force_restart))
