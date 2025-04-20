#!/usr/bin/env python
import argparse
import asyncio
import json
import logging
import os

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


async def get_postgres_session(uri: str) -> AsyncSession:
    """Create and return a PostgreSQL session."""
    engine = create_async_engine(uri, echo=False)
    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    return async_session()


async def scrub_document_metadata(
    postgres_uri: str, preserve_external_id_only: bool = True, batch_size: int = 100
) -> None:
    """
    Scrub metadata from all documents in the database,
    keeping only external_id if preserve_external_id_only is True.

    Args:
        postgres_uri: PostgreSQL connection URI
        preserve_external_id_only: If True, preserve only external_id in metadata
        batch_size: Number of documents to process in each batch
    """
    try:
        async with await get_postgres_session(postgres_uri) as session:
            # Get total count of documents
            count_result = await session.execute(text("SELECT COUNT(*) FROM documents"))
            total_docs = count_result.scalar()
            logger.info(f"Found {total_docs} documents to process")

            # Process in batches
            offset = 0
            total_processed = 0
            total_updated = 0

            while offset < total_docs:
                # Get batch of document IDs
                id_result = await session.execute(
                    text(f"SELECT external_id FROM documents LIMIT {batch_size} OFFSET {offset}")
                )
                doc_ids = [row[0] for row in id_result.all()]

                # Process each document in the batch
                for doc_id in doc_ids:
                    # Create new metadata object with only external_id
                    if preserve_external_id_only:
                        new_metadata = {"external_id": doc_id}
                    else:
                        # This would be where you could implement more complex preservation rules
                        new_metadata = {"external_id": doc_id}

                    # Use fully manual query with directly inserted values
                    json_string = json.dumps(new_metadata).replace("'", "''")
                    # Use a direct query using string formatting (safe in this context since we control the values)
                    query = text(
                        f"""
                        UPDATE documents
                        SET doc_metadata = '{json_string}'::jsonb
                        WHERE external_id = '{doc_id}'
                    """
                    )

                    await session.execute(query)
                    total_updated += 1

                # Commit changes for this batch
                await session.commit()
                total_processed += len(doc_ids)
                offset += batch_size
                logger.info(f"Processed {total_processed}/{total_docs} documents, updated {total_updated}")

            logger.info(f"Metadata scrubbing complete. Processed {total_processed} documents, updated {total_updated}.")

    except Exception as e:
        logger.error(f"Error scrubbing document metadata: {e}")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scrub metadata from documents, preserving only external_id")
    parser.add_argument(
        "--env",
        choices=["docker", "local"],
        required=True,
        help="Environment to run in (docker or local)",
    )
    parser.add_argument("--batch-size", type=int, default=100, help="Number of documents to process in each batch")

    args = parser.parse_args()

    # Get database URI based on environment
    if args.env == "docker":
        # Using Docker container
        postgres_uri = "postgresql+asyncpg://morphik:morphik@localhost:5432/morphik"
    else:
        # Using local .env file
        try:
            # Try to load from .env file
            from dotenv import load_dotenv

            load_dotenv()
            postgres_uri = os.environ.get("POSTGRES_URI")
            if not postgres_uri:
                raise ValueError("POSTGRES_URI not found in environment variables")
        except ImportError:
            # If python-dotenv is not installed
            postgres_uri = os.environ.get("POSTGRES_URI")
            if not postgres_uri:
                raise ValueError(
                    "POSTGRES_URI not found in environment variables. Install python-dotenv or set POSTGRES_URI manually."
                )

    logger.info(f"Starting metadata scrubbing in {args.env} environment")
    asyncio.run(scrub_document_metadata(postgres_uri, batch_size=args.batch_size))
