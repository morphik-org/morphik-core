import botocore
from dotenv import load_dotenv, find_dotenv
import os
import boto3
import logging
import tomli  # for reading toml files
from pathlib import Path
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, OperationFailure
from pymongo.operations import SearchIndexModel
import argparse

# Force reload of environment variables
load_dotenv(find_dotenv(), override=True)

# Set up argument parser
parser = argparse.ArgumentParser(description="Setup S3 bucket and MongoDB collections")
parser.add_argument("--debug", action="store_true", help="Enable debug logging")
parser.add_argument(
    "--quiet", action="store_true", help="Only show warning and error logs"
)
args = parser.parse_args()

# Configure logging based on command line arguments
LOGGER = logging.getLogger(__name__)
if args.debug:
    LOGGER.setLevel(logging.DEBUG)
elif args.quiet:
    LOGGER.setLevel(logging.WARNING)
else:
    LOGGER.setLevel(logging.INFO)

# Add console handler with formatting
console_handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
LOGGER.addHandler(console_handler)

# Load configuration from config.toml
config_path = Path("config.toml")
with open(config_path, "rb") as f:
    CONFIG = tomli.load(f)
    LOGGER.info("Loaded configuration from config.toml")

# Extract configuration values
DEFAULT_REGION = CONFIG["aws"]["default_region"]
DEFAULT_BUCKET_NAME = CONFIG["aws"]["default_bucket_name"]
DATABASE_NAME = CONFIG["mongodb"]["database_name"]
DOCUMENTS_COLLECTION = CONFIG["mongodb"]["documents_collection"]
CHUNKS_COLLECTION = CONFIG["mongodb"]["chunks_collection"]
VECTOR_DIMENSIONS = CONFIG["mongodb"]["vector"]["dimensions"]
VECTOR_INDEX_NAME = CONFIG["mongodb"]["vector"]["index_name"]


def create_s3_bucket(bucket_name, region=DEFAULT_REGION):
    # Clear any existing AWS credentials from environment
    boto3.Session().resource("s3").meta.client.close()

    aws_access_key = os.getenv("AWS_ACCESS_KEY")
    aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
    region = os.getenv("AWS_REGION") if os.getenv("AWS_REGION") else region

    if not aws_access_key or not aws_secret_key:
        LOGGER.error("AWS credentials not found in environment variables.")
        return

    LOGGER.debug("Successfully retrieved AWS credentials and region.")
    # Create new session with explicit credentials
    session = boto3.Session(
        aws_access_key_id=aws_access_key,
        aws_secret_access_key=aws_secret_key,
        region_name=region,
    )

    s3_client = session.client("s3")
    LOGGER.debug("Successfully created S3 client.")

    # create_bucket = not
    if bucket_exists(s3_client, bucket_name):
        LOGGER.info(f"Bucket with name {bucket_name} already exists")
        return
    # Create bucket with location constraint if region is not us-east-1
    if region == "us-east-1":
        s3_client.create_bucket(Bucket=bucket_name)
    else:
        s3_client.create_bucket(
            Bucket=bucket_name, CreateBucketConfiguration={"LocationConstraint": region}
        )

    LOGGER.debug(f"Bucket {bucket_name} created successfully in {region} region.")


def bucket_exists(s3_client, bucket_name):
    """
    Check if an S3 bucket exists.
    """
    try:
        s3_client.head_bucket(Bucket=bucket_name)
        return True
    except botocore.exceptions.ClientError as e:
        error_code = int(e.response["Error"]["Code"])
        if error_code == 404:
            return False
        raise


def setup_mongodb():
    """
    Set up MongoDB database, documents collection, and vector index on documents_chunk collection.
    """
    # Load MongoDB URI from .env file
    mongo_uri = os.getenv("MONGODB_URI")
    if not mongo_uri:
        raise ValueError("MONGODB_URI not found in .env file.")

    try:
        # Connect to MongoDB
        client = MongoClient(mongo_uri)
        client.admin.command("ping")  # Check connection
        LOGGER.info("Connected to MongoDB successfully.")

        # Create or access the database
        db = client[DATABASE_NAME]
        LOGGER.info(f"Database '{DATABASE_NAME}' ready.")

        # Create 'documents' collection
        if DOCUMENTS_COLLECTION not in db.list_collection_names():
            db.create_collection(DOCUMENTS_COLLECTION)
            LOGGER.info(f"Collection '{DOCUMENTS_COLLECTION}' created.")
        else:
            LOGGER.info(f"Collection '{DOCUMENTS_COLLECTION}' already exists.")

        # Create 'documents_chunk' collection with vector index
        if CHUNKS_COLLECTION not in db.list_collection_names():
            db.create_collection(CHUNKS_COLLECTION)
            LOGGER.info(f"Collection '{CHUNKS_COLLECTION}' created.")
        else:
            LOGGER.info(f"Collection '{CHUNKS_COLLECTION}' already exists.")

        vector_index_definition = {
            "fields": [
                {
                    "numDimensions": VECTOR_DIMENSIONS,
                    "path": "embedding",
                    "similarity": "dotProduct",
                    "type": "vector",
                },
                {"path": "document_id", "type": "filter"},
            ]
        }
        vector_index = SearchIndexModel(
            name=VECTOR_INDEX_NAME,
            definition=vector_index_definition,
            type="vectorSearch",
        )
        db[CHUNKS_COLLECTION].create_search_index(model=vector_index)
        LOGGER.info(
            "Vector index 'vector_index' created on 'documents_chunk' collection."
        )

    except ConnectionFailure:
        LOGGER.error(
            "Failed to connect to MongoDB. Check your MongoDB URI and network connection."
        )
    except OperationFailure as e:
        LOGGER.error(f"MongoDB operation failed: {e}")
    except Exception as e:
        LOGGER.error(f"Unexpected error: {e}")
    finally:
        client.close()
        LOGGER.info("MongoDB connection closed.")


def setup():
    LOGGER.info("Creating S3 bucket...")
    create_s3_bucket(DEFAULT_BUCKET_NAME)
    LOGGER.info("S3 bucket created successfully.")

    LOGGER.info("Setting up MongoDB...")
    setup_mongodb()
    LOGGER.info("MongoDB setup completed.")

    LOGGER.info("Setup completed successfully. Feel free to start the server now!")


if __name__ == "__main__":
    setup()
