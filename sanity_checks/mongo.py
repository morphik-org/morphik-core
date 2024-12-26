from pymongo import MongoClient
from dotenv import load_dotenv
import os
import datetime


def test_mongo_operations():
    # Load environment variables
    load_dotenv()

    # Get MongoDB URI from environment variable
    mongo_uri = os.getenv("MONGODB_URI")
    if not mongo_uri:
        raise ValueError("MONGODB_URI environment variable not set")

    try:
        # Connect to MongoDB
        client = MongoClient(mongo_uri)

        # Test connection
        client.admin.command("ping")
        print("✅ Connected successfully to MongoDB")

        # Get database and collection
        db = client.brandsyncaidb  # Using a test database
        collection = db.kb_chunked_embeddings

        # Insert a single document
        test_doc = {
            "name": "Test Document",
            "timestamp": datetime.datetime.now(),
            "value": 42,
        }

        result = collection.insert_one(test_doc)
        print(f"✅ Inserted document with ID: {result.inserted_id}")

        # Insert multiple documents
        test_docs = [
            {"name": "Doc 1", "value": 1},
            {"name": "Doc 2", "value": 2},
            {"name": "Doc 3", "value": 3},
        ]

        result = collection.insert_many(test_docs)
        print(f"✅ Inserted {len(result.inserted_ids)} documents")

        # Retrieve documents
        print("\nRetrieving documents:")
        for doc in collection.find():
            print(f"Found document: {doc}")

        # Find specific documents
        print("\nFinding documents with value >= 2:")
        query = {"value": {"$gte": 2}}
        for doc in collection.find(query):
            print(f"Found document: {doc}")

        # Clean up - delete all test documents
        # DON'T DELETE IF It'S BRANDSYNCAI
        # result = collection.delete_many({})
        print(f"\n✅ Cleaned up {result.deleted_count} test documents")

    except Exception as e:
        print(f"❌ Error: {str(e)}")
    finally:
        client.close()
        print("\n✅ Connection closed")


if __name__ == "__main__":
    test_mongo_operations()
