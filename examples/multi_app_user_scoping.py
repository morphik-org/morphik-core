import os
from dotenv import load_dotenv
from morphik import Morphik

# Load environment variables
load_dotenv()

# Connect to Morphik
db = Morphik(os.getenv("MORPHIK_URI"), timeout=10000, is_local=True)

print("========== Folder Scoping Example ==========")
# Create a folder for application data
app_folder = db.create_folder("customer-support")
print(f"Created folder: {app_folder.name}")

# Ingest documents into the folder
folder_doc = app_folder.ingest_text(
    "Customer reported an issue with login functionality. Steps to reproduce: "
    "1. Go to login page, 2. Enter credentials, 3. Click login button.",
    filename="ticket-001.txt",
    metadata={"category": "bug", "priority": "high", "status": "open"}
)
print(f"Ingested document into folder: {folder_doc.external_id}")

# Perform a query in the folder context
folder_response = app_folder.query(
    "What issues have been reported?",
    k=2
)
print("\nFolder Query Results:")
print(folder_response.completion)

# Get statistics for the folder
folder_docs = app_folder.list_documents()
print(f"\nFolder Statistics: {len(folder_docs)} documents in '{app_folder.name}'")

print("\n========== User Scoping Example ==========")
# Create a user scope
user_email = "support@example.com"
user = db.signin(user_email)
print(f"Created user scope for: {user.end_user_id}")

# Ingest a document as this user 
user_doc = user.ingest_text(
    "User requested information about premium features. They are interested in the collaboration tools.",
    filename="inquiry-001.txt",
    metadata={"category": "inquiry", "priority": "medium", "status": "open"}
)
print(f"Ingested document as user: {user_doc.external_id}")

# Query as this user
user_response = user.query(
    "What customer inquiries do we have?",
    k=2
)
print("\nUser Query Results:")
print(user_response.completion)

# Get documents for this user
user_docs = user.list_documents()
print(f"\nUser Statistics: {len(user_docs)} documents for user '{user.end_user_id}'")

print("\n========== Combined Folder and User Scoping ==========")
# Create a user scoped to a specific folder
folder_user = app_folder.signin(user_email)
print(f"Created user scope for {folder_user.end_user_id} in folder {folder_user.folder_name}")

# Ingest a document as this user in the folder context
folder_user_doc = folder_user.ingest_text(
    "Customer called to follow up on ticket-001. They are still experiencing the login issue on Chrome.",
    filename="ticket-002.txt",
    metadata={"category": "follow-up", "priority": "high", "status": "open"}
)
print(f"Ingested document as user in folder: {folder_user_doc.external_id}")

# Query as this user in the folder context
folder_user_response = folder_user.query(
    "What high priority issues require attention?",
    k=2
)
print("\nFolder User Query Results:")
print(folder_user_response.completion)

# Update a document with user scope in folder context
updated_doc = folder_user.update_document_with_text(
    document_id=folder_doc.external_id,
    content="\nUpdate: Issue confirmed in Chrome version 120. Working on a fix.",
    update_strategy="add",
    metadata={"status": "in-progress"}
)
print(f"\nUser updated document in folder. New status: {updated_doc.metadata.get('status')}")

print("\nExample complete! Successfully demonstrated folder and user scoping")