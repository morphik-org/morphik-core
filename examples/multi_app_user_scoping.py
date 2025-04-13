import os
from dotenv import load_dotenv
from morphik import Morphik

# Load environment variables
load_dotenv()

# Connect to Morphik
db = Morphik(os.getenv("MORPHIK_URI"), timeout=10000, is_local=True)

# Example: Multiple applications and users with folder scoping
print("Example: Multiple applications and users with folder scoping")

# Create folders for different applications
app1_folder = db.create_folder("customer-service-app")
app2_folder = db.create_folder("content-management-app")
app3_folder = db.create_folder("knowledge-base-app")
print(f"Created folders for different applications")

# Simulate different users in each application
users = {
    "app1": ["user1@example.com", "user2@example.com"],
    "app2": ["editor@example.com", "reviewer@example.com", "admin@example.com"],
    "app3": ["kb-user@example.com", "kb-admin@example.com"]
}

# Create user scopes for each user
user_scopes = {}
for app, app_users in users.items():
    for user_email in app_users:
        user_scopes[user_email] = db.signin(user_email)
        print(f"Created user scope for {user_email}")

# Ingest documents for different applications and users
print("\nIngesting documents for different applications and users")

# App 1: Customer Service App - Ingest customer support tickets
user1 = user_scopes["user1@example.com"]
ticket1 = app1_folder.ingest_text(
    "Customer reported an issue with login functionality. Steps to reproduce: 1. Go to login page, 2. Enter credentials, 3. Click login button.",
    filename="ticket-001.txt",
    metadata={"category": "bug", "priority": "high", "status": "open"},
    end_user_id="user1@example.com"
)

user2 = user_scopes["user2@example.com"]
ticket2 = app1_folder.ingest_text(
    "Customer requested information about premium features. They are interested in the collaboration tools.",
    filename="ticket-002.txt",
    metadata={"category": "inquiry", "priority": "medium", "status": "open"},
    end_user_id="user2@example.com"
)

# App 2: Content Management App - Ingest content drafts
editor = user_scopes["editor@example.com"]
draft1 = app2_folder.ingest_text(
    "# New Product Launch\n\nWe're excited to announce our newest product that will revolutionize the industry...",
    filename="product-launch-draft.md",
    metadata={"type": "blog", "status": "draft", "campaign": "summer-launch"},
    end_user_id="editor@example.com"
)

reviewer = user_scopes["reviewer@example.com"]
feedback = app2_folder.ingest_text(
    "Reviewed the product launch draft. Needs more specific details about features and benefits. Also, add a section about pricing tiers.",
    filename="product-launch-feedback.txt",
    metadata={"type": "review", "status": "pending", "reference": "product-launch-draft.md"},
    end_user_id="reviewer@example.com"
)

# App 3: Knowledge Base App - Ingest knowledge articles
kb_admin = user_scopes["kb-admin@example.com"]
article1 = app3_folder.ingest_text(
    "# How to Reset Your Password\n\n1. Go to the login page\n2. Click 'Forgot Password'\n3. Follow the instructions sent to your email",
    filename="password-reset.md",
    metadata={"category": "account", "visibility": "public", "version": "1.0"},
    end_user_id="kb-admin@example.com"
)

article2 = app3_folder.ingest_text(
    "# Advanced Configuration Settings\n\nThis guide explains the various configuration options available to administrators...",
    filename="admin-config.md",
    metadata={"category": "administration", "visibility": "internal", "version": "2.1"},
    end_user_id="kb-admin@example.com"
)

print(f"Ingested documents across multiple applications and users")

# Demonstrate querying within specific application context
print("\nQuerying within specific application contexts")

# Query within Customer Service App
app1_response = app1_folder.query(
    "What are the current open tickets?",
    k=2
)
print("Customer Service App Query Results:")
print(app1_response.completion)

# Query within Content Management App
app2_response = app2_folder.query(
    "What content is being worked on for the summer launch?",
    k=2
)
print("\nContent Management App Query Results:")
print(app2_response.completion)

# Query within Knowledge Base App with user scope
kb_user = user_scopes["kb-user@example.com"]
kb_user_response = kb_user.query(
    "Show me public knowledge base articles",
    k=2,
    folder_name="knowledge-base-app"
)
print("\nKnowledge Base App Query Results (User View):")
print(kb_user_response.completion)

# Demonstrate user-specific operations
print("\nUser-specific operations")

# User1 updating their ticket
updated_ticket = user1.update_document_with_text(
    document_id=ticket1.external_id,
    content="\nUpdate: Confirmed this happens on Chrome but not on Firefox.",
    update_strategy="add",
    metadata={"status": "in-progress"},
    folder_name="customer-service-app"  # Specify the application context
)
print(f"User1 updated their ticket: {updated_ticket.metadata}")

# Editor updating their draft based on reviewer feedback
updated_draft = editor.update_document_with_text(
    document_id=draft1.external_id,
    content="\n\n## Features\n\n* Real-time collaboration\n* Advanced analytics\n* Custom integrations\n\n## Pricing\n\n* Basic: $10/month\n* Pro: $25/month\n* Enterprise: Contact sales",
    update_strategy="add",
    metadata={"status": "revised"},
    folder_name="content-management-app"  # Specify the application context
)
print(f"Editor updated their draft: {updated_draft.metadata}")

# Cross-application operations (administrative)
print("\nCross-application operations (administrative view)")

# Get statistics across all applications
app_stats = {}
for app_name, folder in [("Customer Service", app1_folder), 
                         ("Content Management", app2_folder),
                         ("Knowledge Base", app3_folder)]:
    docs = folder.get_documents()
    app_stats[app_name] = {
        "document_count": len(docs),
        "unique_users": len(set(doc.system_metadata.get("end_user_id") for doc in docs if doc.system_metadata.get("end_user_id")))
    }

print("Application Statistics:")
for app_name, stats in app_stats.items():
    print(f"  {app_name}: {stats['document_count']} documents, {stats['unique_users']} users")

# Global search across all applications (admin function)
all_docs = db.get_documents()
print(f"\nTotal documents across all applications: {len(all_docs)}")

# Example of how a multi-tenant application would work
print("\nMulti-tenant application example")
print("Each folder represents a separate tenant's data:")
for folder_name in ["customer-service-app", "content-management-app", "knowledge-base-app"]:
    folder = db.get_folder(folder_name)
    tenant_docs = folder.get_documents()
    print(f"  Tenant '{folder_name}': {len(tenant_docs)} documents")
    
    # Users within each tenant
    tenant_users = set()
    for doc in tenant_docs:
        user_id = doc.system_metadata.get("end_user_id")
        if user_id:
            tenant_users.add(user_id)
            
    print(f"  Users in tenant '{folder_name}': {', '.join(tenant_users)}")

print("\nDemo complete: Successfully demonstrated folder and user scoping across multiple applications")