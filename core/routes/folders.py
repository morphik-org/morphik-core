import asyncio
import json
import logging
import uuid
from typing import Dict, List, Union

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import text

from core.auth_utils import verify_token
from core.models.auth import AuthContext
from core.models.folders import Folder, FolderCreate, FolderSummary
from core.models.request import SetFolderRuleRequest
from core.models.responses import (
    DocumentAddToFolderResponse,
    DocumentDeleteResponse,
    FolderDeleteResponse,
    FolderRuleResponse,
)
from core.models.workflows import Workflow
from core.services.telemetry import TelemetryService
from core.services_init import document_service, workflow_service

# ---------------------------------------------------------------------------
# Router initialization & shared singletons
# ---------------------------------------------------------------------------

router = APIRouter(prefix="/folders", tags=["Folders"])
logger = logging.getLogger(__name__)
telemetry = TelemetryService()


# ---------------------------------------------------------------------------
# Folder management endpoints
# ---------------------------------------------------------------------------


@router.post("", response_model=Folder)
@telemetry.track(operation_type="create_folder", metadata_resolver=telemetry.create_folder_metadata)
async def create_folder(
    folder_create: FolderCreate,
    auth: AuthContext = Depends(verify_token),
) -> Folder:
    """
    Create a new folder.

    Args:
        folder_create: Folder creation request containing name and optional description
        auth: Authentication context

    Returns:
        Folder: Created folder
    """
    try:
        # Create a folder object with explicit ID
        folder_id = str(uuid.uuid4())
        logger.info(f"Creating folder with ID: {folder_id}, auth.user_id: {auth.user_id}")

        # Set up access control with user_id
        folder = Folder(
            id=folder_id, name=folder_create.name, description=folder_create.description, app_id=auth.app_id
        )

        # Store in database
        success = await document_service.db.create_folder(folder, auth)

        if not success:
            raise HTTPException(status_code=500, detail="Failed to create folder")

        return folder
    except Exception as e:
        logger.error(f"Error creating folder: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("", response_model=List[Folder])
@telemetry.track(operation_type="list_folders", metadata_resolver=telemetry.list_folders_metadata)
async def list_folders(
    auth: AuthContext = Depends(verify_token),
) -> List[Folder]:
    """
    List all folders the user has access to.

    Args:
        auth: Authentication context

    Returns:
        List[Folder]: List of folders
    """
    try:
        folders = await document_service.db.list_folders(auth)
        return folders
    except Exception as e:
        logger.error(f"Error listing folders: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/summary", response_model=List[FolderSummary])
@telemetry.track(operation_type="list_folders_summary")
async def list_folder_summaries(auth: AuthContext = Depends(verify_token)) -> List[FolderSummary]:
    """Return compact folder list (id, name, doc_count, updated_at)."""

    try:
        summaries = await document_service.db.list_folders_summary(auth)
        return summaries  # type: ignore[return-value]
    except Exception as exc:  # noqa: BLE001
        logger.error("Error listing folder summaries: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/{folder_id}", response_model=Folder)
@telemetry.track(operation_type="get_folder", metadata_resolver=telemetry.get_folder_metadata)
async def get_folder(
    folder_id: str,
    auth: AuthContext = Depends(verify_token),
) -> Folder:
    """
    Get a folder by ID.

    Args:
        folder_id: ID of the folder
        auth: Authentication context

    Returns:
        Folder: Folder if found and accessible
    """
    try:
        folder = await document_service.db.get_folder(folder_id, auth)

        if not folder:
            raise HTTPException(status_code=404, detail=f"Folder {folder_id} not found")

        return folder
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting folder: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{folder_name}", response_model=FolderDeleteResponse)
@telemetry.track(operation_type="delete_folder", metadata_resolver=telemetry.delete_folder_metadata)
async def delete_folder(
    folder_name: str,
    auth: AuthContext = Depends(verify_token),
):
    """
    Delete a folder and all associated documents.

    Args:
        folder_name: Name of the folder to delete
        auth: Authentication context (must have write access to the folder)

    Returns:
        Deletion status
    """
    try:
        folder = await document_service.db.get_folder_by_name(folder_name, auth)
        folder_id = folder.id
        if not folder:
            raise HTTPException(status_code=404, detail="Folder not found")

        document_ids = folder.document_ids
        tasks = [remove_document_from_folder(folder_id, document_id, auth) for document_id in document_ids]
        results = await asyncio.gather(*tasks)
        stati = [res.get("status", False) for res in results]
        if not all(stati):
            failed = [doc for doc, stat in zip(document_ids, stati) if not stat]
            msg = "Failed to remove the following documents from folder: " + ", ".join(failed)
            logger.error(msg)
            raise HTTPException(status_code=500, detail=msg)

        # folder is empty now
        delete_tasks = [document_service.db.delete_document(document_id, auth) for document_id in document_ids]
        stati = await asyncio.gather(*delete_tasks)
        if not all(stati):
            failed = [doc for doc, stat in zip(document_ids, stati) if not stat]
            msg = "Failed to delete the following documents: " + ", ".join(failed)
            logger.error(msg)
            raise HTTPException(status_code=500, detail=msg)

        # just remove the folder too now.
        status = await document_service.db.delete_folder(folder_id, auth)
        if not status:
            logger.error(f"Failed to delete folder {folder_id}")
            raise HTTPException(status_code=500, detail=f"Failed to delete folder {folder_id}")
        return {"status": "success", "message": f"Folder {folder_id} deleted successfully"}
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error deleting folder: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# Folder-Document association endpoints
# ---------------------------------------------------------------------------


@router.post("/{folder_id}/documents/{document_id}", response_model=DocumentAddToFolderResponse)
@telemetry.track(operation_type="add_document_to_folder", metadata_resolver=telemetry.add_document_to_folder_metadata)
async def add_document_to_folder(
    folder_id: str,
    document_id: str,
    auth: AuthContext = Depends(verify_token),
):
    """
    Add a document to a folder.

    Args:
        folder_id: ID of the folder
        document_id: ID of the document
        auth: Authentication context

    Returns:
        Success status
    """
    try:
        success = await document_service.db.add_document_to_folder(folder_id, document_id, auth)

        if not success:
            raise HTTPException(status_code=500, detail="Failed to add document to folder")

        return {"status": "success"}
    except Exception as e:
        logger.error(f"Error adding document to folder: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{folder_id}/documents/{document_id}", response_model=DocumentDeleteResponse)
@telemetry.track(
    operation_type="remove_document_from_folder", metadata_resolver=telemetry.remove_document_from_folder_metadata
)
async def remove_document_from_folder(
    folder_id: str,
    document_id: str,
    auth: AuthContext = Depends(verify_token),
):
    """
    Remove a document from a folder.

    Args:
        folder_id: ID of the folder
        document_id: ID of the document
        auth: Authentication context

    Returns:
        Success status
    """
    try:
        success = await document_service.db.remove_document_from_folder(folder_id, document_id, auth)

        if not success:
            raise HTTPException(status_code=500, detail="Failed to remove document from folder")

        return {"status": "success"}
    except Exception as e:
        logger.error(f"Error removing document from folder: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# Folder rules endpoints
# ---------------------------------------------------------------------------


@router.post("/{folder_id}/set_rule", response_model=FolderRuleResponse)
@telemetry.track(operation_type="set_folder_rule", metadata_resolver=telemetry.set_folder_rule_metadata)
async def set_folder_rule(
    folder_id: str,
    request: SetFolderRuleRequest,
    auth: AuthContext = Depends(verify_token),
    apply_to_existing: bool = True,
):
    """
    Set extraction rules for a folder.

    Args:
        folder_id: ID of the folder to set rules for
        request: SetFolderRuleRequest containing metadata extraction rules
        auth: Authentication context
        apply_to_existing: Whether to apply rules to existing documents in the folder

    Returns:
        Success status with processing results
    """
    try:
        # Log detailed information about the rules
        logger.debug(f"Setting rules for folder {folder_id}")
        logger.debug(f"Number of rules: {len(request.rules)}")

        for i, rule in enumerate(request.rules):
            logger.debug(f"\nRule {i + 1}:")
            logger.debug(f"Type: {rule.type}")
            logger.debug("Schema:")
            for field_name, field_config in rule.schema.items():
                logger.debug(f"  Field: {field_name}")
                logger.debug(f"    Type: {field_config.get('type', 'unknown')}")
                logger.debug(f"    Description: {field_config.get('description', 'No description')}")
                if "schema" in field_config:
                    logger.debug("    Has JSON schema: Yes")
                    logger.debug(f"    Schema: {field_config['schema']}")

        # Get the folder
        folder = await document_service.db.get_folder(folder_id, auth)
        if not folder:
            raise HTTPException(status_code=404, detail=f"Folder {folder_id} not found")

        # Note: Write access will be checked by the database operations

        # Update folder with rules
        # Convert rules to dicts for JSON serialization
        rules_dicts = [rule.model_dump() for rule in request.rules]

        # Update the folder in the database
        async with document_service.db.async_session() as session:
            # Execute update query
            await session.execute(
                text(
                    """
                    UPDATE folders
                    SET rules = :rules
                    WHERE id = :folder_id
                    """
                ),
                {"folder_id": folder_id, "rules": json.dumps(rules_dicts)},
            )
            await session.commit()

        logger.info(f"Successfully updated folder {folder_id} with {len(request.rules)} rules")

        # Get updated folder
        updated_folder = await document_service.db.get_folder(folder_id, auth)

        # If apply_to_existing is True, apply these rules to all existing documents in the folder
        processing_results = {"processed": 0, "errors": []}

        if apply_to_existing and folder.document_ids:
            logger.info(f"Applying rules to {len(folder.document_ids)} existing documents in folder")

            # Get all documents in the folder
            documents = await document_service.db.get_documents_by_id(folder.document_ids, auth)

            # Process each document
            for doc in documents:
                try:
                    # Get document content
                    logger.info(f"Processing document {doc.external_id}")

                    # For each document, apply the rules from the folder
                    doc_content = None

                    # Get content from system_metadata if available
                    if doc.system_metadata and "content" in doc.system_metadata:
                        doc_content = doc.system_metadata["content"]
                        logger.info(f"Retrieved content from system_metadata for document {doc.external_id}")

                    # If we still have no content, log error and continue
                    if not doc_content:
                        error_msg = f"No content found in system_metadata for document {doc.external_id}"
                        logger.error(error_msg)
                        processing_results["errors"].append({"document_id": doc.external_id, "error": error_msg})
                        continue

                    # Process document with rules
                    try:
                        # Convert request rules to actual rule models and apply them
                        from core.models.rules import MetadataExtractionRule

                        for rule_request in request.rules:
                            if rule_request.type == "metadata_extraction":
                                # Create the actual rule model
                                rule = MetadataExtractionRule(type=rule_request.type, schema=rule_request.schema)

                                # Apply the rule with retries
                                max_retries = 3
                                base_delay = 1  # seconds
                                extracted_metadata = None
                                last_error = None

                                for retry_count in range(max_retries):
                                    try:
                                        if retry_count > 0:
                                            # Exponential backoff
                                            delay = base_delay * (2 ** (retry_count - 1))
                                            logger.info(f"Retry {retry_count}/{max_retries} after {delay}s delay")
                                            await asyncio.sleep(delay)

                                        extracted_metadata, _ = await rule.apply(doc_content, {})
                                        logger.info(
                                            f"Successfully extracted metadata on attempt {retry_count + 1}: "
                                            f"{extracted_metadata}"
                                        )
                                        break  # Success, exit retry loop

                                    except Exception as rule_apply_error:
                                        last_error = rule_apply_error
                                        logger.warning(
                                            f"Metadata extraction attempt {retry_count + 1} failed: "
                                            f"{rule_apply_error}"
                                        )
                                        if retry_count == max_retries - 1:  # Last attempt
                                            logger.error(f"All {max_retries} metadata extraction attempts failed")
                                            processing_results["errors"].append(
                                                {
                                                    "document_id": doc.external_id,
                                                    "error": f"Failed to extract metadata after {max_retries} "
                                                    f"attempts: {str(last_error)}",
                                                }
                                            )
                                            continue  # Skip to next document

                                # Update document metadata if extraction succeeded
                                if extracted_metadata:
                                    # Merge new metadata with existing
                                    doc.metadata.update(extracted_metadata)

                                    # Create an updates dict that only updates metadata
                                    # We need to create system_metadata with all preserved fields
                                    # Note: In the database, metadata is stored as 'doc_metadata', not 'metadata'
                                    updates = {
                                        "doc_metadata": doc.metadata,  # Use doc_metadata for the database
                                        "system_metadata": {},  # Will be merged with existing in update_document
                                    }

                                    # Explicitly preserve the content field in system_metadata
                                    if "content" in doc.system_metadata:
                                        updates["system_metadata"]["content"] = doc.system_metadata["content"]

                                    # Log the updates we're making
                                    logger.info(
                                        f"Updating document {doc.external_id} with metadata: {extracted_metadata}"
                                    )
                                    logger.info(f"Full metadata being updated: {doc.metadata}")
                                    logger.info(f"Update object being sent to database: {updates}")
                                    logger.info(
                                        f"Preserving content in system_metadata: {'content' in doc.system_metadata}"
                                    )

                                    # Update document in database
                                    app_db = document_service.db
                                    success = await app_db.update_document(doc.external_id, updates, auth)

                                    if success:
                                        logger.info(f"Updated metadata for document {doc.external_id}")
                                        processing_results["processed"] += 1
                                    else:
                                        logger.error(f"Failed to update metadata for document {doc.external_id}")
                                        processing_results["errors"].append(
                                            {
                                                "document_id": doc.external_id,
                                                "error": "Failed to update document metadata",
                                            }
                                        )
                    except Exception as rule_error:
                        logger.error(f"Error processing rules for document {doc.external_id}: {rule_error}")
                        processing_results["errors"].append(
                            {
                                "document_id": doc.external_id,
                                "error": f"Error processing rules: {str(rule_error)}",
                            }
                        )

                except Exception as doc_error:
                    logger.error(f"Error processing document {doc.external_id}: {doc_error}")
                    processing_results["errors"].append({"document_id": doc.external_id, "error": str(doc_error)})

            return {
                "status": "success",
                "message": "Rules set successfully",
                "folder_id": folder_id,
                "rules": updated_folder.rules,
                "processing_results": processing_results,
            }
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Error setting folder rules: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# Folder-Workflow association endpoints
# ---------------------------------------------------------------------------


@router.post("/{folder_id}/workflows/{workflow_id}")
@telemetry.track(operation_type="associate_workflow_to_folder")
async def associate_workflow_to_folder(
    folder_id: str,
    workflow_id: str,
    auth: AuthContext = Depends(verify_token),
) -> Dict[str, Union[bool, str]]:
    """Associate a workflow with a folder for automatic execution on document ingestion."""
    try:
        # Verify the workflow exists and is accessible
        workflow = await workflow_service.get_workflow(workflow_id, auth)
        if not workflow:
            raise HTTPException(status_code=404, detail=f"Workflow {workflow_id} not found or not accessible")

        # Use the database method which handles access control properly
        success = await document_service.db.associate_workflow_to_folder(folder_id, workflow_id, auth)

        if not success:
            # Check if folder exists by trying to get it
            folder = await document_service.db.get_folder(folder_id, auth)
            if not folder:
                raise HTTPException(status_code=404, detail=f"Folder {folder_id} not found")
            else:
                raise HTTPException(status_code=403, detail="You don't have write access to this folder")

        return {"success": True, "message": f"Successfully associated workflow {workflow_id} with folder {folder_id}"}
    except Exception as e:
        logger.error(f"Error associating workflow to folder: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{folder_id}/workflows/{workflow_id}")
@telemetry.track(operation_type="disassociate_workflow_from_folder")
async def disassociate_workflow_from_folder(
    folder_id: str,
    workflow_id: str,
    auth: AuthContext = Depends(verify_token),
) -> Dict[str, Union[bool, str]]:
    """Remove a workflow association from a folder."""
    try:
        # Use the database method which handles access control properly
        success = await document_service.db.disassociate_workflow_from_folder(folder_id, workflow_id, auth)

        if not success:
            # Check if folder exists by trying to get it
            folder = await document_service.db.get_folder(folder_id, auth)
            if not folder:
                raise HTTPException(status_code=404, detail=f"Folder {folder_id} not found")
            else:
                raise HTTPException(status_code=403, detail="You don't have write access to this folder")

        return {"success": True, "message": f"Successfully removed workflow {workflow_id} from folder {folder_id}"}
    except Exception as e:
        logger.error(f"Error disassociating workflow from folder: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{folder_id}/workflows", response_model=List[Workflow])
@telemetry.track(operation_type="list_folder_workflows")
async def list_folder_workflows(
    folder_id: str,
    auth: AuthContext = Depends(verify_token),
) -> List[Workflow]:
    """List all workflows associated with a folder."""
    try:
        # Get the folder
        folder = await document_service.db.get_folder(folder_id, auth)
        if not folder:
            raise HTTPException(status_code=404, detail=f"Folder {folder_id} not found")

        # Get all workflows
        workflows = []
        for workflow_id in folder.workflow_ids:
            workflow = await workflow_service.get_workflow(workflow_id, auth)
            if workflow:
                workflows.append(workflow)

        return workflows

    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc))
