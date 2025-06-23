# Workflow System Implementation

## Overview
This document describes the improvements made to the Morphik workflow system, enabling automated document processing through multi-step workflows.

## New Features

### 1. Workflow Actions
Three core workflow actions have been implemented:

#### a. Extract Structured Data (`morphik.actions.extract_structured`)
- Extracts structured JSON data from documents using LLM
- Supports custom schemas with field types and descriptions
- Uses PDF viewer tools for all document types

#### b. Apply Custom Instruction (`morphik.actions.apply_instruction`)
- Applies custom LLM instructions to transform documents
- Supports prompt templates with `{input_text}` placeholder
- Configurable model, temperature, and max tokens

#### c. Save to Metadata (`morphik.actions.save_to_metadata`)
- Saves workflow output to document metadata
- Can save output from previous step or all steps
- Stores data under a specified metadata key

### 2. Folder-Workflow Association
- Folders now have a `workflow_ids` field to associate workflows
- API endpoints for managing folder-workflow associations:
  - `POST /folders/{folder_id}/workflows/{workflow_id}` - Associate workflow
  - `DELETE /folders/{folder_id}/workflows/{workflow_id}` - Remove association
  - `GET /folders/{folder_id}/workflows` - List associated workflows

### 3. Automatic Workflow Execution
- Workflows automatically execute when documents are added to folders
- Runs asynchronously to avoid blocking document ingestion
- Each workflow runs independently with error isolation

### 4. Database Persistence
- Workflows and workflow runs are persisted in PostgreSQL
- Database models: `WorkflowModel` and `WorkflowRunModel`
- Migration script provided for existing databases

## Architecture

### Workflow Execution Flow
1. Document is ingested and added to a folder
2. System checks for workflows associated with the folder
3. For each workflow, a run is queued
4. Workflow steps execute sequentially
5. Each step receives outputs from previous steps
6. Results are stored and document metadata is updated

### Action Registry
- Actions are dynamically discovered from `core/workflows/actions/`
- Each action module exports:
  - `definition`: ActionDefinition with metadata
  - `run()`: Async function to execute the action

## Usage Example

### Creating a Workflow
```json
{
  "name": "Invoice Processing",
  "description": "Extract data from invoices and save to metadata",
  "steps": [
    {
      "action_id": "morphik.actions.extract_structured",
      "parameters": {
        "schema": {
          "type": "object",
          "properties": {
            "invoice_number": {"type": "string"},
            "amount": {"type": "number"},
            "date": {"type": "string"}
          },
          "required": ["invoice_number", "amount"]
        }
      }
    },
    {
      "action_id": "morphik.actions.save_to_metadata",
      "parameters": {
        "metadata_key": "invoice_data",
        "source": "previous_step"
      }
    }
  ]
}
```

### Associating with Folder
```bash
# Associate workflow with folder
curl -X POST "http://localhost:8000/folders/{folder_id}/workflows/{workflow_id}" \
  -H "Authorization: Bearer $TOKEN"

# Documents added to this folder will now trigger the workflow
```

## Implementation Details

### Key Files Modified/Added
- `core/workflows/actions/save_to_metadata.py` - Save to metadata action
- `core/workflows/actions/apply_instruction.py` - Apply instruction action
- `core/models/folders.py` - Added workflow_ids field
- `core/api.py` - Added folder-workflow association endpoints
- `core/services/document_service.py` - Added automatic workflow execution
- `core/services/workflow_service.py` - Enhanced to pass outputs between steps

### Database Migration
Run the migration script to add workflow support to existing databases:
```sql
psql -d your_database -f migrations/add_workflow_ids_to_folders.sql
```

## Future Improvements
1. Add more workflow actions (e.g., send to webhook, generate summary)
2. Implement workflow templates for common use cases
3. Add workflow execution monitoring and retry logic
4. Support conditional workflow steps
5. Implement workflow versioning
6. Add bulk workflow operations for folders
