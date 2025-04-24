from __future__ import annotations

import json
from enum import Enum
from typing import Any, Dict, List, Optional, Type, Union

from pydantic import BaseModel, Field


class PatchOp(str, Enum):
    """JSON Patch operation types as defined in RFC 6902."""

    ADD = "add"
    REMOVE = "remove"
    REPLACE = "replace"
    MOVE = "move"
    COPY = "copy"
    TEST = "test"


def gather_paths(schema: Dict[str, Any], base_pointer: str = "") -> Dict[str, Dict[str, Any]]:
    """
    Walk a JSON Schema and map every reachable JSON Pointer -> subschema.
    For objects: recurse into 'properties'.
    For arrays: treat the items schema as the path '/*'.
    """
    paths = {}

    # Add current path
    if base_pointer:
        paths[base_pointer] = schema

    # Handle base schema type
    if "type" not in schema and base_pointer:
        # Ensure there's always a type
        schema["type"] = "object"

    # Handle object types
    if schema.get("type") == "object":
        # Add properties
        for prop, prop_schema in schema.get("properties", {}).items():
            child_ptr = f"{base_pointer}/{prop}"
            paths[child_ptr] = prop_schema
            # Recursively process nested objects
            if prop_schema.get("type") == "object":
                paths.update(gather_paths(prop_schema, child_ptr))
            # Handle array items
            elif prop_schema.get("type") == "array" and "items" in prop_schema:
                array_path = f"{child_ptr}/-"  # '-' represents the end of an array
                paths[array_path] = prop_schema["items"]
                # Process array item indexing
                paths.update(gather_paths(prop_schema["items"], f"{child_ptr}/0"))

        # Handle additionalProperties if present
        if schema.get("additionalProperties"):
            add_props = schema["additionalProperties"]
            if isinstance(add_props, dict):
                wild_path = f"{base_pointer}/*"
                paths[wild_path] = add_props

    # Handle array types
    elif schema.get("type") == "array" and "items" in schema:
        array_path = f"{base_pointer}/-"
        paths[array_path] = schema["items"]
        # Process array item indexing
        paths.update(gather_paths(schema["items"], f"{base_pointer}/0"))

    return paths


def op_schema(op: str, pointer: str, value_schema: dict | None) -> dict:
    """
    Build a schema fragment for one operation / one pointer.
    """
    # Ensure value_schema always has a type
    if value_schema and "type" not in value_schema:
        value_schema = {"type": "object", **value_schema}

    # Base schema for the operation
    base = {
        "type": "object",
        "required": ["op", "path"],
        "properties": {"op": {"type": "string", "enum": [op]}, "path": {"type": "string", "const": pointer}},
        "additionalProperties": False,
    }

    # Add value property for operations that require it
    if op in {"add", "replace", "test"}:
        base["required"].append("value")
        if value_schema:
            base["properties"]["value"] = value_schema
        else:
            base["properties"]["value"] = {"type": "null"}

    # Add from property for operations that require it
    if op in {"move", "copy"}:
        base["required"].append("from")
        base["properties"]["from"] = {"type": "string"}

    return base


def build_patch_schema(model_schema: dict) -> dict:
    """
    Build a JSON Schema that validates JSON Patch operations for the given model schema.
    """
    # Gather all paths in the schema
    paths = gather_paths(model_schema)

    # Generate operation schemas for each path
    variants = []
    for ptr, subschema in paths.items():
        # Handle empty or invalid schemas
        if not subschema or not isinstance(subschema, dict):
            subschema = {"type": "object"}

        # Ensure the subschema has a type
        if "type" not in subschema:
            # Try to infer type from other properties
            if "properties" in subschema:
                subschema["type"] = "object"
            elif "items" in subschema:
                subschema["type"] = "array"
            else:
                # Default to string if can't determine
                subschema["type"] = "string"

        # Generate variants for different operations
        variants.append(op_schema("add", ptr, subschema))
        variants.append(op_schema("replace", ptr, subschema))
        variants.append(op_schema("remove", ptr, None))
        variants.append(op_schema("test", ptr, subschema))

        # For move and copy operations, we'd need to do cross-path validation
        # For simplicity, we'll skip detailed validation of those here

    # Final patch schema - wrap the array in an object to work with OpenAI's API
    patch_schema = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "properties": {"operations": {"type": "array", "items": {"anyOf": variants}}},
        "required": ["operations"],
    }

    return patch_schema


def MakePatchSchema(model_class: Type[BaseModel]) -> Type[BaseModel]:
    """
    Create a Pydantic model that represents valid JSON Patch operations for the given model.

    Args:
        model_class: The Pydantic model to create a patch schema for

    Returns:
        A new Pydantic model class that validates JSON Patch operations
    """
    # Get the JSON schema for the model
    model_schema = model_class.model_json_schema()

    # Build the patch schema
    patch_schema = build_patch_schema(model_schema)

    # Create a Pydantic model for a single patch operation
    class PatchOperation(BaseModel):
        op: PatchOp
        path: str
        value: Optional[Any] = None
        from_: Optional[str] = Field(None, alias="from")

        model_config = {"populate_by_name": True, "extra": "forbid"}

    # Create a model for the patch (array of operations wrapped in an object)
    class JsonPatchDocument(BaseModel):
        operations: List[PatchOperation] = Field(default_factory=list)

        model_config = {
            "populate_by_name": True,
            "extra": "forbid",
            "json_schema_extra": lambda schema: schema.update(patch_schema),
        }

        def __iter__(self):
            return iter(self.operations)

        def __getitem__(self, item):
            return self.operations[item]

        @classmethod
        def parse_obj(cls, obj):
            """Parse a JSON object into a patch document."""
            if isinstance(obj, (str, bytes)):
                obj = json.loads(obj)

            # Expect an object with an "operations" key
            if isinstance(obj, dict) and "operations" in obj:
                return cls(operations=obj["operations"])
            else:
                raise ValueError(
                    "Input must be an object with an 'operations' field containing an array of patch operations"
                )

        def model_json_schema(cls, *args, **kwargs):
            """Return the JSON Schema as a dict."""
            return patch_schema

        def apply(self, target: Union[str, dict, BaseModel]) -> dict:
            """
            Apply the patch to a target, which can be a JSON string, dict, or Pydantic model.

            Args:
                target: The target to apply the patch to

            Returns:
                The patched data as a dict
            """
            # Convert the target to a dict
            if isinstance(target, str):
                target_dict = json.loads(target)
            elif isinstance(target, BaseModel):
                target_dict = target.model_dump(exclude_unset=True)
            else:
                target_dict = dict(target)

            # Apply operations in sequence
            for op in self.operations:
                path_parts = op.path.strip("/").split("/") if op.path != "/" else []

                if op.op == PatchOp.REPLACE:
                    target_dict = apply_replace(target_dict, path_parts, op.value)
                elif op.op == PatchOp.ADD:
                    target_dict = apply_add(target_dict, path_parts, op.value)
                elif op.op == PatchOp.REMOVE:
                    target_dict = apply_remove(target_dict, path_parts)
                elif op.op == PatchOp.MOVE:
                    from_parts = op.from_.strip("/").split("/") if op.from_ else []
                    target_dict = apply_move(target_dict, from_parts, path_parts)
                elif op.op == PatchOp.COPY:
                    from_parts = op.from_.strip("/").split("/") if op.from_ else []
                    target_dict = apply_copy(target_dict, from_parts, path_parts)
                elif op.op == PatchOp.TEST:
                    target_dict = apply_test(target_dict, path_parts, op.value)

            return target_dict

    # Update the model name to reflect the original model
    patch_model_name = f"{model_class.__name__}PatchSchema"
    JsonPatchDocument.__name__ = patch_model_name

    # Override the model_json_schema method to return our custom schema
    JsonPatchDocument.model_json_schema = classmethod(lambda cls, *args, **kwargs: patch_schema)

    return JsonPatchDocument


def apply_replace(target: dict, path_parts: List[str], value: Any) -> dict:
    """Apply a 'replace' operation."""
    target = target.copy()  # Create a copy to avoid modifying the original

    if not path_parts:
        # Replace entire document
        return value

    current = target
    for i, part in enumerate(path_parts[:-1]):
        if part.isdigit() and isinstance(current, list):
            part = int(part)
        if part not in current and i < len(path_parts) - 1:
            raise ValueError(f"Path not found: {'/'.join(path_parts[:i+1])}")
        current = current[part]

    last_part = path_parts[-1]
    if last_part.isdigit() and isinstance(current, list):
        last_part = int(last_part)
    if last_part not in current and not (isinstance(current, list) and last_part == "-"):
        raise ValueError(f"Path not found: {'/'.join(path_parts)}")

    current[last_part] = value
    return target


def apply_add(target: dict, path_parts: List[str], value: Any) -> dict:
    """Apply an 'add' operation."""
    target = target.copy()

    if not path_parts:
        # Replace entire document
        return value

    current = target
    for i, part in enumerate(path_parts[:-1]):
        if part.isdigit() and isinstance(current, list):
            part = int(part)
        if part not in current and i < len(path_parts) - 1:
            if path_parts[i + 1].isdigit() or path_parts[i + 1] == "-":
                current[part] = []
            else:
                current[part] = {}
        current = current[part]

    last_part = path_parts[-1]
    if last_part == "-" and isinstance(current, list):
        current.append(value)
    elif last_part.isdigit() and isinstance(current, list):
        idx = int(last_part)
        if idx > len(current):
            raise ValueError(f"Index out of bounds: {idx}")
        current.insert(idx, value)
    else:
        current[last_part] = value

    return target


def apply_remove(target: dict, path_parts: List[str]) -> dict:
    """Apply a 'remove' operation."""
    target = target.copy()

    if not path_parts:
        raise ValueError("Cannot remove root document")

    current = target
    for i, part in enumerate(path_parts[:-1]):
        if part.isdigit() and isinstance(current, list):
            part = int(part)
        if part not in current:
            raise ValueError(f"Path not found: {'/'.join(path_parts[:i+1])}")
        current = current[part]

    last_part = path_parts[-1]
    if last_part.isdigit() and isinstance(current, list):
        idx = int(last_part)
        if idx >= len(current):
            raise ValueError(f"Index out of bounds: {idx}")
        del current[idx]
    elif last_part not in current:
        raise ValueError(f"Path not found: {'/'.join(path_parts)}")
    else:
        del current[last_part]

    return target


def apply_move(target: dict, from_parts: List[str], path_parts: List[str]) -> dict:
    """Apply a 'move' operation."""
    target = target.copy()

    # Get the value at the 'from' location
    value = get_value(target, from_parts)

    # Remove the value from the 'from' location
    target = apply_remove(target, from_parts)

    # Add the value at the 'path' location
    return apply_add(target, path_parts, value)


def apply_copy(target: dict, from_parts: List[str], path_parts: List[str]) -> dict:
    """Apply a 'copy' operation."""
    target = target.copy()

    # Get the value at the 'from' location
    value = get_value(target, from_parts)

    # Add the value at the 'path' location
    return apply_add(target, path_parts, value)


def apply_test(target: dict, path_parts: List[str], value: Any) -> dict:
    """Apply a 'test' operation."""
    current_value = get_value(target, path_parts)

    if current_value != value:
        raise ValueError(f"Test failed: {'/'.join(path_parts)} does not match expected value")

    return target


def get_value(target: dict, path_parts: List[str]) -> Any:
    """Get a value at a specific path."""
    if not path_parts:
        return target

    current = target
    for i, part in enumerate(path_parts):
        if part.isdigit() and isinstance(current, list):
            part = int(part)
        if part not in current:
            raise ValueError(f"Path not found: {'/'.join(path_parts[:i+1])}")
        current = current[part]

    return current
