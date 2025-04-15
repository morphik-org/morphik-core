from typing import List
import logging
import json
from pydantic import BaseModel
import instructor
import litellm
from core.models.request import SmartQueryRequest, FilterOperation, SortOperation
from core.models.documents import Document
from core.config import get_settings


logger = logging.getLogger(__name__)
settings = get_settings()


class DocumentMetadataBasedResponse(BaseModel):
    """Structured response for filter evaluation"""

    explanation: str
    result: int  # 1 for match, 0 for no match, -1 for undetermined


class FullDocumentBasedResponse(BaseModel):
    """Structured response for content filter evaluation"""

    explanation: str
    result: bool


class QueryExecutor:
    """Executes smart queries by applying filters and sorts to document chunks."""

    def __init__(self):
        pass

    async def execute_query(
        self, request: SmartQueryRequest, documents: List[Document]
    ) -> List[str]:
        """
        Execute a smart query on documents.

        Args:
            request: The smart query request including filter, sorts, and limits
            documents: List of documents to process

        Returns:
            List of documents after applying filter, sorts, and limits
        """
        # Apply filter if provided
        filtered_documents = (
            await self._apply_filter(documents, request.filter) if request.filter else documents
        )

        # Apply sorts
        sorted_documents = await self._apply_sorts(filtered_documents, request.sort_by)

        # Apply limit if specified
        if request.limit is not None and request.limit > 0:
            sorted_documents = sorted_documents[: request.limit]

        return [sorted_document.external_id for sorted_document in sorted_documents]

    def _prepare_metadata_summary(self, document: Document) -> dict:
        """
        Prepare a metadata summary for a document, excluding content.

        Args:
            document: Document to extract metadata from

        Returns:
            Dictionary containing document metadata
        """
        return {
            "filename": document.filename,
            "content_type": document.content_type,
            "metadata": document.metadata,
            "system_metadata": {
                k: v for k, v in document.system_metadata.items() if k != "content"
            },
            "additional_metadata": document.additional_metadata,
        }

    async def _get_llm_response(
        self, messages: List[dict], response_model: type, log_prefix: str = "", log_suffix: str = ""
    ) -> any:
        """
        Helper function to get structured responses from LLM using instructor and litellm.

        Args:
            messages: List of message dictionaries (system, user, etc.)
            response_model: Pydantic model for response structure
            log_prefix: Prefix for log messages
            log_suffix: Suffix for log messages

        Returns:
            Structured response from LLM

        Raises:
            ValueError: If model configuration is not found
        """
        # Get the model configuration
        model_config = settings.REGISTERED_MODELS.get(settings.SMART_QUERY_MODEL, {})
        if not model_config:
            raise ValueError(
                f"Model '{settings.SMART_QUERY_MODEL}' not found in registered_models configuration"
            )

        # Use instructor with litellm for structured responses
        client = instructor.from_litellm(litellm.acompletion, mode=instructor.Mode.JSON)

        try:
            # Extract model name and prepare parameters
            model = model_config.get("model_name")
            model_kwargs = {k: v for k, v in model_config.items() if k != "model_name"}

            # Get structured response
            response = await client.chat.completions.create(
                model=model,
                messages=messages,
                response_model=response_model,
                **model_kwargs,
            )

            if log_prefix:
                logger.debug(f"{log_prefix}: {response.model_dump_json()} {log_suffix}")

            return response

        except Exception as e:
            logger.error(f"Error in LLM request: {e}")
            raise

    async def _evaluate_metadata_filter(
        self, document: Document, filter_op: FilterOperation
    ) -> int:
        """
        Evaluate filter on document metadata using LLM with instructor.

        Returns:
            1 for True (include document)
            0 for False (exclude document)
            -1 for Undetermined (needs further content analysis)
        """
        # Construct metadata summary
        metadata_summary = self._prepare_metadata_summary(document)

        # Create prompt for LLM to evaluate metadata against filter predicate
        prompt = f"""
        You are evaluating if a document matches a predicate based on its metadata.
        
        Predicate: {filter_op.predicate}
        
        Document metadata: {json.dumps(metadata_summary, indent=2)}
        
        Determine if this document matches the predicate.
        
        Return EXACTLY ONE of these values for result:
        - 1 if you're confident the document matches the predicate
        - 0 if you're confident the document does NOT match the predicate
        - -1 if you CANNOT determine from metadata alone and need to examine the document content.
        
        Limit the explanation to 1 sentence.
        """

        system_message = {
            "role": "system",
            "content": "You are a metadata evaluation assistant. Evaluate if documents match predicate based on metadata.",
        }

        user_message = {"role": "user", "content": prompt}
        messages = [system_message, user_message]

        try:
            response = await self._get_llm_response(
                messages=messages,
                response_model=DocumentMetadataBasedResponse,
                log_prefix="Metadata filter evaluation response",
                log_suffix=f"for {document.external_id}",
            )
            return response.result
        except Exception:
            return -1  # Default to undetermined if there's an error

    async def _evaluate_content_filter(
        self, document: Document, filter_op: FilterOperation
    ) -> bool:
        """
        Evaluate filter on document content using LLM with instructor.

        Returns:
            True if document matches filter, False otherwise
        """
        # Create prompt for LLM to evaluate content against filter
        prompt = f"""
        You are evaluating if a document matches a filter based on its content and additional properties.
        
        Filter: {filter_op.predicate}
        
        <DOCUMENT>
        {document.model_dump_json(indent=2)}
        <DOCUMENT>
        
        Does this document match the filter criteria?
        Return EXACTLY ONE of these values for result:
        - true if the document matches the filter
        - false if the document does NOT match the filter
        
        Limit the explanation to 1 sentence.
        """

        system_message = {
            "role": "system",
            "content": "You are a content evaluation assistant. Evaluate if documents match filters based on full document.",
        }

        user_message = {"role": "user", "content": prompt}
        messages = [system_message, user_message]

        try:
            response = await self._get_llm_response(
                messages=messages,
                response_model=FullDocumentBasedResponse,
                log_prefix="Content filter evaluation response",
                log_suffix=f"for {document.external_id}",
            )
            return response.result
        except Exception as e:
            logger.error(f"Error evaluating content filter: {e}")
            return False  # Default to excluding the document if there's an error

    async def _apply_filter(
        self, documents: List[Document], filter_op: FilterOperation
    ) -> List[Document]:
        """
        Apply two-step filter to documents.

        Step 1: Evaluate metadata only
        Step 2: If needed, evaluate content

        Args:
            documents: The documents to filter
            filter_op: Filter operation with natural language predicate

        Returns:
            Filtered list of documents
        """
        if not filter_op or not filter_op.predicate:
            return documents

        filtered_documents = []

        for document in documents:
            # Step 1: Metadata evaluation
            metadata_result = await self._evaluate_metadata_filter(document, filter_op)

            if metadata_result == 0:  # Definite no from metadata
                continue

            elif metadata_result == -1:  # Undetermined from metadata
                # Step 2: Content evaluation
                content_result = await self._evaluate_content_filter(document, filter_op)

                if not content_result:
                    continue

            # If metadata_result is 1 or content_result is True, include the document
            filtered_documents.append(document)

        return filtered_documents

    async def _apply_sorts(
        self, documents: List[Document], sorts: List[SortOperation]
    ) -> List[Document]:
        """
        Apply natural language sorts to documents.

        Args:
            documents: The documents to sort
            sorts: List of sort operations with natural language comparators

        Returns:
            Sorted list of documents
        """
        if not sorts:
            # Default no sorting
            return documents

        # Apply each sort in sequence
        sorted_documents = documents.copy()

        for sort_op in sorts:
            sorted_documents = await self.sort(sorted_documents, sort_op)
        return sorted_documents

    async def sort(self, documents: List[Document], sort_op: SortOperation) -> List[Document]:
        """
        Sort documents using merge sort algorithm, guaranteeing O(n log n) comparisons.

        Args:
            documents: List of documents to sort
            sort_op: Sort operation with natural language comparator

        Returns:
            Sorted list of documents
        """
        if len(documents) <= 1:
            return documents

        # Implement merge sort with async comparison
        async def merge_sort(docs):
            if len(docs) <= 1:
                return docs

            # Divide
            mid = len(docs) // 2
            left = await merge_sort(docs[:mid])
            right = await merge_sort(docs[mid:])

            # Conquer: merge the sorted halves
            return await merge(left, right)

        async def merge(left, right):
            result = []
            i = j = 0

            # Merge the two sorted lists
            while i < len(left) and j < len(right):
                # Use compare_documents to determine order
                # Returns True if left[i] should be ahead of right[j]
                if await self.compare_documents(left[i], right[j], sort_op):
                    result.append(left[i])
                    i += 1
                else:
                    result.append(right[j])
                    j += 1

            # Add remaining elements
            result.extend(left[i:])
            result.extend(right[j:])
            return result

        return await merge_sort(documents)

    async def compare_documents(
        self, doc1: Document, doc2: Document, sort_op: SortOperation
    ) -> bool:
        """
        Compare two documents using LLM to determine which should come first based on sort_op.

        Args:
            doc1: First document to compare
            doc2: Second document to compare
            sort_op: Sort operation with natural language comparator

        Returns:
            True if doc1 should be ahead of doc2 in ASC order, False otherwise
            Order is reversed if sort_op.order is DESC
        """
        # First try metadata-only comparison
        metadata_result = await self._compare_documents_metadata(doc1, doc2, sort_op)
        match metadata_result:
            case 1:  # doc1 should be ahead of doc2
                return True if sort_op.order == "ASC" else False
            case 0:  # doc2 should be ahead of doc1
                return False if sort_op.order == "ASC" else True
            case -1:  # Cannot determine from metadata alone
                result = await self._compare_documents_content(doc1, doc2, sort_op)
                return result if sort_op.order == "ASC" else not result

    async def _compare_documents_metadata(
        self, doc1: Document, doc2: Document, sort_op: SortOperation
    ) -> int:
        """
        Compare documents using metadata only.

        Returns:
            1: doc1 should be ahead of doc2
            0: doc2 should be ahead of doc1
            -1: Cannot determine from metadata alone
        """
        # Prepare metadata summaries for both documents
        doc1_metadata = self._prepare_metadata_summary(doc1)
        doc2_metadata = self._prepare_metadata_summary(doc2)

        # Create prompt for comparison based on metadata
        prompt = f"""
        You are evaluating which document should come first in a sorted order.
        
        Sort criterion: {sort_op.comparator}
        
        Document A metadata: {json.dumps(doc1_metadata, indent=2)}
        Document B metadata: {json.dumps(doc2_metadata, indent=2)}
        
        Determine which document should come first based ONLY on the metadata.
        Assume ASCENDING order (the criterion goes from low to high values).
        
        Return EXACTLY ONE of these values for result:
        - 1 if you're confident Document A should come before Document B
        - 0 if you're confident Document B should come before Document A
        - -1 if you CANNOT determine from metadata alone and need to examine the document content
        
        Limit the explanation to 1 sentence.
        """

        system_message = {
            "role": "system",
            "content": "You are a document sorting assistant. Compare documents based on sort criteria.",
        }

        user_message = {"role": "user", "content": prompt}
        messages = [system_message, user_message]

        try:
            response = await self._get_llm_response(
                messages=messages,
                response_model=DocumentMetadataBasedResponse,
                log_prefix="Metadata comparison response",
                log_suffix=f"for {sort_op.comparator}",
            )
            return response.result
        except Exception:
            return -1  # Default to undetermined on error

    async def _compare_documents_content(
        self, doc1: Document, doc2: Document, sort_op: SortOperation
    ) -> bool:
        """
        Compare documents using full content when metadata comparison is inconclusive.

        Returns:
            True if doc1 should be ahead of doc2 in ASC order, False otherwise
        """
        # Create prompt that includes document content
        prompt = f"""
        You are evaluating which document should come first in a sorted order.
        
        Sort criterion: {sort_op.comparator}
        
        <DOCUMENT A>
        {doc1.model_dump_json(indent=2)}
        </DOCUMENT A>
        
        <DOCUMENT B>
        {doc2.model_dump_json(indent=2)}
        </DOCUMENT B>
        
        Determine which document should come first based on the sort criterion.
        Assume ASCENDING order (the criterion goes from low to high values).
        
        Result should be TRUE if Document A should come before Document B, 
        and FALSE if Document B should come before Document A.
        
        Limit the explanation to 1 sentence.
        """

        system_message = {
            "role": "system",
            "content": "You are a document sorting assistant. Compare documents based on sort criteria.",
        }

        user_message = {"role": "user", "content": prompt}
        messages = [system_message, user_message]

        try:
            response = await self._get_llm_response(
                messages=messages,
                response_model=FullDocumentBasedResponse,
                log_prefix="Content comparison response",
                log_suffix=f"for {sort_op.comparator}",
            )
            return response.result
        except Exception as e:
            logger.error(f"Error in content comparison: {e}")
            # Default to simple filename comparison if LLM fails
            return doc1.filename < doc2.filename
