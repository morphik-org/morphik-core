from typing import Dict, List, Optional, Tuple, Set, Any
import logging
from enum import Enum
import re
import uuid
from pydantic import BaseModel, Field
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

class RetrievalType(str, Enum):
    """Types of retrieval methods available"""
    SEMANTIC_SEARCH = "semantic_search"
    KNOWLEDGE_GRAPH = "knowledge_graph"
    MULTIMODAL = "multimodal"
    WEB_SEARCH = "web_search"
    STRUCTURED_DATA = "structured_data"
    CODE_SEARCH = "code_search"


class SubQueryState(str, Enum):
    """Possible states for a subquery"""
    UNANSWERED = "unanswered"
    PROCESSING = "processing"
    ANSWERED = "answered"
    FAILED = "failed"


class SubQuery(BaseModel):
    """Represents a subquery generated from the original query"""
    text: str
    retrieval_types: List[RetrievalType] = Field(default_factory=list)
    state: SubQueryState = SubQueryState.UNANSWERED
    answer: Optional[str] = None
    confidence: float = 0.0  # 0.0-1.0 confidence in the answer
    parent_id: Optional[str] = None
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))  # Unique identifier for the subquery
    child_subqueries: List["SubQuery"] = Field(default_factory=list)
    retrieved_context: Dict[str, Any] = Field(default_factory=dict)
    depth: int = 0  # Depth in the subquery tree (0 for root queries)
    
    def transition_to_answered(self, answer: str, confidence: float):
        """Transition the subquery to the answered state"""
        self.state = SubQueryState.ANSWERED
        self.answer = answer
        self.confidence = confidence
    
    def transition_to_processing(self):
        """Transition the subquery to the processing state"""
        self.state = SubQueryState.PROCESSING
    
    def transition_to_failed(self):
        """Transition the subquery to the failed state"""
        self.state = SubQueryState.FAILED
    
    def add_child_subquery(self, subquery: "SubQuery"):
        """Add a child subquery"""
        subquery.parent_id = self.id
        subquery.depth = self.depth + 1
        self.child_subqueries.append(subquery)
    
    def is_answerable(self) -> bool:
        """Check if this subquery is answerable"""
        if self.state == SubQueryState.ANSWERED:
            return True
        
        # If it has child subqueries, check if all of them are answerable
        if self.child_subqueries:
            return all(sq.is_answerable() for sq in self.child_subqueries)
        
        return False
    
    def get_answer(self) -> Tuple[str, float]:
        """Get the answer and confidence for this subquery"""
        if self.state == SubQueryState.ANSWERED:
            return self.answer or "", self.confidence
        
        # If it has child subqueries, combine their answers
        if self.child_subqueries and all(sq.is_answerable() for sq in self.child_subqueries):
            child_answers = [sq.get_answer()[0] for sq in self.child_subqueries]
            child_confidences = [sq.get_answer()[1] for sq in self.child_subqueries]
            
            # Calculate average confidence
            avg_confidence = sum(child_confidences) / len(child_confidences) if child_confidences else 0.0
            
            # The answer is a combination of child answers (in a real implementation, this would use an LLM)
            combined_answer = "\n\n".join(child_answers)
            
            return combined_answer, avg_confidence
        
        return "", 0.0
    
    def add_retrieved_context(self, retrieval_type: RetrievalType, context: Any):
        """Add retrieved context for this subquery"""
        if retrieval_type.value not in self.retrieved_context:
            self.retrieved_context[retrieval_type.value] = []
            
        if isinstance(context, list):
            self.retrieved_context[retrieval_type.value].extend(context)
        else:
            self.retrieved_context[retrieval_type.value].append(context)


class QueryPlan(BaseModel):
    """Represents a plan for answering a user query"""
    original_query: str
    root_subqueries: List[SubQuery] = Field(default_factory=list)
    is_answerable: bool = False
    search_iterations: int = 0
    max_iterations: int = 3  # Maximum number of search iterations before giving up
    final_answer: Optional[str] = None
    max_recursion_depth: int = 2  # Maximum depth for recursive query expansion
    
    def add_root_subquery(self, subquery: SubQuery):
        """Add a root-level subquery"""
        self.root_subqueries.append(subquery)
    
    def check_answerability(self) -> bool:
        """Check if all subqueries are answerable"""
        self.is_answerable = all(sq.is_answerable() for sq in self.root_subqueries)
        return self.is_answerable
    
    def get_all_subqueries(self) -> List[SubQuery]:
        """Get all subqueries in the plan (root and nested)"""
        result = []
        
        def traverse(subqueries):
            for sq in subqueries:
                result.append(sq)
                traverse(sq.child_subqueries)
        
        traverse(self.root_subqueries)
        return result
    
    def get_unanswered_subqueries(self) -> List[SubQuery]:
        """Get all unanswered subqueries that don't have child subqueries"""
        result = []
        
        def traverse(subqueries):
            for sq in subqueries:
                if sq.state == SubQueryState.UNANSWERED and not sq.child_subqueries:
                    result.append(sq)
                traverse(sq.child_subqueries)
        
        traverse(self.root_subqueries)
        return result


class QueryPlanner:
    """
    Plans and orchestrates the query resolution process.
    Breaks down user queries, determines retrieval methods, and evaluates answerability.
    """
    
    def __init__(self, completion_model=None, embedding_model=None):
        """
        Initialize the query planner.
        
        Args:
            completion_model: LLM for query planning and answer generation
            embedding_model: Embedding model for semantic search
        """
        self.completion_model = completion_model
        self.embedding_model = embedding_model
    
    def create_query_plan(self, query: str) -> QueryPlan:
        """
        Create a query plan for a given user query.
        Breaks down the query and determines required retrieval methods.
        
        Args:
            query: The original user query
            
        Returns:
            QueryPlan: The structured query plan with subqueries
        """
        subqueries = self._generate_subqueries(query)
        retrieval_types = self._determine_retrieval_types(query, subqueries)
        
        query_plan = QueryPlan(
            original_query=query,
            root_subqueries=[
                SubQuery(text=sq, retrieval_types=rt, depth=0)
                for sq, rt in zip(subqueries, retrieval_types)
            ]
        )
        
        return query_plan
    
    def _generate_subqueries(self, query: str) -> List[str]:
        """
        Break down a complex query into simpler subqueries.
        
        Args:
            query: The original user query
            
        Returns:
            List[str]: List of subqueries
        """
        # This would typically be implemented using an LLM
        # For now, using a simplistic approach for demonstration
        
        # If we have a completion model, use it to generate subqueries
        if self.completion_model:
            prompt = f"""
            Break down the following query into 1-5 simpler subqueries that would help answer it completely:
            
            Query: {query}
            
            Output the subqueries as a numbered list.
            """
            response = self.completion_model.generate(prompt)
            # Extract numbered list items
            subqueries = re.findall(r'\d+\.\s+(.*)', response)
            if subqueries:
                return subqueries
        
        # Fallback: Return the original query as the only subquery
        return [query]
    
    def _determine_retrieval_types(self, query: str, subqueries: List[str]) -> List[List[RetrievalType]]:
        """
        Determine which retrieval types to use for each subquery.
        
        Args:
            query: The original user query
            subqueries: List of subqueries
            
        Returns:
            List[List[RetrievalType]]: List of retrieval types for each subquery
        """
        retrieval_types = []
        
        # This would typically be implemented using an LLM
        # For now, using simple heuristics for demonstration
        
        for sq in subqueries:
            types = []
            
            # Always include semantic search as a baseline
            types.append(RetrievalType.SEMANTIC_SEARCH)
            
            # Simple keyword matching for demonstration
            sq_lower = sq.lower()
            
            if any(term in sq_lower for term in ["image", "picture", "photo", "video", "visual"]):
                types.append(RetrievalType.MULTIMODAL)
                
            if any(term in sq_lower for term in ["relation", "connected", "link", "graph"]):
                types.append(RetrievalType.KNOWLEDGE_GRAPH)
                
            if any(term in sq_lower for term in ["latest", "recent", "news", "current", "update"]):
                types.append(RetrievalType.WEB_SEARCH)
                
            if any(term in sq_lower for term in ["table", "database", "sql", "record", "data"]):
                types.append(RetrievalType.STRUCTURED_DATA)
                
            if any(term in sq_lower for term in ["code", "function", "class", "implementation"]):
                types.append(RetrievalType.CODE_SEARCH)
            
            retrieval_types.append(types)
            
        return retrieval_types
    
    def evaluate_answerability(self, subquery: SubQuery) -> bool:
        """
        Evaluate whether a specific subquery is answerable with its current context.
        
        Args:
            subquery: The subquery to evaluate
            
        Returns:
            bool: True if the subquery is answerable, False otherwise
        """
        if not subquery.retrieved_context:
            return False
            
        if self.completion_model:
            # Format the context
            formatted_context = self._format_retrieved_context(subquery.retrieved_context)
            
            # Ask the LLM if the subquery can be answered with the context
            prompt = f"""
            Given the following subquery and retrieved context, determine if the subquery can be answered.
            
            Subquery: {subquery.text}
            
            Retrieved Context:
            {formatted_context}
            
            Can the subquery be answered with this context? (Yes or No)
            If Yes, what is the confidence level (0.0-1.0)?
            """
            
            response = self.completion_model.generate(prompt)
            
            if "yes" in response.lower():
                # Extract confidence if available
                confidence_match = re.search(r'(\d+\.\d+)', response)
                confidence = float(confidence_match.group(1)) if confidence_match else 0.7
                
                # Generate the answer
                answer_prompt = f"""
                Given the following subquery and retrieved context, provide a precise answer.
                
                Subquery: {subquery.text}
                
                Retrieved Context:
                {formatted_context}
                
                Answer:
                """
                
                answer = self.completion_model.generate(answer_prompt)
                
                # Transition to answered state
                subquery.transition_to_answered(answer, confidence)
                return True
        
        return False
    
    def recursively_expand_subquery(self, subquery: SubQuery) -> bool:
        """
        Recursively expand a subquery that couldn't be answered directly.
        
        Args:
            subquery: The subquery to expand
            
        Returns:
            bool: True if the expansion was successful, False otherwise
        """
        # Check if we've reached the maximum recursion depth
        if subquery.depth >= self.max_recursion_depth:
            logger.warning(f"Maximum recursion depth reached for subquery: {subquery.text}")
            return False
            
        # Generate child subqueries
        child_texts = self._generate_subqueries(subquery.text)
        child_retrieval_types = self._determine_retrieval_types(subquery.text, child_texts)
        
        # Create and add child subqueries
        for text, retrieval_types in zip(child_texts, child_retrieval_types):
            child = SubQuery(
                text=text,
                retrieval_types=retrieval_types,
                depth=subquery.depth + 1
            )
            subquery.add_child_subquery(child)
            
        return len(subquery.child_subqueries) > 0
    
    def _format_retrieved_context(self, context_dict: Dict[str, List[str]]) -> str:
        """
        Format retrieved context into a string.
        
        Args:
            context_dict: Dictionary mapping retrieval types to lists of context strings
            
        Returns:
            str: Formatted context string
        """
        parts = []
        for rt, contexts in context_dict.items():
            parts.append(f"--- {rt.upper()} RESULTS ---")
            parts.append("\n\n".join(str(c) for c in contexts))
        
        return "\n\n".join(parts)
    
    def process_subquery_tree(self, subquery: SubQuery, retrieval_functions: Dict[RetrievalType, callable]) -> bool:
        """
        Process a subquery and its children recursively.
        
        Args:
            subquery: The subquery to process
            retrieval_functions: Dictionary mapping retrieval types to retrieval functions
            
        Returns:
            bool: True if the subquery is answerable, False otherwise
        """
        # If already answered, return True
        if subquery.state == SubQueryState.ANSWERED:
            return True
            
        # Retrieve information for this subquery
        subquery.transition_to_processing()
        
        for rt in subquery.retrieval_types:
            # Skip if we don't have a function for this retrieval type
            if rt not in retrieval_functions:
                logger.warning(f"No retrieval function for {rt}")
                continue
                
            # Call the retrieval function
            retrieval_func = retrieval_functions[rt]
            context = retrieval_func(subquery.text)
            
            # Store the retrieved context
            subquery.add_retrieved_context(rt, context)
        
        # Try to answer with the retrieved context
        if self.evaluate_answerability(subquery):
            return True
            
        # If not answerable and not at max depth, recursively expand
        if subquery.depth < self.max_recursion_depth:
            # Create child subqueries
            if self.recursively_expand_subquery(subquery):
                # Process each child subquery
                all_children_answerable = True
                for child in subquery.child_subqueries:
                    if not self.process_subquery_tree(child, retrieval_functions):
                        all_children_answerable = False
                
                # If all children are answerable, synthesize an answer for this subquery
                if all_children_answerable:
                    return self.synthesize_answer_from_children(subquery)
        
        # If we get here, the subquery is not answerable
        subquery.transition_to_failed()
        return False
    
    def synthesize_answer_from_children(self, subquery: SubQuery) -> bool:
        """
        Synthesize an answer for a subquery from its children's answers.
        
        Args:
            subquery: The subquery to synthesize an answer for
            
        Returns:
            bool: True if synthesis was successful, False otherwise
        """
        if not subquery.child_subqueries or not all(child.is_answerable() for child in subquery.child_subqueries):
            return False
            
        if self.completion_model:
            # Compile all child answers
            child_answers = []
            for i, child in enumerate(subquery.child_subqueries):
                answer, confidence = child.get_answer()
                child_answers.append(f"Subquery {i+1}: {child.text}\nAnswer: {answer}\nConfidence: {confidence}")
            
            # Ask the LLM to synthesize an answer
            prompt = f"""
            Given the following original subquery and answers to its component subqueries, 
            provide a comprehensive answer to the original subquery.
            
            Original Subquery: {subquery.text}
            
            Component Answers:
            {"".join(child_answers)}
            
            Comprehensive Answer to Original Subquery:
            """
            
            answer = self.completion_model.generate(prompt)
            
            # Calculate average confidence
            avg_confidence = sum(child.get_answer()[1] for child in subquery.child_subqueries) / len(subquery.child_subqueries)
            
            # Update the subquery
            subquery.transition_to_answered(answer, avg_confidence)
            return True
        
        return False
    
    def generate_final_answer(self, query_plan: QueryPlan) -> str:
        """
        Generate a final answer based on the query plan and subquery answers.
        
        Args:
            query_plan: The completed query plan with answers
            
        Returns:
            str: The final answer to the original query
        """
        if not query_plan.is_answerable:
            return "I'm sorry, but I don't have enough information to answer your question."
        
        if self.completion_model:
            # Compile all root subquery answers
            subquery_answers = []
            for i, sq in enumerate(query_plan.root_subqueries):
                answer, confidence = sq.get_answer()
                subquery_answers.append(f"Subquery {i+1}: {sq.text}\nAnswer: {answer}\nConfidence: {confidence}")
            
            # Ask the LLM to synthesize a final answer
            prompt = f"""
            Given the following original query and answers to its subqueries, 
            provide a comprehensive answer to the original query.
            
            Original Query: {query_plan.original_query}
            
            Subquery Answers:
            {"".join(subquery_answers)}
            
            Comprehensive Answer to Original Query:
            """
            
            final_answer = self.completion_model.generate(prompt)
            query_plan.final_answer = final_answer
            return final_answer
        
        # Fallback if no completion model
        answers = []
        for sq in query_plan.root_subqueries:
            answer, _ = sq.get_answer()
            if answer:
                answers.append(answer)
                
        if answers:
            return "\n\n".join(answers)
        else:
            return "I'm sorry, but I don't have enough information to answer your question."
    
    def execute_query_plan(self, query: str, retrieval_functions: Dict[RetrievalType, callable]) -> str:
        """
        Execute a query plan from start to finish, including iterative search.
        
        Args:
            query: The original user query
            retrieval_functions: Dictionary mapping retrieval types to retrieval functions
            
        Returns:
            str: The final answer to the query
        """
        # Create the initial query plan
        query_plan = self.create_query_plan(query)
        
        # Process each root subquery
        for subquery in query_plan.root_subqueries:
            self.process_subquery_tree(subquery, retrieval_functions)
            
        # Check if the plan is answerable
        query_plan.check_answerability()
        
        # Generate the final answer
        return self.generate_final_answer(query_plan)
    
    def _refine_retrieval_strategy(self, subquery: SubQuery) -> None:
        """
        Refine the retrieval strategy for an unanswered subquery.
        
        Args:
            subquery: The subquery to refine retrieval strategy for
        """
        # Add web search as a last resort
        if subquery.state == SubQueryState.UNANSWERED:
            if RetrievalType.WEB_SEARCH not in subquery.retrieval_types:
                subquery.retrieval_types.append(RetrievalType.WEB_SEARCH)
