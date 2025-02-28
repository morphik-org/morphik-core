import unittest
from unittest.mock import MagicMock, patch
from core.planner.planner import QueryPlanner, QueryPlan, SubQuery, RetrievalType


class TestQueryPlanner(unittest.TestCase):
    """Tests for the QueryPlanner class"""

    def setUp(self):
        """Set up test fixtures"""
        self.mock_completion_model = MagicMock()
        self.mock_embedding_model = MagicMock()
        self.planner = QueryPlanner(
            completion_model=self.mock_completion_model,
            embedding_model=self.mock_embedding_model
        )

    def test_create_query_plan(self):
        """Test creating a query plan from a user query"""
        # Mock the subquery generation and retrieval type determination
        self.planner._generate_subqueries = MagicMock(return_value=["What is the capital of France?"])
        self.planner._determine_retrieval_types = MagicMock(return_value=[[RetrievalType.SEMANTIC_SEARCH]])
        
        # Test creating a query plan
        query = "Tell me about the capital of France"
        plan = self.planner.create_query_plan(query)
        
        # Verify the plan was created correctly
        self.assertEqual(plan.original_query, query)
        self.assertEqual(len(plan.subqueries), 1)
        self.assertEqual(plan.subqueries[0].text, "What is the capital of France?")
        self.assertEqual(plan.subqueries[0].retrieval_types, [RetrievalType.SEMANTIC_SEARCH])
        self.assertFalse(plan.is_answerable)

    def test_generate_subqueries_with_completion_model(self):
        """Test generating subqueries using the completion model"""
        # Mock the completion model response
        self.mock_completion_model.generate = MagicMock(return_value="""
        1. What is the history of Paris?
        2. What are the main attractions in Paris?
        3. What is the population of Paris?
        """)
        
        # Call the method
        query = "Tell me about Paris"
        subqueries = self.planner._generate_subqueries(query)
        
        # Verify the subqueries were generated correctly
        self.assertEqual(len(subqueries), 3)
        self.assertEqual(subqueries[0], "What is the history of Paris?")
        self.assertEqual(subqueries[1], "What are the main attractions in Paris?")
        self.assertEqual(subqueries[2], "What is the population of Paris?")

    def test_generate_subqueries_fallback(self):
        """Test generating subqueries with fallback when completion model fails"""
        # Mock the completion model to return an empty response
        self.mock_completion_model.generate = MagicMock(return_value="")
        
        # Call the method
        query = "Tell me about Paris"
        subqueries = self.planner._generate_subqueries(query)
        
        # Verify the fallback works
        self.assertEqual(len(subqueries), 1)
        self.assertEqual(subqueries[0], query)

    def test_determine_retrieval_types(self):
        """Test determining retrieval types for subqueries"""
        # Define subqueries with different characteristics
        subqueries = [
            "What is the population of Paris?",
            "Show me images of the Eiffel Tower",
            "What are the latest news about Paris?",
            "How are Paris and London connected?",
            "What is the database schema for Paris attractions?"
        ]
        
        # Call the method
        query = "Tell me about Paris"
        retrieval_types = self.planner._determine_retrieval_types(query, subqueries)
        
        # Verify the retrieval types were determined correctly
        # All should have semantic search as a baseline
        self.assertTrue(all(RetrievalType.SEMANTIC_SEARCH in types for types in retrieval_types))
        
        # Check specific types based on query content
        self.assertTrue(RetrievalType.MULTIMODAL in retrieval_types[1])  # "images" keyword
        self.assertTrue(RetrievalType.WEB_SEARCH in retrieval_types[2])  # "latest" keyword
        self.assertTrue(RetrievalType.KNOWLEDGE_GRAPH in retrieval_types[3])  # "connected" keyword
        self.assertTrue(RetrievalType.STRUCTURED_DATA in retrieval_types[4])  # "database" keyword

    def test_evaluate_answerability(self):
        """Test evaluating if a query plan is answerable"""
        # Create a query plan with two subqueries
        plan = QueryPlan(
            original_query="Tell me about Paris",
            subqueries=[
                SubQuery(text="What is the history of Paris?", retrieval_types=[RetrievalType.SEMANTIC_SEARCH]),
                SubQuery(text="What are the main attractions in Paris?", retrieval_types=[RetrievalType.SEMANTIC_SEARCH])
            ]
        )
        
        # Mock the LLM to say one subquery is answerable and one is not
        self.planner._extract_relevant_context = MagicMock(return_value="Relevant context about Paris")
        
        def mock_generate(prompt):
            if "history of Paris" in prompt:
                return "Yes, this can be answered. Confidence: 0.8"
            else:
                return "No, I don't have enough information."
                
        self.mock_completion_model.generate = mock_generate
        
        # Create mock retrieved context
        retrieved_context = {
            "semantic_search": "Paris is the capital of France. It has a rich history."
        }
        
        # Call the method
        result = self.planner.evaluate_answerability(plan, retrieved_context)
        
        # Verify the result
        self.assertFalse(result)  # Plan is not answerable because only one subquery is answered
        self.assertTrue(plan.subqueries[0].is_answered)  # First subquery should be answered
        self.assertFalse(plan.subqueries[1].is_answered)  # Second subquery should not be answered
        
    def test_execute_query_plan(self):
        """Test executing a query plan from start to finish"""
        # Mock all the internal methods
        self.planner.create_query_plan = MagicMock(return_value=QueryPlan(
            original_query="Tell me about Paris",
            subqueries=[
                SubQuery(text="What is the history of Paris?", retrieval_types=[RetrievalType.SEMANTIC_SEARCH])
            ]
        ))
        
        self.planner.evaluate_answerability = MagicMock(side_effect=lambda plan, ctx: setattr(plan, "is_answerable", True) or True)
        self.planner.generate_final_answer = MagicMock(return_value="Paris is the capital of France with a rich history.")
        
        # Create mock retrieval functions
        retrieval_functions = {
            RetrievalType.SEMANTIC_SEARCH: MagicMock(return_value="Paris is the capital of France.")
        }
        
        # Call the method
        result = self.planner.execute_query_plan("Tell me about Paris", retrieval_functions)
        
        # Verify the result
        self.assertEqual(result, "Paris is the capital of France with a rich history.")
        retrieval_functions[RetrievalType.SEMANTIC_SEARCH].assert_called_once()


if __name__ == "__main__":
    unittest.main() 