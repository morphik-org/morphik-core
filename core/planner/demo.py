import logging
import json
from typing import Dict, Any
import uuid

from core.planner.planner import QueryPlanner, QueryPlan, SubQuery, SubQueryState, RetrievalType

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MockCompletionModel:
    """
    A mock completion model for demonstration purposes.
    In a real application, this would be a connection to an LLM like GPT, Claude, etc.
    """
    
    def generate(self, prompt: str) -> str:
        """Simulate generating text from an LLM"""
        logger.info(f"MockCompletionModel received prompt: {prompt[:100]}...")
        
        # Simulated responses for different prompt types
        if "Break down the following query into" in prompt:
            # Different query breakdown based on the query content
            if "iPhone 14" in prompt:
                return """
                1. What are the main features of the iPhone 14 Pro?
                2. How does the camera of iPhone 14 Pro compare to previous models?
                3. What's the battery life of the iPhone 14 Pro?
                4. What are expert reviews saying about the iPhone 14 Pro?
                5. What are the pricing options for the iPhone 14 Pro?
                """
            elif "COVID-19" in prompt:
                return """
                1. What is COVID-19 and how does it spread?
                2. What are the main symptoms of COVID-19?
                3. How effective are vaccines against COVID-19?
                4. What are the long-term effects of COVID-19?
                """
            elif "economic impact" in prompt.lower():
                return """
                1. How did COVID-19 affect global GDP?
                2. What industries were most negatively impacted by COVID-19?
                3. What industries saw growth during the COVID-19 pandemic?
                4. What economic recovery strategies were implemented?
                """
            else:
                return """
                1. What is the definition of the topic?
                2. What are the key characteristics of the topic?
                3. What are some examples or applications of the topic?
                """
        elif "determine if the subquery can be answered" in prompt:
            # Answerability check
            if "battery life" in prompt.lower():
                return "Yes, this can be answered with the provided context. Confidence: 0.8"
            elif "camera" in prompt.lower():
                return "Yes, this can be answered. Confidence: 0.9"
            elif "pricing" in prompt.lower() and "web_search" in prompt.lower():
                return "Yes, with the web search results, I can answer this. Confidence: 0.95"
            elif "main features" in prompt.lower():
                return "No, I don't have enough information to answer this question."
            elif "economic" in prompt.lower() and "gdp" in prompt.lower():
                return "Yes, this can be answered with the provided context. Confidence: 0.85"
            else:
                return "No, I don't have enough information to answer this question."
        elif "provide a precise answer" in prompt:
            # Answer generation
            if "battery life" in prompt.lower():
                return "The iPhone 14 Pro offers up to 23 hours of video playback, 20 hours of streaming video, and 75 hours of audio playback."
            elif "camera" in prompt.lower():
                return "The iPhone 14 Pro features a 48MP main camera with a quad-pixel sensor, which is a significant upgrade from the 12MP camera in previous models. It also includes a 12MP ultra-wide camera and a 12MP telephoto camera with 3x optical zoom."
            elif "pricing" in prompt.lower():
                return "The iPhone 14 Pro starts at $999 for the 128GB model, $1,099 for 256GB, $1,299 for 512GB, and $1,499 for 1TB."
            elif "gdp" in prompt.lower():
                return "The COVID-19 pandemic caused a global GDP contraction of approximately 3.5% in 2020, the worst economic downturn since the Great Depression."
            else:
                return "I don't have enough information to provide a detailed answer."
        elif "component subqueries" in prompt:
            # Answer synthesis for subquery children
            if "main features" in prompt:
                return """
                The main features of the iPhone 14 Pro include its A16 Bionic chip, ProMotion display with Always-On capability, 
                Dynamic Island interface, 48MP camera system, and improved battery life. It also features enhanced durability 
                with Ceramic Shield and water resistance.
                """
            else:
                return "Based on the component subqueries, I can synthesize that this topic has several important aspects that work together to form a comprehensive understanding."
        elif "comprehensive answer to the original query" in prompt:
            # Final answer synthesis
            if "iPhone 14" in prompt:
                return """
                The iPhone 14 Pro is Apple's premium smartphone offering a range of advanced features. 
                
                Camera System: It features a significantly upgraded 48MP main camera (up from 12MP in previous models), along with a 12MP ultra-wide and 12MP telephoto camera with 3x optical zoom.
                
                Battery Life: The device offers impressive battery performance with up to 23 hours of video playback, 20 hours of streaming video, and 75 hours of audio playback.
                
                Pricing: The iPhone 14 Pro is available in different storage configurations starting at $999 for 128GB, $1,099 for 256GB, $1,299 for 512GB, and $1,499 for 1TB.
                
                Based on the available information, these are the key aspects of the iPhone 14 Pro. For a complete picture of all features and expert reviews, additional research would be beneficial.
                """
            elif "COVID-19" in prompt:
                return """
                COVID-19 is a respiratory illness caused by the SARS-CoV-2 virus that emerged in late 2019 and became a global pandemic. 

                Economic Impact: The pandemic caused a severe global economic contraction, with GDP declining by approximately 3.5% in 2020. 
                
                Industries most negatively affected included travel, hospitality, and brick-and-mortar retail, while technology, e-commerce, 
                and healthcare saw significant growth.
                
                Various economic recovery strategies were implemented worldwide, including stimulus packages, business loans, and employment 
                support programs to mitigate the economic damage.
                """
            else:
                return "Based on the available information, I can provide a comprehensive answer to your query that synthesizes the key points from each subquery."
        else:
            return "I don't have a specific response for this prompt type."


def mock_semantic_search(query: str) -> str:
    """Mock semantic search function"""
    logger.info(f"Performing semantic search for: {query}")
    
    query_lower = query.lower()
    if "battery life" in query_lower and "iphone 14" in query_lower:
        return "The iPhone 14 Pro has a battery capacity of 3,200 mAh. It offers up to 23 hours of video playback, 20 hours of streaming video, and 75 hours of audio playback."
    elif "camera" in query_lower and "iphone 14" in query_lower:
        return "The iPhone 14 Pro features a 48MP main camera with a quad-pixel sensor and second-generation sensor-shift optical image stabilization. It also includes a 12MP ultra-wide camera and a 12MP telephoto camera with 3x optical zoom."
    elif "covid" in query_lower and "gdp" in query_lower:
        return "The COVID-19 pandemic caused the global economy to contract by approximately 3.5% in 2020. This was the worst global economic crisis since the Great Depression."
    elif "covid" in query_lower and "industry" in query_lower and "negative" in query_lower:
        return "Industries most negatively impacted by COVID-19 included airlines, tourism, hospitality, brick-and-mortar retail, and oil and gas."
    elif "covid" in query_lower and "industry" in query_lower and "growth" in query_lower:
        return "Industries that saw growth during the pandemic included e-commerce, video streaming services, telehealth, remote work software, and home fitness equipment."
    else:
        return "No relevant information found in semantic search."


def mock_web_search(query: str) -> str:
    """Mock web search function"""
    logger.info(f"Performing web search for: {query}")
    
    query_lower = query.lower()
    if "pricing" in query_lower and "iphone 14" in query_lower:
        return "According to Apple's website, the iPhone 14 Pro starts at $999 for the 128GB model, $1,099 for 256GB, $1,299 for 512GB, and $1,499 for 1TB."
    elif "main features" in query_lower and "iphone 14" in query_lower:
        return "The iPhone 14 Pro's main features include the A16 Bionic chip, Dynamic Island notification system, Always-On display, 48MP camera system, Crash Detection, and Emergency SOS via satellite."
    elif "economic recovery" in query_lower and "covid" in query_lower:
        return "Governments worldwide implemented various economic recovery strategies, including fiscal stimulus packages, central bank monetary policies, business loan programs, and employment support schemes."
    else:
        return "No relevant information found in web search."


def mock_multimodal_search(query: str) -> str:
    """Mock multimodal search function"""
    logger.info(f"Performing multimodal search for: {query}")
    
    query_lower = query.lower()
    if "iphone" in query_lower and "camera" in query_lower:
        return "Image analysis shows the iPhone 14 Pro has a distinctive camera module with three lenses arranged in a triangular pattern."
    else:
        return "No relevant multimodal information found."


def print_query_tree(query_plan: QueryPlan, level: int = 0):
    """Print the query tree with states and answers for visualization"""
    indent = "  " * level
    
    for i, sq in enumerate(query_plan.root_subqueries):
        print(f"{indent}Root Query {i+1}: {sq.text}")
        print(f"{indent}State: {sq.state}")
        if sq.state == SubQueryState.ANSWERED:
            print(f"{indent}Answer: {sq.answer}")
            print(f"{indent}Confidence: {sq.confidence}")
        
        # Print retrieval types
        rt_str = ", ".join([rt.value for rt in sq.retrieval_types])
        print(f"{indent}Retrieval Types: {rt_str}")
        
        # Recursively print children
        if sq.child_subqueries:
            print(f"{indent}Child Subqueries:")
            print_subquery_children(sq.child_subqueries, level + 1)
        
        print()


def print_subquery_children(subqueries: list, level: int):
    """Helper function to print child subqueries"""
    indent = "  " * level
    
    for i, sq in enumerate(subqueries):
        print(f"{indent}Subquery {i+1}: {sq.text}")
        print(f"{indent}State: {sq.state}")
        if sq.state == SubQueryState.ANSWERED:
            print(f"{indent}Answer: {sq.answer}")
            print(f"{indent}Confidence: {sq.confidence}")
        
        # Print retrieval types
        rt_str = ", ".join([rt.value for rt in sq.retrieval_types])
        print(f"{indent}Retrieval Types: {rt_str}")
        
        # Recursively print children
        if sq.child_subqueries:
            print(f"{indent}Child Subqueries:")
            print_subquery_children(sq.child_subqueries, level + 1)
        
        print()


def demo_query_planner():
    """Demonstrate the query planner in action"""
    logger.info("Starting query planner demonstration")
    
    # Initialize the query planner with our mock completion model
    planner = QueryPlanner(completion_model=MockCompletionModel())
    
    # Define the user query
    user_query = "Tell me about the economic impact of COVID-19"
    logger.info(f"User query: {user_query}")
    
    # Set up retrieval functions
    retrieval_functions = {
        RetrievalType.SEMANTIC_SEARCH: mock_semantic_search,
        RetrievalType.WEB_SEARCH: mock_web_search,
        RetrievalType.MULTIMODAL: mock_multimodal_search,
    }
    
    # Execute the query plan
    final_answer = planner.execute_query_plan(user_query, retrieval_functions)
    
    # Print the query tree structure
    print("\n--- QUERY TREE ---")
    query_plan = planner.create_query_plan(user_query)
    for subquery in query_plan.root_subqueries:
        planner.process_subquery_tree(subquery, retrieval_functions)
    print_query_tree(query_plan)
    
    logger.info("Final answer:")
    logger.info(final_answer)
    
    # Try a different query that requires recursive expansion
    print("\n--- SECOND QUERY DEMONSTRATION ---")
    user_query2 = "Tell me all about the iPhone 14 Pro"
    logger.info(f"User query: {user_query2}")
    
    # Execute the query plan
    final_answer2 = planner.execute_query_plan(user_query2, retrieval_functions)
    
    # Print the query tree structure
    print("\n--- QUERY TREE ---")
    query_plan2 = planner.create_query_plan(user_query2)
    for subquery in query_plan2.root_subqueries:
        planner.process_subquery_tree(subquery, retrieval_functions)
    print_query_tree(query_plan2)
    
    logger.info("Final answer:")
    logger.info(final_answer2)


if __name__ == "__main__":
    demo_query_planner() 