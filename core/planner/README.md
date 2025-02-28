# DataBridge Query Planner

The Query Planner is a component that takes a user query, breaks it down into subqueries, determines appropriate retrieval methods for each subquery, and iteratively searches for information until it can provide a comprehensive answer.

## Architecture

The planner consists of several key components:

1. **QueryPlanner**: The main class that orchestrates the query planning process.
2. **QueryPlan**: A data structure representing the plan for answering a user query.
3. **SubQuery**: Represents individual subqueries with state transitions.
4. **SubQueryState**: Enum defining possible states (UNANSWERED, PROCESSING, ANSWERED, FAILED).
5. **RetrievalType**: An enum defining different retrieval methods (semantic search, web search, etc.).

## State Design Pattern

The Query Planner uses a state design pattern for handling subqueries:

1. **Unanswered State**: Initial state of subqueries, indicating they need information.
2. **Processing State**: Indicates that information is being gathered for the subquery.
3. **Answered State**: Indicates the subquery has been successfully answered.
4. **Failed State**: Indicates that answering the subquery failed.

Subqueries transition between these states based on retrieval results and LLM evaluations. This approach provides clear status tracking and enables conditional logic based on the current state.

## Hierarchical Query Expansion

The system uses a hierarchical approach to query decomposition:

1. The original query is broken down into root-level subqueries.
2. Each subquery can be recursively broken down into child subqueries if it cannot be answered directly.
3. This creates a tree structure of queries, where leaf nodes are directly answerable.
4. Answers propagate up the tree - child answers are combined to answer their parent.
5. The final answer is synthesized from the answers to all root subqueries.

This recursive approach allows handling complex queries by breaking them down until they reach a level of simplicity that can be answered directly.

## Core Functions

The Query Planner performs the following key steps:

1. **Query Decomposition**: Breaks down complex queries into simpler subqueries.
2. **Retrieval Method Identification**: Determines which retrieval methods to use for each subquery.
3. **Answerability Evaluation**: Assesses whether a query can be answered with available information.
4. **Recursive Expansion**: Further breaks down unanswered subqueries into simpler components.
5. **Answer Synthesis**: Combines answers from child subqueries to answer their parent.
6. **Final Answer Generation**: Creates a comprehensive answer from the collected information.

## How It Works

The typical flow for processing a query is:

1. Create a query plan by breaking down the original query into root subqueries
2. For each root subquery:
   a. Retrieve information using the determined retrieval methods
   b. Evaluate if the subquery can be answered with the retrieved information
   c. If not answerable and not at max depth, recursively break it down into child subqueries
   d. Process each child subquery using the same approach
   e. Once all children are answered, synthesize their answers to answer the parent
3. Once all root subqueries are answered, synthesize a final answer to the original query

## Integration with DataBridge

The Query Planner can be integrated with the DataBridge system in the following ways:

1. **API Integration**: Add a new endpoint in core/api.py that uses the QueryPlanner.
2. **Retrieval Function Mapping**: Connect the planner's RetrievalType enum to actual DataBridge retrieval functions.
3. **LLM Connection**: Provide a real completion model for the planner to use instead of mocks.

## Usage Example

```python
from core.planner import QueryPlanner, RetrievalType
from core.completion.openai_completion import OpenAICompletionModel

# Initialize components
completion_model = OpenAICompletionModel()
planner = QueryPlanner(completion_model=completion_model)

# Define retrieval functions
retrieval_functions = {
    RetrievalType.SEMANTIC_SEARCH: vector_store.search,
    RetrievalType.WEB_SEARCH: web_search_function,
    RetrievalType.KNOWLEDGE_GRAPH: knowledge_graph_search,
    # Add other retrieval methods as needed
}

# Execute query
user_query = "What were the economic impacts of the COVID-19 pandemic on the airline industry?"
answer = planner.execute_query_plan(user_query, retrieval_functions)
```

## Demo

A demonstration script is provided in `demo.py` showing how the Query Planner works with mock retrieval functions. Run it with:

```
python -m core.planner.demo
```

The demo visualizes the hierarchical query tree and shows how subqueries transition through different states.

## Configuration

The Query Planner can be configured in several ways:

- **max_iterations**: The maximum number of search iterations before giving up
- **max_recursion_depth**: The maximum depth for recursive query expansion
- **Embedding model**: Can be provided to support semantic similarity for subquery matching
- **Custom prompts**: The internal prompts can be overridden for custom behavior

## Future Improvements

- Support for structured queries (SQL, SPARQL, etc.)
- Learning from past query plans to improve efficiency
- Adding reasoning about confidence scores for different retrieval methods
- Supporting more retrieval types like tabular data, code search, and time-series data
- Parallelizing subquery processing for better performance 