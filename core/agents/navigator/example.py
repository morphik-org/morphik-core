#!/usr/bin/env python3
"""
Example usage of the PDF Document Navigator Agent.

This script demonstrates how to use the DocumentNavigatorAgent to analyze
PDF documents and answer complex questions about their content.
"""

import asyncio
import os

from core.agents.navigator.agent import DocumentNavigatorAgent
from core.database.base_database import BaseDatabase
from core.models.auth import AuthContext
from core.storage.base_storage import BaseStorage


class MockStorage(BaseStorage):
    """Mock storage implementation for example purposes."""

    async def download_file(self, bucket: str, key: str) -> bytes:
        # In a real implementation, this would download from cloud storage
        # For this example, we'll assume the file exists locally
        file_path = f"./documents/{key}"
        if os.path.exists(file_path):
            with open(file_path, "rb") as f:
                return f.read()
        else:
            raise FileNotFoundError(f"File not found: {file_path}")


class MockDatabase(BaseDatabase):
    """Mock database implementation for example purposes."""

    async def get_document(self, document_id: str, auth: AuthContext):
        # Mock document metadata
        class MockDocument:
            def __init__(self):
                self.storage_files = [MockStorageInfo()]

        class MockStorageInfo:
            def __init__(self):
                self.bucket = "documents"
                self.key = f"{document_id}.pdf"

        return MockDocument()


class MockAuthContext(AuthContext):
    """Mock authentication context for example purposes."""

    def __init__(self, user_id: str = "example_user"):
        self.user_id = user_id


async def analyze_research_paper(agent: DocumentNavigatorAgent):
    """Example: Analyze a research paper."""
    print("🔬 Analyzing Research Paper...")
    print("=" * 50)

    query = """
    Please analyze this research paper and provide a comprehensive summary including:

    1. **Research Question & Hypothesis**: What is the main research question and hypothesis?
    2. **Methodology**: What research methods were used?
    3. **Key Findings**: What are the most important results?
    4. **Conclusions**: What conclusions do the authors draw?
    5. **Limitations**: What limitations do the authors acknowledge?
    6. **Future Work**: What future research directions are suggested?

    Please navigate through the document to gather this information and provide
    a well-structured analysis.
    """

    try:
        response = await agent.analyze_document(query, max_iterations=15)
        print(response)
    except Exception as e:
        print(f"Error analyzing research paper: {e}")


async def analyze_financial_report(agent: DocumentNavigatorAgent):
    """Example: Analyze a financial report."""
    print("\n💰 Analyzing Financial Report...")
    print("=" * 50)

    query = """
    Extract and analyze key financial information from this report:

    1. **Revenue Trends**: What are the revenue figures and trends?
    2. **Profitability**: What are the profit margins and trends?
    3. **Key Metrics**: What are the important financial ratios and KPIs?
    4. **Risk Factors**: What risks are mentioned?
    5. **Management Outlook**: What is management's outlook for the future?
    6. **Charts & Graphs**: Analyze any financial charts or graphs present.

    Please navigate through the document to find all relevant financial data.
    """

    try:
        response = await agent.analyze_document(query, max_iterations=12)
        print(response)
    except Exception as e:
        print(f"Error analyzing financial report: {e}")


async def analyze_technical_documentation(agent: DocumentNavigatorAgent):
    """Example: Analyze technical documentation."""
    print("\n🔧 Analyzing Technical Documentation...")
    print("=" * 50)

    query = """
    Review this technical document and extract:

    1. **System Overview**: What system or technology is being described?
    2. **Architecture**: What is the system architecture?
    3. **Requirements**: What are the system requirements?
    4. **APIs & Interfaces**: What APIs or interfaces are documented?
    5. **Configuration**: What configuration options are available?
    6. **Diagrams**: Analyze any technical diagrams or flowcharts.

    Focus on providing actionable technical insights.
    """

    try:
        response = await agent.analyze_document(query, max_iterations=10)
        print(response)
    except Exception as e:
        print(f"Error analyzing technical documentation: {e}")


async def demonstrate_navigation(agent: DocumentNavigatorAgent):
    """Demonstrate basic navigation capabilities."""
    print("\n🧭 Demonstrating Navigation...")
    print("=" * 50)

    # Get current state
    state = agent.get_current_state()
    print(f"Current state: {state}")

    # Navigate to different pages
    print("\nNavigating to page 3...")
    result = await agent.quick_navigate("jump", page=3)
    print(f"Navigation result: {result}")

    # Move to next page
    print("\nMoving to next page...")
    result = await agent.quick_navigate("next")
    print(f"Navigation result: {result}")

    # Zoom into a region (top-left quadrant)
    print("\nZooming into top-left region...")
    result = await agent.quick_navigate("zoom", ymin=0, ymax=500, xmin=0, xmax=500)
    print(f"Zoom result: {result}")

    # Get page info
    result = await agent.quick_navigate("info")
    print(f"Page info: {result}")


async def interactive_mode(agent: DocumentNavigatorAgent):
    """Interactive mode for asking custom questions."""
    print("\n💬 Interactive Mode")
    print("=" * 50)
    print("Ask questions about the document (type 'quit' to exit):")

    while True:
        try:
            query = input("\n📝 Your question: ").strip()

            if query.lower() in ["quit", "exit", "q"]:
                break

            if not query:
                continue

            print("\n🤔 Analyzing...")
            response = await agent.analyze_document(query)
            print(f"\n📋 Response:\n{response}")

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")

    print("\nGoodbye! 👋")


async def main():
    """Main example function."""
    print("🚀 PDF Document Navigator Agent Example")
    print("=" * 60)

    # Check for API key
    if not os.getenv("GEMINI_API_KEY"):
        print("❌ Error: GEMINI_API_KEY environment variable not set")
        print("Please set your Gemini API key:")
        print("export GEMINI_API_KEY='your-api-key-here'")
        return

    # Mock dependencies
    storage = MockStorage()
    database = MockDatabase()
    auth = MockAuthContext()

    # Document ID (should correspond to a PDF file in ./documents/)
    document_id = input("📄 Enter document ID (filename without .pdf): ").strip()

    if not document_id:
        print("❌ Document ID is required")
        return

    try:
        # Initialize agent
        print(f"\n🔄 Initializing agent for document: {document_id}")
        agent = DocumentNavigatorAgent(document_id, storage, database, auth)
        await agent.initialize()
        print("✅ Agent initialized successfully!")

        # Show available examples
        print("\n📋 Available Examples:")
        print("1. Research Paper Analysis")
        print("2. Financial Report Analysis")
        print("3. Technical Documentation Analysis")
        print("4. Navigation Demonstration")
        print("5. Interactive Mode")

        choice = input("\n🎯 Choose an example (1-5): ").strip()

        if choice == "1":
            await analyze_research_paper(agent)
        elif choice == "2":
            await analyze_financial_report(agent)
        elif choice == "3":
            await analyze_technical_documentation(agent)
        elif choice == "4":
            await demonstrate_navigation(agent)
        elif choice == "5":
            await interactive_mode(agent)
        else:
            print("❌ Invalid choice")
            return

    except FileNotFoundError as e:
        print(f"❌ File not found: {e}")
        print("Make sure the PDF file exists in the ./documents/ directory")
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    # Create documents directory if it doesn't exist
    os.makedirs("./documents", exist_ok=True)

    # Run the example
    asyncio.run(main())
