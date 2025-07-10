# Financial Document RAG Evaluation Framework

This evaluation framework tests RAG systems on challenging financial document analysis tasks using documents from NVIDIA, Palantir, and JPMorgan.

## 📋 Overview

The framework evaluates RAG systems on 45 carefully curated questions from TLDC (The LLM Data Company) covering:
- Complex numerical calculations across multiple pages
- Multi-step reasoning requiring document synthesis
- Cross-document analysis and comparisons
- Time-series data interpretation
- Financial metric derivations

## 📁 Files

- **`questions_and_answers.csv`** - 45 golden questions and answers
- **`docs/`** - Financial documents (NVIDIA 10-Q, Palantir investor presentation, JPMorgan midyear report)
- **`base_eval.py`** - Abstract base class for RAG evaluators
- **`morphik_eval.py`** - Morphik system implementation
- **`openai_eval.py`** - OpenAI baseline example
- **`evaluate.py`** - LLM judge for binary evaluation
- **`analyze_eval.py`** - Results analysis utility

## 🚀 Quick Start

### 1. Run Evaluation

```bash
# Run Morphik evaluation
python morphik_eval.py

# Results saved to: morphik_results.csv
```

### 2. Judge Results

```bash
# Evaluate system answers against golden answers
python evaluate.py morphik_results.csv

# Creates: morphik_results_judged.csv
```

### 3. Analyze Results

```bash
# Simple accuracy analysis
python analyze_eval.py morphik_results_judged.csv

# With custom golden answers file
python analyze_eval.py results.csv --golden my_questions.csv

# Show more examples
python analyze_eval.py results.csv --show-examples 5
```

## 📊 Analysis Features

The `analyze_eval.py` script provides:

- **Accuracy Metrics**: Overall accuracy with correct/incorrect breakdown
- **Answer Length Analysis**: Compare lengths of correct vs incorrect answers
- **Error Pattern Detection**: Common words/phrases in incorrect explanations
- **Sample Analysis**: View examples of correct and incorrect answers

### Required CSV Format

The evaluation results must be in the following format:

```csv
question,system_answer,golden_answer,judgment,explanation
"What is X?","System answer","Golden answer","CORRECT","Explanation of judgment"
```

Required columns:
- `question`: The question being evaluated
- `system_answer`: The RAG system's response
- `golden_answer`: The expected correct answer
- `judgment`: "CORRECT" or "INCORRECT"
- `explanation`: Optional explanation of the judgment

## 📈 Example Output

```
================================================================================
EVALUATION RESULTS ANALYSIS
================================================================================
Total Questions: 45
Correct Answers: 32
Incorrect Answers: 13
Accuracy: 71.11%

ANSWER LENGTH ANALYSIS
----------------------------------------
Average length of correct answers: 245.3 chars
Average length of incorrect answers: 198.7 chars
Overall average answer length: 231.4 chars

COMMON ERROR PATTERNS
----------------------------------------
'calculation': 8 occurrences (61.5% of errors)
'missing': 5 occurrences (38.5% of errors)

================================================================================
SUMMARY
================================================================================
📊 Accuracy: 71.11% (32/45)
✅ Correct: 32
❌ Incorrect: 13
👍 Good performance!
```

## 🏆 Benchmark Results (July 8, 2025)

Performance comparison on the 45-question financial document evaluation set:

| System | Accuracy | Correct | Incorrect | Performance |
|--------|----------|---------|-----------|-------------|
| **Morphik** | **95.56%** | **43/45** | **2** | 🎉 Excellent |
| OpenAI GPT-4 | 13.33% | 6/45 | 39 | 🔧 Needs improvement |

### Key Insights:
- **Morphik** demonstrates exceptional performance with 95.56% accuracy, correctly answering 43 out of 45 complex financial questions
- **OpenAI GPT-4** baseline struggles with document-grounded reasoning, achieving only 13.33% accuracy
- Average answer length: Morphik (834 chars) vs OpenAI (1,888 chars) - Morphik provides more concise, accurate responses
- Error patterns show most failures involve content/numerical differences rather than formatting issues

This demonstrates Morphik's superior capability for complex financial document analysis and multi-step reasoning tasks.

## 🏗️ Adding New RAG Systems

Create a new evaluator by inheriting from `BaseRAGEvaluator`:

```python
from base_eval import BaseRAGEvaluator

class MyRAGEvaluator(BaseRAGEvaluator):
    def setup_client(self):
        # Initialize your RAG system
        pass

    def ingest(self, docs_folder: str):
        # Ingest documents
        pass

    def query(self, question: str) -> str:
        # Query and return answer
        return "answer"

# Usage
evaluator = MyRAGEvaluator("my_results.csv")
evaluator.run()
```

## 🎯 Evaluation Criteria

The LLM judge uses binary scoring (1 = CORRECT, 0 = INCORRECT) based on:
- **Content Equivalence**: Same factual information and conclusions
- **Numerical Accuracy**: Allows rounding differences up to 0.1 decimal places
- **Flexible Formatting**: Ignores minor stylistic differences
- **Completeness**: All required components present

## 📚 Documents

### NVIDIA 10-Q (Form 10-Q for Q1 FY2026)
- Financial statements and metrics
- Business segment performance
- Forward-looking statements

### Palantir Q1 2025 Investor Presentation
- Revenue growth and profitability metrics
- Customer acquisition and retention
- Rule of 40 performance

### JPMorgan Midyear 2024 Report
- Economic outlook and market analysis
- Risk management and regulatory updates
- Strategic initiatives and performance

## 🔧 Configuration

Judge model can be configured in `evaluate.py`:
```python
MODEL = "o3"  # Current judge model
```

## 📋 Requirements

- Python 3.8+
- OpenAI API key (for judge)
- RAG system dependencies (varies by implementation)

## 🤝 Contributing

1. Add new evaluator classes for different RAG systems
2. Extend analysis capabilities in `analyze_eval.py`
3. Add new question sets for different domains
4. Improve judge prompt engineering
