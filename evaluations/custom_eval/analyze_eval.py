#!/usr/bin/env python3
"""Evaluation Results Analyzer

This script analyzes evaluation results by comparing system answers against golden answers.
It calculates accuracy and provides detailed breakdowns of correct vs incorrect responses.

Usage:
    python analyze_eval.py eval_results.csv
    python analyze_eval.py eval_results.csv --golden questions_and_answers.csv
    python analyze_eval.py eval_results.csv --show-examples 5
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List


def load_golden_answers(golden_path: Path) -> Dict[str, str]:
    """Load golden answers from CSV file."""
    golden_answers = {}

    with open(golden_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            golden_answers[row["question"]] = row["answer"]

    return golden_answers


def load_eval_results(eval_path: Path) -> List[Dict[str, str]]:
    """Load evaluation results from CSV file."""
    results = []

    with open(eval_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []

        # Check for required columns
        required_columns = ["question", "system_answer", "golden_answer", "judgment"]
        missing_columns = [col for col in required_columns if col not in fieldnames]

        if missing_columns:
            raise ValueError(
                f"CSV must contain columns: {', '.join(required_columns)}. " f"Missing: {', '.join(missing_columns)}"
            )

        for row in reader:
            result = {
                "question": row["question"],
                "system_answer": row["system_answer"],
                "golden_answer": row["golden_answer"],
                "judgment": row["judgment"],
                "explanation": row.get("explanation", ""),
            }
            results.append(result)

    return results


def calculate_accuracy(results: List[Dict[str, str]]) -> Dict[str, int]:
    """Calculate accuracy metrics."""
    total = len(results)
    correct = sum(1 for r in results if r["judgment"] == "CORRECT")
    incorrect = total - correct

    return {
        "total": total,
        "correct": correct,
        "incorrect": incorrect,
        "accuracy": (correct / total * 100) if total > 0 else 0,
    }


def analyze_answer_lengths(results: List[Dict[str, str]]) -> Dict[str, float]:
    """Analyze answer lengths for correct vs incorrect responses."""
    correct_lengths = [len(r["system_answer"]) for r in results if r["judgment"] == "CORRECT"]
    incorrect_lengths = [len(r["system_answer"]) for r in results if r["judgment"] == "INCORRECT"]

    return {
        "correct_avg": sum(correct_lengths) / len(correct_lengths) if correct_lengths else 0,
        "incorrect_avg": sum(incorrect_lengths) / len(incorrect_lengths) if incorrect_lengths else 0,
        "overall_avg": sum(len(r["system_answer"]) for r in results) / len(results) if results else 0,
    }


def find_common_error_patterns(results: List[Dict[str, str]]) -> Counter:
    """Find common words/phrases in error explanations."""
    error_words = []

    for result in results:
        if result["judgment"] == "INCORRECT" and result["explanation"]:
            # Extract key words from explanations
            words = result["explanation"].lower().split()
            # Filter for meaningful words
            meaningful_words = [
                w for w in words if len(w) > 3 and w not in ["the", "and", "that", "this", "with", "from", "they"]
            ]
            error_words.extend(meaningful_words)

    return Counter(error_words)


def print_sample_results(results: List[Dict[str, str]], judgment: str, limit: int = 3):
    """Print sample results for a given judgment."""
    samples = [r for r in results if r["judgment"] == judgment][:limit]

    if not samples:
        print(f"No {judgment.lower()} examples found.")
        return

    for i, result in enumerate(samples, 1):
        print(f"\nExample {i}:")
        print(f"Question: {result['question'][:100]}...")
        print(f"Golden:   {result['golden_answer'][:100]}...")
        print(f"System:   {result['system_answer'][:100]}...")
        if result["explanation"]:
            print(f"Reason:   {result['explanation'][:100]}...")


def main():
    parser = argparse.ArgumentParser(description="Analyze evaluation results")
    parser.add_argument("eval_file", help="Path to evaluation results CSV file")
    parser.add_argument("--golden", help="Path to golden answers CSV file (optional)")
    parser.add_argument("--show-examples", type=int, default=3, help="Number of examples to show for each category")

    args = parser.parse_args()

    eval_path = Path(args.eval_file)
    if not eval_path.exists():
        print(f"Error: File {eval_path} not found")
        sys.exit(1)

    try:
        print(f"Loading evaluation results from: {eval_path}")
        results = load_eval_results(eval_path)
        print(f"Loaded {len(results)} evaluation results")

        # Calculate metrics
        accuracy_metrics = calculate_accuracy(results)
        length_metrics = analyze_answer_lengths(results)
        error_patterns = find_common_error_patterns(results)

        # Print results
        print("=" * 80)
        print("EVALUATION RESULTS ANALYSIS")
        print("=" * 80)
        print(f"Total Questions: {accuracy_metrics['total']}")
        print(f"Correct Answers: {accuracy_metrics['correct']}")
        print(f"Incorrect Answers: {accuracy_metrics['incorrect']}")
        print(f"Accuracy: {accuracy_metrics['accuracy']:.2f}%")

        print("\nANSWER LENGTH ANALYSIS")
        print("-" * 40)
        print(f"Average length of correct answers: {length_metrics['correct_avg']:.1f} chars")
        print(f"Average length of incorrect answers: {length_metrics['incorrect_avg']:.1f} chars")
        print(f"Overall average answer length: {length_metrics['overall_avg']:.1f} chars")

        if error_patterns:
            print("\nCOMMON ERROR PATTERNS")
            print("-" * 40)
            total_errors = accuracy_metrics["incorrect"]
            for word, count in error_patterns.most_common(5):
                percentage = (count / total_errors * 100) if total_errors > 0 else 0
                print(f"'{word}': {count} occurrences ({percentage:.1f}% of errors)")

        print(f"\nSAMPLE INCORRECT ANSWERS (showing first {args.show_examples}):")
        print("-" * 80)
        print_sample_results(results, "INCORRECT", args.show_examples)

        print(f"\nSAMPLE CORRECT ANSWERS (showing first {args.show_examples}):")
        print("-" * 80)
        print_sample_results(results, "CORRECT", args.show_examples)

        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)
        print(
            f"📊 Accuracy: {accuracy_metrics['accuracy']:.2f}% ({accuracy_metrics['correct']}/{accuracy_metrics['total']})"
        )
        print(f"✅ Correct: {accuracy_metrics['correct']}")
        print(f"❌ Incorrect: {accuracy_metrics['incorrect']}")

        if accuracy_metrics["accuracy"] >= 80:
            print("🎉 Excellent performance!")
        elif accuracy_metrics["accuracy"] >= 60:
            print("👍 Good performance")
        elif accuracy_metrics["accuracy"] >= 40:
            print("⚠️  Moderate performance")
        else:
            print("🔧 Needs improvement")

    except Exception as e:
        print(f"Error analyzing results: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
