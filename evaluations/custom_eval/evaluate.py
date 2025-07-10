#!/usr/bin/env python3
"""LLM-as-a-Judge Evaluation Script using OpenAI's o4-mini-high model.

This script evaluates system-generated answers against golden answers using OpenAI's o4-mini-high model
as a judge. It compares answers from a provided CSV file against the golden answers in
questions_and_answers.csv and outputs accuracy scores.

Usage:
    python evaluate.py answers_system.csv
    python evaluate.py answers_morphik.csv
    python evaluate.py answers_baseline.csv

Input CSV format:
    question,answer
    "What is...",""System's answer here""
    ...

Output:
    - eval_results.csv: Detailed results with correct/incorrect judgments
    - Accuracy score printed to console

Requirements:
    pip install openai pandas
    export OPENAI_API_KEY=your_api_key_here
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

# Initialize OpenAI client
load_dotenv(override=True)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

JUDGE_SYSTEM_PROMPT = "You are an evaluator comparing two answers to the same question."

JUDGE_USER_PROMPT = """Here is a question based on a document or set of documents, along with a reference answer and a ground truth answer.
        Question: {question}
        Reference answer: {system_answer}
        Ground truth answer: {golden_answer}

        Compare the reference answer to the ground truth answer.

        Give a binary result based on the following:

        1 - Each answer is the same in the context of answering the question. Any difference in number is at MOST a rounding error at the tenth of a decimal. There can be difference in style, grammar, or sentence presentation but the core answer is the same.
        0 - The answers are numerically different by more than a rounding error at the tenth of a decimal or saying fundamentally something different."""


def load_golden_answers(golden_path: Path) -> Dict[str, str]:
    """Load golden answers from CSV file."""
    golden_answers = {}

    with open(golden_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            question = row["question"].strip()
            answer = row["answer"].strip()
            golden_answers[question] = answer

    return golden_answers


def load_system_answers(system_path: Path) -> Dict[str, str]:
    """Load system answers from CSV file."""
    system_answers = {}

    with open(system_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            question = row["question"].strip()
            answer = row["answer"].strip()
            system_answers[question] = answer

    return system_answers


def judge_answer(question: str, golden_answer: str, system_answer: str) -> Tuple[str, str]:
    """Use o4-mini-high to judge if the system answer is correct."""
    user_prompt = JUDGE_USER_PROMPT.format(question=question, golden_answer=golden_answer, system_answer=system_answer)

    try:
        response = client.chat.completions.create(
            model="o3",
            messages=[{"role": "system", "content": JUDGE_SYSTEM_PROMPT}, {"role": "user", "content": user_prompt}],
            max_completion_tokens=2000,
        )

        content = response.choices[0].message.content.strip()

        # Parse the binary response (1 for CORRECT, 0 for INCORRECT)
        if content.strip().startswith("1"):
            judgment = "CORRECT"
            explanation = "Answers are equivalent (allowing for minor stylistic differences)"
        elif content.strip().startswith("0"):
            judgment = "INCORRECT"
            explanation = "Answers differ significantly in content or numbers"
        else:
            # Try to extract 1 or 0 from the response
            if "1" in content and "0" not in content:
                judgment = "CORRECT"
                explanation = f"Judge response: {content}"
            elif "0" in content:
                judgment = "INCORRECT"
                explanation = f"Judge response: {content}"
            else:
                # Fallback - treat as incorrect if unclear
                judgment = "INCORRECT"
                explanation = f"Unclear judge response: {content}"

        return judgment, explanation

    except Exception as e:
        print(f"Error evaluating answer: {e}")
        return "ERROR", str(e)


def evaluate_answers(golden_answers: Dict[str, str], system_answers: Dict[str, str]) -> List[Dict]:
    """Evaluate all system answers against golden answers."""
    results = []

    for question, golden_answer in golden_answers.items():
        if question not in system_answers:
            results.append(
                {
                    "question": question,
                    "golden_answer": golden_answer,
                    "system_answer": "[MISSING]",
                    "judgment": "INCORRECT",
                    "explanation": "System answer not provided",
                }
            )
            continue

        system_answer = system_answers[question]
        judgment, explanation = judge_answer(question, golden_answer, system_answer)

        results.append(
            {
                "question": question,
                "golden_answer": golden_answer,
                "system_answer": system_answer,
                "judgment": judgment,
                "explanation": explanation,
            }
        )

        print(f"Evaluated {len(results)}/{len(golden_answers)} questions", end="\r")

    print()  # New line after progress
    return results


def calculate_accuracy(results: List[Dict]) -> float:
    """Calculate accuracy percentage."""
    correct_count = sum(1 for r in results if r["judgment"] == "CORRECT")
    total_count = len(results)
    return (correct_count / total_count) * 100 if total_count > 0 else 0


def save_results(results: List[Dict], output_path: Path) -> None:
    """Save detailed results to CSV."""
    df = pd.DataFrame(results)
    df.to_csv(output_path, index=False)
    print(f"Detailed results saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate system answers using LLM-as-a-Judge")
    parser.add_argument("system_answers_csv", help="Path to CSV file with system answers")
    parser.add_argument(
        "--golden",
        default="questions_and_answers.csv",
        help="Path to golden answers CSV (default: questions_and_answers.csv)",
    )
    parser.add_argument(
        "--output", default="eval_results.csv", help="Output CSV file for detailed results (default: eval_results.csv)"
    )

    args = parser.parse_args()

    # Check if OpenAI API key is set
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set")
        print("Please set it with: export OPENAI_API_KEY=your_api_key_here")
        sys.exit(1)

    # Convert to Path objects
    script_dir = Path(__file__).parent
    system_path = Path(args.system_answers_csv)
    golden_path = script_dir / args.golden
    output_path = Path(args.output)

    # Check if files exist
    if not system_path.exists():
        print(f"Error: System answers file not found: {system_path}")
        sys.exit(1)

    if not golden_path.exists():
        print(f"Error: Golden answers file not found: {golden_path}")
        sys.exit(1)

    print(f"Loading golden answers from: {golden_path}")
    golden_answers = load_golden_answers(golden_path)
    print(f"Loaded {len(golden_answers)} golden answers")

    print(f"Loading system answers from: {system_path}")
    system_answers = load_system_answers(system_path)
    print(f"Loaded {len(system_answers)} system answers")

    print("\nEvaluating answers using o3 as judge...")
    results = evaluate_answers(golden_answers, system_answers)

    # Calculate and display accuracy
    accuracy = calculate_accuracy(results)
    correct_count = sum(1 for r in results if r["judgment"] == "CORRECT")
    total_count = len(results)

    print(f"{'='*60}")
    print("EVALUATION RESULTS")
    print(f"{'='*60}")
    print(f"Total Questions: {total_count}")
    print(f"Correct Answers: {correct_count}")
    print(f"Incorrect Answers: {total_count - correct_count}")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"{'='*60}")

    # Save detailed results
    save_results(results, output_path)

    # Show some example incorrect answers
    incorrect_results = [r for r in results if r["judgment"] == "INCORRECT"]
    if incorrect_results:
        print("\nSample incorrect answers (showing first 3):")
        for i, result in enumerate(incorrect_results[:3]):
            print(f"\nQuestion {i+1}: {result['question'][:100]}...")
            print(f"Golden: {result['golden_answer'][:100]}...")
            print(f"System: {result['system_answer'][:100]}...")
            print(f"Reason: {result['explanation']}")


if __name__ == "__main__":
    main()
