import os
import sys
import uuid
from pathlib import Path

# Add the SDK path to the Python path
sdk_path = str(Path(__file__).parent.parent / "sdks" / "python")
sys.path.insert(0, sdk_path)

from datasets import load_dataset, Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_correctness, context_precision
import pandas as pd
from dotenv import load_dotenv
from morphik import Morphik
from tqdm import tqdm
import argparse

# Load environment variables
load_dotenv()

# Connect to Morphik
db = Morphik(timeout=10000, is_local=True)

# Generate a run identifier
def generate_run_id():
    """Generate a unique run identifier"""
    return str(uuid.uuid4())


def load_hotpotqa_dataset(num_samples=10, split="validation"):
    """Load and prepare the HotpotQA dataset"""
    dataset = load_dataset("hotpot_qa", "distractor", split=split, trust_remote_code=True)

    # Sample a subset
    dataset = dataset.select(range(min(num_samples, len(dataset))))

    return dataset


def process_with_morphik(dataset, run_id=None):
    """
    Process dataset with Morphik and prepare data for RAGAS evaluation
    
    Args:
        dataset: The dataset to process
        run_id: Unique identifier for this evaluation run
    """
    # Generate a run_id if not provided
    if run_id is None:
        run_id = generate_run_id()
        
    print(f"Using run identifier: {run_id}")
    
    data_samples = {
        "question": [], 
        "answer": [], 
        "contexts": [], 
        "ground_truth": [],
        "run_id": []  # Store run_id for each sample
    }

    for i, item in enumerate(tqdm(dataset, desc="Processing documents")):
        try:
            # Extract question and ground truth
            question = item["question"].strip()
            ground_truth = item["answer"].strip()

            if not question or not ground_truth:
                print(f"Skipping item {i}: Empty question or answer")
                continue

            # Ingest the document's context into Morphik
            context = ""
            for title, sentences in zip(item["context"]["title"], item["context"]["sentences"]):
                paragraph = " ".join(sentences)
                context += f"{title}:\n{paragraph}\n\n"

            # Handle a potentially longer context
            # if len(context) > 10000:
            #     print(f"Warning: Long context ({len(context)} chars), truncating...")
            #     context = context[:10000]

            # Ingest text with run_id in metadata
            doc_id = db.ingest_text(
                context,
                metadata={
                    "source": "hotpotqa",
                    "question_id": item.get("_id", ""),
                    "item_index": i,
                    "evaluation_run_id": run_id  # Add run_id to metadata
                },
                embed_as_image=False
            ).external_id

            # Query Morphik for the answer with concise prompt override
            prompt_override = {
                "query": {
                    "prompt_template": "Answer the following question based on the provided context. Your answer should be as concise as possible. If a yes/no answer is appropriate, just respond with 'Yes' or 'No'. Do not provide explanations or additional context unless absolutely necessary.\n\nQuestion: {question}\n\nContext: {context}"
                }
            }
            response = db.query(
                question, 
                embed_as_image=False, 
                k=10, 
                filters={"evaluation_run_id": run_id},
                prompt_overrides=prompt_override
            )
            answer = response.completion

            if not answer:
                print(f"Warning: Empty answer for question: {question[:50]}...")
                answer = "No answer provided"

            # Get retrieved chunks for context with filter by run_id
            chunks = db.retrieve_chunks(
                query=question, 
                k=10, 
                filters={"evaluation_run_id": run_id}  # Filter by run_id
            )
            context_texts = [chunk.content for chunk in chunks]

            if not context_texts:
                print(f"Warning: No contexts retrieved for question: {question[:50]}...")
                context_texts = ["No context retrieved"]

            # Add to our dataset
            data_samples["question"].append(question)
            data_samples["answer"].append(answer)
            data_samples["contexts"].append(context_texts)
            data_samples["ground_truth"].append(ground_truth)
            data_samples["run_id"].append(run_id)

        except Exception as e:
            import traceback

            print(f"Error processing item {i}:")
            print(f"Question: {item.get('question', 'N/A')[:50]}...")
            print(f"Error: {e}")
            traceback.print_exc()
            continue

    return data_samples, run_id


def run_evaluation(num_samples=5, output_file="ragas_results.csv", run_id=None):
    """
    Run the full evaluation pipeline
    
    Args:
        num_samples: Number of samples to use from the dataset
        output_file: Path to save the results CSV
        run_id: Optional run identifier. If None, a new one will be generated
    """
    try:
        # Load dataset
        print("Loading HotpotQA dataset...")
        hotpot_dataset = load_hotpotqa_dataset(num_samples=num_samples)
        print(f"Loaded {len(hotpot_dataset)} samples from HotpotQA")

        # Process with Morphik
        print("Processing with Morphik...")
        data_samples, run_id = process_with_morphik(hotpot_dataset, run_id=run_id)

        # Check if we have enough samples
        if len(data_samples["question"]) == 0:
            print("Error: No samples were successfully processed. Exiting.")
            return

        print(f"Successfully processed {len(data_samples['question'])} samples")

        # Convert to RAGAS format
        ragas_dataset = Dataset.from_dict(data_samples)

        # Run RAGAS evaluation
        print("Running RAGAS evaluation...")
        metrics = [faithfulness, answer_correctness, context_precision]

        result = evaluate(ragas_dataset, metrics=metrics)

        # Convert results to DataFrame and save
        df_result = result.to_pandas()
        
        # Add run_id to the results DataFrame
        df_result['run_id'] = run_id
        
        print("\nRAGAS Evaluation Results:")
        print(df_result)

        # Add more detailed analysis
        print("\nDetailed Metric Analysis:")
        # First ensure all metric columns are numeric
        for column in ["faithfulness", "answer_correctness", "context_precision"]:
            if column in df_result.columns:
                try:
                    # Convert column to numeric, errors='coerce' will replace non-numeric values with NaN
                    df_result[column] = pd.to_numeric(df_result[column], errors="coerce")
                    # Calculate and print mean, ignoring NaN values
                    mean_value = df_result[column].mean(skipna=True)
                    if pd.notna(mean_value):  # Check if mean is not NaN
                        print(f"{column}: {mean_value:.4f}")
                    else:
                        print(f"{column}: No valid numeric values found")
                except Exception as e:
                    print(f"Error processing {column}: {e}")
                    print(f"Values: {df_result[column].head().tolist()}")

        # Include run_id in the output filename if not explicitly provided
        if output_file == "ragas_results.csv":
            # Get just the filename without extension
            base_name = output_file.rsplit('.', 1)[0]
            output_file = f"{base_name}_{run_id}.csv"
            
        # Save results
        df_result.to_csv(output_file, index=False)
        print(f"Results saved to {output_file}")

        return df_result, run_id

    except Exception as e:
        import traceback

        print(f"Error in evaluation: {e}")
        traceback.print_exc()
        print("Exiting due to error.")
        return None


def main():
    """Command-line entry point"""
    parser = argparse.ArgumentParser(
        description="Run RAGAS evaluation on Morphik using HotpotQA dataset"
    )
    parser.add_argument(
        "--samples", type=int, default=5, help="Number of samples to use (default: 5)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="ragas_results.csv",
        help="Output file for results (default: ragas_results.csv)",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Specific run identifier to use (default: auto-generated UUID)",
    )
    args = parser.parse_args()

    run_evaluation(
        num_samples=args.samples,
        output_file=args.output,
        run_id=args.run_id
    )


if __name__ == "__main__":
    main()
