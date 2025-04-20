# SciER Evaluation for Morphik

This directory contains scripts for evaluating different language models' entity and relation extraction capabilities against the [SciER (Scientific Entity Recognition) dataset](https://github.com/allenai/SciERC).

## Overview

The evaluation workflow is split into two parts:
1. **Graph Creation**: Generate a knowledge graph using your configured model in `morphik.toml`
2. **Graph Evaluation**: Evaluate the created graph against the ground truth annotations

This separation allows you to test different models by changing the configuration in `morphik.toml` between runs.

The evaluation uses the SciER test dataset which can be found at [https://github.com/edzq/SciER/blob/main/SciER/LLM/test.jsonl](https://github.com/edzq/SciER/blob/main/SciER/LLM/test.jsonl).

## Setup

1. Make sure you have a local Morphik server running
2. Set up your OpenAI API key for evaluation:
   ```bash
   export OPENAI_API_KEY=your_key_here
   ```

## Running the Evaluation

### Step 1: Configure Your Model

Edit `morphik.toml` to specify the model you want to test:

```toml
[graph]
model = "openai_gpt4o" # Reference to a key in registered_models
enable_entity_resolution = true
```

### Step 2: Create a Knowledge Graph

Run the graph creation script:

```bash
python scier_evaluation.py --model-name gpt4o
```

The script will output a graph name like `scier_gpt4o_12345678` when complete.

### Step 3: Evaluate the Knowledge Graph

Evaluate the created graph:

```bash
python evaluate_result.py --graph-name scier_gpt4o_12345678
```

### Step 4: Test Different Models

To compare models:
1. Change the model in `morphik.toml`
2. Repeat steps 2-3 with a different `--model-name`
3. Compare the resulting metrics and visualizations

## Command Arguments

### scier_evaluation.py
- `--model-name`: Name to identify this model in results (default: "default_model")
- `--limit`: Maximum number of documents to process (default: 57)
- `--run-id`: Unique identifier for the run (default: auto-generated)

### evaluate_result.py
- `--graph-name`: Name of the graph to evaluate (**required**)
- `--model-name`: Name for the evaluation results (default: "existing_model_openai")
- `--similarity-threshold`: Threshold for semantic similarity matching (default: 0.70)
- `--embedding-model`: OpenAI embedding model to use (default: "text-embedding-3-small")

## Results

The evaluation generates:
- CSV files with precision, recall, and F1 metrics
- Visualizations comparing entity and relation extraction performance
- Entity and relation count comparisons

Results are saved in a directory with the format: `scier_results_{model_name}_{run_id}/`

## Tips for Model Comparison

- Use descriptive model names in the `--model-name` parameter
- Keep the same similarity threshold across evaluations
- Compare models using the generated visualization charts
- Look at both entity and relation extraction metrics
