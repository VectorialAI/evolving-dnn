# Evolving Deep Neural Networks

This repository implements an evolutionary algorithm framework for automatically evolving Deep Neural Network architectures and hyperparameters, with a specific focus on GPT (Generative Pre-trained Transformer) models represented as Directed Acyclic Graphs (DAGs) via [torch.fx](https://docs.pytorch.org/docs/2.7/). However, the code is domain-agnostic and allows non-transformer architectures, and even non-neural-network evolutionary search.

## Overview

The project uses evolutionary algorithms to:
- **Evolve neural network architectures** by adding/removing nodes in the DAG representation of the neural network
- **Optimize hyperparameters** including learning rates, batch sizes, and optimizer settings  
- **Train and evaluate** models using fitness-based selection
- **Generate populations** of diverse neural network configurations

## Project Structure

```
src/
├── evolution.py      # Core evolutionary algorithm framework
├── individual.py     # Base individual class
├── mingpt_altered/   # Modified minGPT implementation
│   ├── model.py
│   ├── trainer.py
│   └── utils.py
├── nn/               # Neural network specific implementations
│   ├── core.py                     # Graph representation utilities
│   ├── individual.py               # Neural network individual class
│   ├── individual_graph_module.py  # PyTorch FX graph module wrapper
│   ├── evolution.py                # NN-specific evolution class
│   ├── evaluate.py                 # Fitness evaluation (perplexity-based)
│   ├── dataset.py                  # Text dataset for training
│   ├── visualization.py            # Graph visualization utilities
│   └── variation/                  # Mutation and crossover operations
│       ├── architecture_mutation.py     # Add/remove layers, connections
│       ├── architecture_crossover.py    # Architecture crossover operations  
│       ├── architecture_adaptation.py   # Architecture adaptation operations
│       ├── hyperparam_variation.py      # Hyperparameter mutations/crossover
│       └── utils.py                     # Helper functions for graph manipulation
└── gpt_evolution/    # GPT-specific evolution setup
    ├── run.py                      # Main execution script
    ├── initial_population.py       # Generate initial GPT population
    ├── helpers.py                  # Helper functions for GPT evolution execution
    ├── default_run_config.json     # Default run configuration
    └── local_run_config.json       # Local override configuration
```

## Key Components

### 1. Evolution Framework (`evolution.py`)

The core `Evolution` class implements a genetic algorithm with:
- **Population management**: Maintains a population of individuals
- **Selection**: Fitness-based selection of parents
- **Reproduction**: Crossover and mutation operations
- **Fitness evaluation**: Variable fitness functions

### 2. Neural Network Individuals (`nn/individual.py`)

`NeuralNetworkIndividual` extends the base `Individual` class with:
- **Graph representation**: Uses PyTorch FX for computational graphs
- **Training configuration**: Stores hyperparameters and optimizer settings
- **Deep copying**: Supports proper cloning for evolution operations

### 3. Architecture Evolution (`nn/variation/`)

#### Mutations:
- `mutation_add_linear`: Add fully connected layers
- `mutation_add_relu`: Add ReLU activation layers  
- `mutation_add_skip_connection`: Add residual connections
- `mutation_add_branch`: Add parallel computation branches
- `mutation_remove_node`: Remove unnecessary nodes

#### Crossover:
- `crossover_subgraph`: Exchange subgraphs between parents
- Hyperparameter crossover: Average training parameters

### 4. Hyperparameter Evolution

Evolves training configurations:
- **Learning rates**: Log-normal mutations
- **Batch sizes**: Normal distribution with bounds
- **Optimizer parameters**: Beta values, weight decay, gradient clipping

### 5. Fitness Evaluation (`nn/evaluate.py`)

Uses **negative perplexity** as fitness:
- Trains models for limited iterations
- Evaluates on validation set
- Returns `-perplexity` (higher is better for evolution)

## Usage

### Basic Usage

```python
from src.gpt_evolution.run import *

# Run GPT evolution
python src/gpt_evolution/run.py
```

### Custom Evolution Setup

```python
from src.evolution import Evolution
from src.nn.individual import NeuralNetworkIndividual
from src.nn.evaluate import calculate_fitness
from src.nn.gpt_evolution import generate_initial_population

# Create initial population
population = generate_initial_population(
    population_size=10,
    vocab_size=2000, 
    gpt_config_params={
        "block_size": 128,
        "layer_bounds": (2, 5),
        "head_bounds": (2, 5), 
        "embed_bounds": (128, 512)
    },
    train_config_params={"training_total_batches": 100, "device": "cpu"}
)

# Setup evolution
evolution = NeuralNetworkEvolution(
    population=population,
    fitness_fn=calculate_fitness,
    crossover_instead_of_mutation_rate=0.5,
    mutation_fns_and_probabilities=[
        (mutation_add_linear, 0.2),
        (mutation_add_relu, 0.2),
        # ... other mutations
    ],
    crossover_fns_and_probabilities=[
        (crossover_subgraph, 0.3),
        # ... other crossovers  
    ],
    target_population_size=10,
    num_children_per_generation=10
)

# Run evolution
evolution.run_evolution(num_generations=5)
```

## Configuration

### GPT Configuration Parameters

```python
gpt_config_params = {
    "block_size": 128,           # Sequence length
    "layer_bounds": (2, 5),      # Min/max transformer layers
    "head_bounds": (2, 5),       # Min/max attention heads  
    "embed_bounds": (128, 512),  # Min/max embedding dimension
}
```

### Training Configuration Parameters

```python
train_config_params = {
    "training_total_batches": 100,            # Training iterations
    "device": "cpu",             # Device (cpu/cuda)
    "batch_size_bounds": (32, 128),
    "learning_rate_bounds": (1e-5, 1e-3),
    # ... other hyperparameter bounds
}
```

## Dependencies

- **PyTorch**: Neural network framework and FX graph representation
- **mingpt**: Base GPT implementation (external dependency loaded into the repository for ease of use) from [Andrej Karpathy's mingpt](https://github.com/karpathy/minGPT)
- **numpy**: Numerical operations for mutations

## Input Data

The system expects:
- **Text file**: `mingpt/input.txt` containing training text
- **Tokenization**: BPE preprocessing creates vocabulary and token mappings
- **Dataset**: Sliding window approach for sequence prediction

## Output

- **Evolved models**: Best performing architectures and hyperparameters
- **Fitness logs**: Generation-by-generation performance tracking
- **Graph visualizations**: SVG exports of network architectures (if enabled)
- **Population history**: Complete evolution trace

## Key Features

- **PyTorch FX Integration**: Dynamic graph manipulation and shape propagation
- **Modular Design**: Easy to extend with new mutation/crossover operations
- **Robust Error Handling**: Graceful handling of invalid architectures
- **Shape Adaptation**: Automatic tensor size matching for graph modifications
- **Comprehensive Logging**: Detailed evolution progress tracking

## Extending the Framework

### Adding New Mutations

```python
def my_custom_mutation(individual):
    # Implement your mutation logic
    # Modify individual.graph_module or individual.train_config
    return individual

# Add to evolution setup
mutation_fns_and_probabilities=[
    (my_custom_mutation, 0.1),
    # ... existing mutations
]
```

### Adding New Crossovers

```python
def my_custom_crossover(individual):
    # Implement your mutation logic
    # Modify individual.graph_module or individual.train_config
    return individual

# Add to evolution setup
crossover_fns_and_probabilities=[
    (my_custom_crossover, 0.1),
    # ... existing crossovers
]
```

### Custom Fitness Functions

```python
def my_fitness_function(individual):
    # Implement custom evaluation
    # Return float (higher = better)
    return fitness_score

evolution = Evolution(
    population=population,
    fitness_fn=my_fitness_function,
    # ... other parameters
)
```

## Running an Experiment on a Fresh Ubuntu 20.04 Server

To quickly set up and run an experiment on a new Ubuntu 20.04 server, execute the following commands **manually in your terminal**. These commands will install necessary dependencies, clone the repository, and launch the experiment:

```bash
apt update
apt install git software-properties-common -y
git clone https://www.github.com/PatrickNercessian/evolving-dnn
cd evolving-dnn/
chmod +x run_experiment.sh
./run_experiment.sh
```

> **Note:**  
> These commands are also available in the `commands.txt` file in the repository root.  
> The script will set up Python 3.11, create a virtual environment, install requirements, and start the experiment.