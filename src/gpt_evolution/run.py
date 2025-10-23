import os
import argparse

import json
import logging
import platform
import subprocess
import sys

from ..experiment_recorder import ExperimentRecorder

from ..gpt_evolution.initial_population import generate_initial_population
from ..gpt_evolution.helpers import set_random_seeds, deep_merge_dicts, configure_logger, validate_flops_config
from ..nn.evaluate import calculate_fitness
from ..nn.individual import NeuralNetworkIndividual
from ..nn.evolution import NeuralNetworkEvolution
from ..nn.visualization import log_best_individual
from ..nn.variation.hyperparam_variation import (
    mutate_batch_size, crossover_batch_size,
    mutate_learning_rate, crossover_learning_rate,
    mutate_learning_rate_scheduler, crossover_learning_rate_scheduler,
    mutate_optimizer_parameters, crossover_optimizer_parameters,
)
from ..nn.variation.architecture_mutation import (
    mutation_add_linear, mutation_add_relu, mutation_add_skip_connection,
    mutation_add_branch, mutation_remove_node
)
from ..nn.variation.architecture_crossover import crossover_subgraph

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from datasets import load_dataset

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
import torch

VOCAB_SIZE = 2000
RANDOM_SEED = 42


def main():
    parser = argparse.ArgumentParser(description="Run GPT Evolution experiment.")
    parser.add_argument(
        "--config",
        type=str,
        default="src/gpt_evolution/default_run_config.json",
        help="Path to the run configuration JSON file."
    )
    parser.add_argument(
        "--experiment_path",
        type=str,
        default="./default_experiment_path",
        help="Path to the experiment directory."
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default="",
        help="Path to the tokenizer file."
    )
    args = parser.parse_args()

    with open('./src/gpt_evolution/default_run_config.json', 'r') as f:
        default_run_config = json.load(f)
    with open(args.config, 'r') as f:
        override_run_config = json.load(f)

    run_config = deep_merge_dicts(default_run_config, override_run_config)
    experiment_path = args.experiment_path
    tokenizer_path = args.tokenizer_path

    tokenizer_config = run_config["tokenizer"]
    evolution_config = run_config["evolution"]
    training_config = run_config["training"]
    gpt_config = run_config["gpt_config"]

    experiment_recorder = ExperimentRecorder(experiment_path, run_config)
    experiment_recorder.update_system_info(
        {
            "python_version": sys.version,
            "platform": platform.platform(),
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        }
    )
    experiment_recorder.update_run_metadata(
        cli_args={
            "config": args.config,
            "experiment_path": args.experiment_path,
            "tokenizer_path": args.tokenizer_path,
        }
    )
    try:
        git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=os.getcwd()).decode("utf-8").strip()
        experiment_recorder.update_run_metadata(git_commit=git_commit)
    except Exception:
        logging.debug("Unable to determine git commit for experiment metadata", exc_info=True)

    try:
        run_experiment(
            experiment_path,
            run_config,
            tokenizer_config,
            evolution_config,
            training_config,
            gpt_config,
            tokenizer_path,
            experiment_recorder,
        )
        run_status = "completed"
    except Exception:
        run_status = "failed"
        logging.exception("Unhandled exception during evolution run")
        raise
    finally:
        experiment_recorder.finalize(status=run_status)

def run_experiment(
    experiment_path: str,
    run_config: dict,
    tokenizer_config: dict,
    evolution_config: dict,
    training_config: dict,
    gpt_config: dict,
    tokenizer_path: str,
    experiment_recorder: ExperimentRecorder,
):
    os.makedirs(experiment_path, exist_ok=True)

    configure_logger(experiment_path, run_config.get("logging", {"overwrite_logs": False}))

    # Validate FLOPs configuration
    validate_flops_config(training_config)

    set_random_seeds(evolution_config["random_seed"])

    load_dataset_constant_kwargs = {
        "path": tokenizer_config["dataset"],
        "name": tokenizer_config["dataset_name"],
        "streaming": True,
    }
    if "data_files_prefixes" in tokenizer_config:
        suffix = tokenizer_config["data_files_suffix"]
        train_data_files = [f"{prefix}{suffix}" for prefix in tokenizer_config["data_files_prefixes"]["train"]]
        validation_data_files = [f"{prefix}{suffix}" for prefix in tokenizer_config["data_files_prefixes"]["validation"]]
        # TODO: test if the split is needed
        iterable_train_dataset = load_dataset(
            **load_dataset_constant_kwargs,
            split="train",
            data_dir=tokenizer_config["data_dir"],
            data_files=train_data_files,
        )
        iterable_validation_dataset = load_dataset(
            **load_dataset_constant_kwargs,
            split="train",
            data_dir=tokenizer_config["data_dir"],
            data_files=validation_data_files,
        )
        dataset_metadata = {
            "train_files": train_data_files,
            "validation_files": validation_data_files,
            "data_dir": tokenizer_config["data_dir"],
        }
    else:
        datasets = load_dataset(**load_dataset_constant_kwargs)
        iterable_train_dataset = datasets["train"]
        iterable_validation_dataset = datasets["validation"]
        dataset_metadata = {
            "dataset_split_keys": list(datasets.keys()),
        }

    experiment_recorder.update_run_metadata(dataset=dataset_metadata)

    tokenizer_path = tokenizer_path or os.path.join(experiment_path, tokenizer_config["tokenizer_filename"])
    tokenizer_copy_path = None

    if os.path.exists(tokenizer_path):
        logging.info("Loading tokenizer from file")
        tokenizer = Tokenizer.from_file(tokenizer_path)
        tokenizer_copy_path = os.path.join(experiment_path, tokenizer_config["tokenizer_filename"])
        tokenizer.save(tokenizer_copy_path)  # bring to new experiment path for cohesive storage
    else:
        tokenizer = Tokenizer(BPE())
        tokenizer.pre_tokenizer = Whitespace()

        def text_generator():
            count = 0
            total_samples = tokenizer_config.get("tokenizer_training_samples", 10000)  # Default to 10k samples
            for example in iterable_train_dataset:
                if count >= total_samples:
                    break
                yield example["text"]
                count += 1
        
        tokenizer.train_from_iterator(text_generator(), trainer=BpeTrainer(vocab_size=tokenizer_config["vocab_size"]))
        tokenizer.save(tokenizer_path)
        tokenizer_copy_path = tokenizer_path

    tokenizer_metadata = {
        "path": tokenizer_path,
        "filename": tokenizer_config["tokenizer_filename"],
        "vocab_size": tokenizer_config["vocab_size"],
        "training_samples": tokenizer_config.get("tokenizer_training_samples"),
    }
    if tokenizer_copy_path and tokenizer_copy_path != tokenizer_path:
        tokenizer_metadata["copied_to_experiment"] = tokenizer_copy_path
    experiment_recorder.update_run_metadata(tokenizer=tokenizer_metadata)

    train_config_params = {
        "training_total_batches": training_config.get("training_total_batches"),
        "device": training_config["device"],
    }
    experiment_recorder.update_system_info({"requested_training_device": training_config["device"]})

    # Note: FLOPs-based batch computation is applied per-individual below in fitness_wrapper

    # Create a wrapper for calculate_fitness that only takes individual
    def fitness_wrapper(individual: NeuralNetworkIndividual) -> float:
        return calculate_fitness(
            individual,
            iterable_train_dataset,
            iterable_validation_dataset,
            tokenizer,
            block_size=gpt_config["block_size"],
            total_batches_for_evaluation=training_config.get("evaluation_total_batches", 20),
            num_train_steps=train_config_params["training_total_batches"],
            device=train_config_params["device"],
            loss_log_frequency=training_config.get("loss_log_frequency", 100),
            iter_timeout=training_config.get("iter_timeout", 20.0),
            secondary_iter_timeout=training_config.get("secondary_iter_timeout", 0.2),
            flops_budget=training_config.get("flops_budget"),
            validation_batch_size=training_config.get("validation_batch_size", 32),
        )

    evolution = NeuralNetworkEvolution(
        population=generate_initial_population(
            evolution_config["target_population_size"],
            tokenizer_config["vocab_size"],
            gpt_config,
            train_config_params,
        ),
        fitness_fn=fitness_wrapper,
        experiment_recorder=experiment_recorder,
        crossover_instead_of_mutation_rate=evolution_config["crossover_instead_of_mutation_rate"],
        mutation_fns_and_probabilities=[  # These need to be imported above for it to work
            (globals()[name], prob) for name, prob in evolution_config["mutation_probabilities"].items()
        ],
        crossover_fns_and_probabilities=[  # These need to be imported above for it to work
            (globals()[name], prob) for name, prob in evolution_config["crossover_probabilities"].items()
        ],
        target_population_size=evolution_config["target_population_size"],
        num_children_per_generation=evolution_config["num_children_per_generation"],
        experiment_path=experiment_path,
        visualize_graphs=run_config.get("visualization", True),
        max_subgraph_attempts=evolution_config.get("max_subgraph_attempts", 100),
        unremovable_node_targets=evolution_config.get("unremovable_node_targets", []),
    )
    evolution.run_evolution(evolution_config["num_generations"])

    log_best_individual(evolution, experiment_path, run_config.get("logging", {}).get("overwrite_logs", False))
    if evolution.best_individual:
        experiment_recorder.update_run_metadata(best_individual_id=evolution.best_individual.id)


if __name__ == '__main__':
    main()
