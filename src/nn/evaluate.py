import json
import logging

import torch
from ptflops import get_model_complexity_info
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..mingpt_altered.trainer import Trainer
from .individual import NeuralNetworkIndividual
from .dataset import HuggingFaceIterableDataset

def calculate_fitness(
    individual: NeuralNetworkIndividual,
    iterable_train_dataset,
    iterable_test_dataset, 
    tokenizer,
    block_size: int,
    total_batches_for_evaluation: int = 20,
    num_train_steps: int = 100,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    loss_log_frequency: int = 100,
    max_iter_timeout: float = 20.0,
    secondary_iter_timeout: float = 0.2,
    flops_budget: int = None,
) -> float:
    """
    Calculate fitness of a GPT model by training it and returning negative loss
    (negative because evolution maximizes fitness, but we want to minimize loss)
    
    Args:
        individual: The NeuralNetworkIndividual to evaluate
        iterable_train_dataset: HuggingFace iterable dataset for training
        iterable_test_dataset: HuggingFace iterable dataset for testing
        tokenizer: Tokenizer for encoding text
        num_train_steps: Number of training steps to perform
        device: Device to train on
        loss_log_frequency: How often to log training loss (every N iterations)
        max_iter_timeout: Maximum seconds per iteration before terminating
        secondary_iter_timeout: Secondary timeout for iterations that are too slow
        
    Returns:
        float: Fitness score (higher is better)
    """
    
    # If FLOPs budget is set, compute per-individual training steps
    if flops_budget is not None:
        # Get the example input from the FX graph
        example_input = getattr(individual.graph_module, "example_input", None)
        batch_size = int(getattr(individual.train_config, "batch_size", 1))
        
        # Create a copy of the model for FLOPs calculation to avoid corrupting the original
        import copy
        model_copy = copy.deepcopy(individual.graph_module)
        
        # Calculate FLOPs for this model
        forward_flops, flops_per_train_step = calculate_model_flops(
            model_copy,
            batch_size,
            block_size,
            example_input
        )
        
        if flops_per_train_step <= 0:
            raise ValueError(f"FLOPs calculation failed for individual. Got flops_per_train_step={flops_per_train_step}. "
                           f"This usually means the model is incompatible with FLOPs calculation or has an error.")
        
        computed_batches = int(flops_budget // flops_per_train_step)
        
        # Apply sensible constraints (min 1 batch, max 10000)
        computed_batches = max(1, min(computed_batches, 10000))
        
        if computed_batches <= 0:
            computed_batches = 1
            logging.warning(f"FLOPs budget {flops_budget} too small for model (needs {flops_per_train_step} per step). Using minimum 1 batch.")
        
        # Log FLOPs information
        total_flops_used = computed_batches * flops_per_train_step
        flops_efficiency = (total_flops_used / flops_budget) * 100 if flops_budget > 0 else 0
        
        logging.info(f"Individual FLOPs: {forward_flops:,} forward, {flops_per_train_step:,} per step")
        logging.info(f"Training: {computed_batches} batches, {total_flops_used:,} total FLOPs ({flops_efficiency:.1f}% of budget)")
        
        num_train_steps = computed_batches
    
    # Ensure trainer runs desired number of steps
    individual.train_config.max_iters = num_train_steps

    # Create train dataset
    train_dataset = HuggingFaceIterableDataset(
        iterable_train_dataset,
        tokenizer,
        block_size,
        max_samples=num_train_steps * individual.train_config.batch_size * 2  # Provide enough samples
    )
    
    # Run training
    trainer = Trainer(individual.train_config, individual.graph_module, train_dataset)
    def batch_end_callback(trainer):
        # Use timeout values passed from run config
        if trainer.iter_dt > max_iter_timeout:  # if it even has one that's this bad, just kill it
            raise ValueError(f"Iteration took too long: {trainer.iter_dt} seconds at iter {trainer.iter_num}")
        if trainer.iter_num % loss_log_frequency == 0:  # Use configurable frequency
            logging.debug(f"iter_dt {trainer.iter_dt * 1000:.2f}ms; iter {trainer.iter_num}: train loss {trainer.loss.item():.5f}")

            # # TODO better to do some averaging here instead of just checking 1/100
            # What's the point of this?
            if trainer.iter_dt > secondary_iter_timeout:  # Do it here so less likely that a random slow iteration will cause the entire train to fail
                print("secondary_timeout", secondary_iter_timeout)
                raise ValueError(f"Iteration took too long: {trainer.iter_dt} seconds at iter {trainer.iter_num}")
    trainer.set_callback('on_batch_end', batch_end_callback)
    trainer.set_callback('on_train_end', batch_end_callback)
    trainer.run()

    # Calculate perplexity on the validation set
    perplexity = calculate_perplexity(
        individual.graph_module, 
        iterable_test_dataset,
        tokenizer,
        block_size,
        device=device,
        total_batches_for_evaluation=total_batches_for_evaluation
    )

    individual.graph_module = individual.graph_module.to('cpu')  # Move the model back to CPU, since we're not going to run it again
    if device == 'cuda': torch.cuda.empty_cache()

    fitness = -perplexity  # negative perplexity as fitness (lower perplexity = better) so that we can go uppies :)
    
    return fitness

def calculate_perplexity(
    model: torch.nn.Module,
    iterable_test_dataset,
    tokenizer,
    block_size: int,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    batch_size: int = 32,
    total_batches_for_evaluation: int = 20
) -> float:
    """
    Calculate perplexity of a GPT model on the provided data
    
    Args:
        model: The GPT model to evaluate
        iterable_test_dataset: HuggingFace iterable dataset for testing
        tokenizer: Tokenizer for encoding text
        block_size: Sequence length for the model
        device: Device to evaluate on
        batch_size: Batch size for evaluation
        
    Returns:
        float: Perplexity score (lower is better)
    """
    logging.debug(f"Calculating perplexity in device: {device}")
    model = model.to(device)
    model.eval()  # Set model to evaluation mode
    
    # Create test dataset
    test_dataset = HuggingFaceIterableDataset(
        iterable_test_dataset,
        tokenizer,
        block_size,
        max_samples=total_batches_for_evaluation * batch_size  # Limit samples for evaluation
    )
    
    # Create DataLoader
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        num_workers=0,  # Important for iterable datasets
        pin_memory=True
    )
    
    total_loss = 0.0
    total_tokens = 0
    
    # Disable gradient computation for efficiency
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            if i >= total_batches_for_evaluation:
                break
            idx, targets = batch
            idx, targets = idx.to(device), targets.to(device)
            
            # Forward pass
            logits = model(idx)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
            
            # Accumulate loss (weighted by number of tokens)
            total_loss += loss.item() * targets.numel()
            total_tokens += targets.numel()
    
    # Calculate average loss
    avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
    logging.debug(f"avg_loss: {avg_loss}")
    
    # Perplexity is exp(average negative log likelihood)
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    logging.debug(f"perplexity: {perplexity}")
    return perplexity

def calculate_model_flops(
    model: torch.nn.Module,
    batch_size: int,
    block_size: int,
    example_input: torch.Tensor = None,
) -> tuple[int, int]:
    """
    Calculate the number of FLOPs (floating point operations) for a model.
    
    Args:
        model: The model to analyze
        batch_size: Batch size to use for calculation
        block_size: Sequence length for the model
        example_input: Optional example input tensor to determine shape/dtype
        
    Returns:
        tuple[int, int]: A tuple containing (flops_per_forward_pass, flops_per_train_step)
    """
    if example_input is None:
        example_input = torch.zeros(1, block_size, dtype=torch.long)
    
    # Build input that matches the model's batch size
    seq_len = int(example_input.shape[1]) if example_input.dim() >= 2 else int(block_size)
    dtype = example_input.dtype
    
    def input_constructor(input_res):
        # Return a single tensor, not a tuple
        return torch.zeros(batch_size, seq_len, dtype=dtype)
    
    try:
        # Compute MACs for a single forward pass with this batch size
        macs, _params = get_model_complexity_info(
            model,
            input_res=(batch_size, seq_len),
            input_constructor=input_constructor,
            as_strings=False,
            print_per_layer_stat=False,
            verbose=False,
        )
        
        # Check if MACs calculation was successful
        if macs is None or macs <= 0:
            raise ValueError(f"FLOPs calculation failed: macs={macs}. This usually means the model is incompatible with FLOPs calculation.")
        
        # Approximate FLOPs per forward as 2 * MACs
        forward_flops = int(2 * macs)
        # Per train step ~ 2x forward (forward+backward)
        train_step_flops = int(2 * forward_flops)
        
        return forward_flops, train_step_flops
        
    except Exception as e:
        raise ValueError(f"FLOPs calculation failed: {e}. This usually means the model is incompatible with FLOPs calculation.")
