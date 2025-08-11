import math
import logging

import torch
import torch.nn as nn

from .utils import add_specific_node

def _adapt_tensor_size(graph, node, current_size: int, target_size: int, target_user=None):
    """
    Helper function to adapt a tensor's size using repeat_interleave, circular padding, or adaptive pooling.
    
    Args:
        graph: The FX graph
        node: The node to adapt
        current_size: Current size (integer)
        target_size: Target size (integer)
        target_user: Optional specific node that should use the new node
    Returns:
        graph: The modified graph
        adapted_node: The node after adaptation
    """
    if current_size < target_size:
        length_multiplier = target_size // current_size
        
        if length_multiplier > 1:
            remainder = target_size % current_size
        
            # First repeat the tensor as many times as possible
            graph, repeat_node = add_specific_node(
                graph, 
                node, 
                torch.repeat_interleave, 
                kwargs={"repeats": length_multiplier, "dim": 1},
                target_user=target_user  # Intermediate node
            )
            logging.debug(f"Added repeat node {repeat_node.name} after node {node.name}, repeats: {length_multiplier}")

            if remainder > 0:
                # Then use circular padding for the remainder
                graph, adapted_node = add_specific_node(
                    graph, 
                    repeat_node, 
                    nn.CircularPad1d((0, remainder)),
                    target_user=target_user
                )
                logging.debug(f"Added circular pad node {adapted_node.name} after repeat node {repeat_node.name}, remainder: {remainder}")
            else:
                adapted_node = repeat_node
        else:
            # If we only need to wrap once, just use circular padding
            graph, adapted_node = add_specific_node(
                graph, 
                node, 
                nn.CircularPad1d((0, target_size - current_size)),
                target_user=target_user
            )
            logging.debug(f"Added circular pad node {adapted_node.name} after node {node.name}, target size: {target_size}, current size: {current_size}")
    else:
        # Need to decrease size - use adaptive pooling
        graph, adapted_node = add_specific_node(
            graph, 
            node, 
            nn.AdaptiveAvgPool1d(target_size),
            target_user=target_user
        )
        logging.debug(f"Added adaptive avg pool node {adapted_node.name} after node {node.name}, target size: {target_size}, current size: {current_size}")

    return graph, adapted_node

class ReshapeModule(nn.Module):
    """A PyTorch module for reshaping tensors to a specific target size."""
    
    def __init__(self, target_size):
        super().__init__()
        self.target_size = target_size
    
    def forward(self, x):
        return x.reshape(-1, *self.target_size)

@torch.fx.wrap
def custom_cat(tensor1, tensor2, dim=-1):
    return torch.cat((tensor1, tensor2), dim=dim)

def _unflatten_linear_flatten(graph, node, adapt_shape_values: tuple[int, int, int], target_user=None):
    """
    Helper function to perform unflatten-linear-flatten sequence on the last dimension.
    
    Args:
        graph: The FX graph
        node: The input node
        adapt_shape_values: Tuple of dimensions for reshaping (length, in_features, out_features)
        target_user: Optional specific node that should use the output
    Returns:
        graph: The modified graph
        output_node: The node after unflatten-linear-flatten
    """
    # Reshape last dimension
    graph, node = add_specific_node(
        graph,
        node,
        nn.Unflatten(dim=-1, unflattened_size=(adapt_shape_values[0], adapt_shape_values[1])),
        target_user=target_user
    )

    # Add linear layer on last dimension
    graph, node = add_specific_node(
        graph,
        node,
        nn.Linear(adapt_shape_values[1], adapt_shape_values[2], bias=False),
        target_user=target_user
    )

    # Flatten last dimension
    graph, node = add_specific_node(
        graph,
        node,
        nn.Flatten(start_dim=-2, end_dim=-1),
        target_user=target_user
    )
    
    return graph, node

def adapt_node_shape_basic(graph, node, current_size, target_size, target_user=None, adapt_type='regular'):
    """
    Adapts a node's output shape to match a target size using repetition, adaptive pooling or circular padding.
    
    Args:
        graph: The FX graph
        node: The node whose shape needs to be adapted
        current_size: Current size of the node's output, no batch dimension
        target_size: Desired size of the node's output, no batch dimension
        target_user: Optional specific node that should use the adapted output. If None, all users will be updated.
        adapt_type: Type of adaptation to use. Can be 'regular' or 'linear'
    Returns:
        graph: The modified graph
        adapted_node: The node after shape adaptation
    """
    current_dims = len(current_size)
    target_dims = len(target_size)
    
    # Handle 1D to 1D case directly
    if current_dims == 1 and target_dims == 1:
        if adapt_type == 'linear':
            return add_specific_node(
                graph,
                node,
                nn.Linear(current_size[0], target_size[0]),
                target_user=target_user
            )
        return _adapt_tensor_size(graph, node, current_size[0], target_size[0], target_user=target_user)
    
    # Calculate total elements
    current_total = math.prod(current_size)
    target_total = math.prod(target_size)
    
    # If total elements are the same, just reshape and return
    if current_total == target_total:
        return add_specific_node(
            graph,
            node,
            ReshapeModule(target_size),
            target_user=target_user
        )
    
    if adapt_type == 'linear' and current_size[:-1] == target_size[:-1]:  # Only use linear adapter if all but last dims are the same
        return add_specific_node(
            graph,
            node,
            nn.Linear(current_size[-1], target_size[-1]),
            target_user=target_user
        )
    
    # Step 1: Flatten if starting from multi-dimensional (2+:1 or 2+:2+)
    if current_dims > 1:
        graph, node = add_specific_node(
            graph, 
            node, 
            nn.Flatten(start_dim=1, end_dim=-1),
            target_user=target_user
        )
    
    # Step 2: Adapt tensor size (total elements differ, so this is always needed)
    graph, node = _adapt_tensor_size(
        graph, 
        node, 
        current_total, 
        target_total, 
        target_user=target_user
    )
    
    # Step 3: Unflatten if ending with multi-dimensional (1:2+ or 2+:2+)
    if target_dims > 1:
        graph, node = add_specific_node(
            graph, 
            node, 
            nn.Unflatten(dim=1, unflattened_size=target_size),
            target_user=target_user
        )
    
    return graph, node

def _calculate_gcf(a: int, b: int) -> int:
    """Calculate the Greatest Common Factor (GCF) between two numbers."""
    while b:
        a, b = b, a % b
    return a

def gcf_adapt_node_shape(graph, node, current_size, target_size, target_user=None):
    """
    Args:
        graph: The FX graph
        node: The node whose shape needs to be adapted
        current_size: Current size of the node's output, no batch dimension
        target_size: Desired size of the node's output, no batch dimension
        target_user: Optional specific node that should use the adapted output. If None, all users will be updated.
    Returns:
        graph: The modified graph
        adapted_node: The node after shape adaptation
    """
    # Get total elements of current_size and target_size
    current_total = math.prod(current_size)
    target_total = math.prod(target_size)
    logging.debug(f"Shape adaptation: {current_size} -> {target_size} (total elements: {current_total} -> {target_total})")
    
    # Calculate GCF between current_total and target_total
    gcf = _calculate_gcf(current_total, target_total)
    reduced_current = current_total // gcf
    reduced_target = target_total // gcf
    logging.debug(f"Found GCF={gcf}, reducing problem from {current_total}:{target_total} to {reduced_current}:{reduced_target}")
    
    # Use a single reshape to (gcf, reduced_current)
    graph, node = add_specific_node(
        graph,
        node,
        ReshapeModule((gcf, reduced_current)),
        target_user=target_user
    )
    logging.debug(f"Reshaped to factor out GCF: (batch, {gcf}, {reduced_current})")
    
    # Determine if we're upsampling or downsampling the reduced dimension
    is_upsampling = reduced_target > reduced_current
    logging.debug(f"Operation type: {'upsampling' if is_upsampling else 'downsampling'}")
    
    if is_upsampling:
        numerator, denominator = reduced_target, reduced_current
    else:
        numerator, denominator = reduced_current, reduced_target
    
    feature_ratio = numerator / denominator
    r1 = math.floor(feature_ratio)
    r2 = math.ceil(feature_ratio)
    
    if r1 != r2:
        length_scale = int((numerator - (r2*denominator)) / (r1 - r2))
    else:
        length_scale = 1

    if is_upsampling:
        # For upsampling, work with target/current ratio
        r1_slice_length = length_scale
        r1_shape_values = (length_scale, 1, r1)
        r2_slice_length = reduced_current - r1_slice_length
        r2_shape_values = (r2_slice_length, 1, r2)
        logging.debug(f"Upsampling ratios: r1={r1}, r2={r2}, scale={length_scale}")
    else:
        # For downsampling, work with current/target ratio
        r1_slice_length = r1 * length_scale
        r1_shape_values = (length_scale, r1, 1)
        r2_slice_length = reduced_current - r1_slice_length
        r2_shape_values = (r2_slice_length//r2, r2, 1)
        logging.debug(f"Downsampling ratios: r1={r1}, r2={r2}, scale={length_scale}")

    if r1 != r2:
        # Create a slicing operation to get the first part
        logging.debug(f"Slicing first part: dim=2, start=0, length={r1_slice_length}")
        graph, r1_node = add_specific_node(
            graph,
            node,
            torch.narrow,
            kwargs={"dim": 2, "start": 0, "length": r1_slice_length},
            target_user=target_user
        )
        logging.debug(f"Added narrow node for r1 slice (length={r1_slice_length})")
    else:
        r1_node = node

    # Only apply reshape-linear-flatten if r1 > 1
    if r1 > 1:
        graph, r1_node = _unflatten_linear_flatten(graph, r1_node, r1_shape_values, target_user)
        logging.debug(f"Added unflatten-linear-flatten sequence for r1 (shape={r1_shape_values})")
    
    # Chunking needed if r1 != r2
    if r1 != r2:
        graph, concat_node = add_specific_node(
            graph,
            r1_node,
            custom_cat,
            target_user=target_user
        )

        graph, r2_node = add_specific_node(
            graph,
            node,
            torch.narrow,
            kwargs={"dim": 2, "start": r1_slice_length, "length": r2_slice_length},
            target_user=concat_node
        )
        logging.debug(f"Added narrow node for r2 slice (length={r2_slice_length})")
        
        graph, r2_node = _unflatten_linear_flatten(graph, r2_node, r2_shape_values, concat_node)
        logging.debug(f"Added unflatten-linear-flatten sequence for r2 (shape={r2_shape_values})")
        
        concat_node.args = (r1_node, r2_node,-1)
        logging.debug("Added concatenation node")
    else:
        concat_node = r1_node
    
    # Use a single reshape to final target shape
    graph, concat_node = add_specific_node(
        graph,
        concat_node,
        ReshapeModule(target_size),
        target_user=target_user
    )
    logging.debug(f"Reshaped to final target shape {target_size}")
    return graph, concat_node

def adapt_node_shape(graph, node, current_size, target_size, target_user=None, adapt_type='gcf'):
    """
    Adapts a node's output shape to match a target size using repetition, adaptive pooling or circular padding.
    
    Args:
        graph: The FX graph
        node: The node whose shape needs to be adapted
        current_size: Current size of the node's output, no batch dimension
        target_size: Desired size of the node's output, no batch dimension
        target_user: Optional specific node that should use the adapted output. If None, all users will be updated.
        adapt_type: Type of adaptation to use. Can be 'regular', 'linear', or 'gcf'
    Returns:
        graph: The modified graph
        adapted_node: The node after shape adaptation
    """
    # Convert current_size and target_size to tuples if they are not already
    current_size = tuple(current_size)
    target_size = tuple(target_size)
    
    # If current_size = target_size, return the node
    if current_size == target_size:
        return graph, node
    
    # Get total elements of current_size and target_size
    current_total = math.prod(current_size)
    target_total = math.prod(target_size)
    
    # If current_total = target_total, return a reshape node
    if current_total == target_total:
        return add_specific_node(
            graph,
            node,
            ReshapeModule(target_size),
            target_user=target_user
        )
    
    if adapt_type == 'gcf':
        return gcf_adapt_node_shape(graph, node, current_size, target_size, target_user)
    else:
        return adapt_node_shape_basic(graph, node, current_size, target_size, target_user, adapt_type)
