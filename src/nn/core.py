import torch
import torch.nn as nn
import torch.fx
from torch.fx.passes.shape_prop import ShapeProp

from .individual_graph_module import NeuralNetworkIndividualGraphModule
from .variation.utils import node_has_shape


def get_graph(
    model: nn.Module,
    input_shape: tuple | None = None,
    safe_output_dims: list[int] = [0],
    example_input: torch.Tensor | None = None,
):
    """
    Takes a PyTorch model and returns its computation graph using torch.fx
    
    Args:
        model: A PyTorch model (nn.Module)
        input_shape: tuple specifying input tensor shape (batch, seq_len, dim)  # SHAPE NOTE: input_shape includes batch dimension
        safe_output_dims: Indices that describe the "safe" dimensions for
            the initial node output; stored in node metadata for downstream passes. Defaults to only the batch dimension.
    Returns:
        graph: The computation graph object from torch.fx.symbolic_trace with shape information
    """
        
    # Symbolically trace the model to get computation graph
    if example_input is not None:
        graph = NeuralNetworkIndividualGraphModule(torch.fx.symbolic_trace(model), example_input=example_input)
    else:
        graph = NeuralNetworkIndividualGraphModule(torch.fx.symbolic_trace(model))
    
    # Perform shape propagation if input_shape is provided
    if input_shape is not None and example_input is None:
        # Create example input
        example_input = torch.randn(input_shape)  # SHAPE NOTE: Using full shape including batch dimension
        graph.example_input = example_input
        
    if example_input is not None:
        # Get the first node (should be placeholder/input)
        placeholder = next(iter(graph.graph.nodes))
        placeholder.meta['tensor_meta'] = {
            'dtype': example_input.dtype,
            'shape': example_input.shape,  # SHAPE NOTE: Storing full shape including batch dimension
            'requires_grad': example_input.requires_grad
        }
        num_dims = len(example_input.shape)
        if any(dim < 0 or dim >= num_dims for dim in safe_output_dims):
            raise ValueError(
                f"safe_output_dims must reference valid dimensions in range [0, {num_dims - 1}]"
            )
        ShapeProp(graph).propagate(example_input)  # SHAPE NOTE: Shape propagation uses full shape including batch dimension
        _set_safe_dims_on_meta(placeholder, safe_output_dims)
        _propagate_safe_dims(graph)
    
    return graph


def _propagate_safe_dims(graph: NeuralNetworkIndividualGraphModule) -> None:
    for node in graph.graph.nodes:
        if node.op == 'placeholder' or node.op == 'output':
            continue
        if not node_has_shape(node):
            continue

        input_nodes = _collect_shapeful_inputs(node)
        if not input_nodes:
            continue

        inferred_safe_dims = _infer_safe_dims(node, input_nodes)
        if inferred_safe_dims is None:
            continue

        _set_safe_dims_on_meta(node, inferred_safe_dims)


def _collect_shapeful_inputs(node: torch.fx.Node) -> list[torch.fx.Node]:
    inputs: list[torch.fx.Node] = []
    for candidate in _iterate_candidate_nodes(node.args):
        if node_has_shape(candidate):
            inputs.append(candidate)
    for candidate in _iterate_candidate_nodes(node.kwargs):
        if node_has_shape(candidate):
            inputs.append(candidate)
    return inputs


def _iterate_candidate_nodes(value):
    if isinstance(value, torch.fx.Node):
        yield value
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from _iterate_candidate_nodes(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from _iterate_candidate_nodes(item)


def _infer_safe_dims(node: torch.fx.Node, inputs: list[torch.fx.Node]) -> list[int] | None:
    if len(inputs) == 1:
        dims = _get_safe_dims_from_meta(inputs[0])
        return list(dims) if dims is not None else None

    safe_dim_records: list[tuple[list[int], tuple[int, ...]]] = []
    for input_node in inputs:
        dims = _get_safe_dims_from_meta(input_node)
        if dims is None:
            return None
        shape = _get_shape_from_meta(input_node)
        if shape is None:
            return None
        safe_dim_records.append((list(dims), tuple(shape)))

    if not safe_dim_records:
        return None

    if not _node_allows_safe_dim_divergence(node):
        reference_safe_dims = safe_dim_records[0][0]
        if all(record[0] == reference_safe_dims for record in safe_dim_records[1:]):
            return reference_safe_dims
        return None

    output_shape = _get_shape_from_meta(node)
    if output_shape is None:
        return None

    inferred: set[int] = set()
    output_shape_tuple = tuple(output_shape)

    for dims, input_shape in safe_dim_records:
        for dim in dims:
            if dim < 0 or dim >= len(input_shape):
                continue
            size = input_shape[dim]
            matches = [
                idx for idx, out_size in enumerate(output_shape_tuple)
                if out_size == size
            ]
            if len(matches) == 1:
                inferred.add(matches[0])

    if not inferred:
        return None

    return sorted(inferred)


def _node_allows_safe_dim_divergence(node: torch.fx.Node) -> bool:
    matmul_targets = {torch.matmul, torch.bmm}
    method_names = {'matmul', 'bmm'}
    if node.op == 'call_function' and node.target in matmul_targets:
        return True
    if node.op == 'call_method' and node.target in method_names:
        return True
    return False


def _set_safe_dims_on_meta(node: torch.fx.Node, safe_dims: list[int]) -> None:
    tensor_meta = node.meta.get('tensor_meta')
    dims_list = [int(dim) for dim in safe_dims]
    if tensor_meta is None:
        node.meta['tensor_meta'] = {'safe_dims': dims_list}
    elif isinstance(tensor_meta, dict):
        tensor_meta['safe_dims'] = dims_list
    else:
        setattr(tensor_meta, 'safe_dims', dims_list)


def _get_safe_dims_from_meta(node: torch.fx.Node):
    tensor_meta = node.meta.get('tensor_meta')
    if tensor_meta is None:
        return None
    if isinstance(tensor_meta, dict):
        return tensor_meta.get('safe_dims')
    return getattr(tensor_meta, 'safe_dims', None)


def _get_shape_from_meta(node: torch.fx.Node):
    tensor_meta = node.meta.get('tensor_meta')
    if tensor_meta is None:
        return None
    if isinstance(tensor_meta, dict):
        return tensor_meta.get('shape')
    return getattr(tensor_meta, 'shape', None)
