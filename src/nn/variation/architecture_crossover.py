from collections import deque, defaultdict
import random
import copy
import logging
import os

import torch
from torch.fx.passes.shape_prop import ShapeProp

from ..individual import NeuralNetworkIndividual
from ..variation.utils import get_unique_name, node_has_shape
from ..visualization import visualize_graph
from ..variation.architecture_adaptation import adapt_node_shape

MAX_BOUNDARY_NODES = 10
MIN_NODES = 4
MAX_NODES = 32

CROSSOVER_VISUALIZATION_DIR = "crossover_visualization"

# TODO need to prune after to save computation

def crossover_subgraph(child: NeuralNetworkIndividual, parent: NeuralNetworkIndividual, **kwargs):
    crossover_visualization_dir = os.path.join(kwargs.get("experiment_path", ""), CROSSOVER_VISUALIZATION_DIR)
    os.makedirs(crossover_visualization_dir, exist_ok=True)

    subgraph_nodes = set()
    lowest_num_boundary_nodes = float('inf')
    broken_subgraphs = 0
    max_subgraph_attempts = kwargs.get("max_subgraph_attempts", 100)
    for attempt in range(max_subgraph_attempts):
        try:
            num_nodes = random.randint(MIN_NODES, MAX_NODES)
            subgraph_nodes, input_boundary_nodes, output_boundary_nodes = random_subgraph(parent.graph_module, num_nodes)
            num_boundary_nodes = len(input_boundary_nodes) + len(output_boundary_nodes)
            actual_subgraph_size = len(subgraph_nodes)
            
            logging.debug(f"Attempt {attempt + 1}: Generated subgraph with {actual_subgraph_size} nodes, {num_boundary_nodes} boundary nodes ({len(input_boundary_nodes)} input + {len(output_boundary_nodes)} output)")
            
            # Check constraints
            if num_boundary_nodes > MAX_BOUNDARY_NODES:
                logging.debug(f"Attempt {attempt + 1}: REJECTED - too many boundary nodes ({num_boundary_nodes} > {MAX_BOUNDARY_NODES})")
                continue
                
            if actual_subgraph_size < MIN_NODES:
                logging.debug(f"Attempt {attempt + 1}: REJECTED - too few nodes ({actual_subgraph_size} < {MIN_NODES})")
                continue
                
            # Reject subgraphs without proper input/output boundaries
            if len(input_boundary_nodes) == 0:
                logging.debug(f"Attempt {attempt + 1}: REJECTED - no input boundary nodes")
                continue
                
            if len(output_boundary_nodes) == 0:
                logging.debug(f"Attempt {attempt + 1}: REJECTED - no output boundary nodes")
                continue
                
            if num_boundary_nodes >= lowest_num_boundary_nodes:
                logging.debug(f"Attempt {attempt + 1}: REJECTED - not better than current best ({num_boundary_nodes} >= {lowest_num_boundary_nodes} boundary nodes)")
                continue
            
            # Try to find connections
            input_mapping, topo_target_input_nodes, output_mapping = find_subgraph_connections(child.graph_module.graph, input_boundary_nodes, output_boundary_nodes)
            
            lowest_num_boundary_nodes = num_boundary_nodes
            logging.debug(f"Attempt {attempt + 1}: ACCEPTED - new best subgraph with {num_boundary_nodes} boundary nodes")

            insert_subgraph_kwargs = {
                "subgraph_nodes": subgraph_nodes,
                "input_mapping": input_mapping,
                "topo_target_input_nodes": topo_target_input_nodes,
                "output_mapping": output_mapping
            }
        except ValueError as e:
            logging.warning(f"error finding subgraph: {e}")
            broken_subgraphs += 1
    logging.debug(f"broken_subgraphs: {broken_subgraphs}")

    # Extract node names for highlighting
    subgraph_node_names = {node.name for node in insert_subgraph_kwargs["subgraph_nodes"]}

    # Visualize the graph with the subgraph highlighted (only if visualization is enabled)
    visualize_graphs = kwargs.get("visualize_graphs", True)
    if visualize_graphs:
        random_int = random.randint(0, 1000000)
        visualize_graph(parent.graph_module, "model_graph_highlighted", os.path.join(crossover_visualization_dir, f"{random_int}_{parent.id}_graph_highlighted.svg"), highlight_nodes=subgraph_node_names)

    child.graph_module, new_node_names = insert_subgraph(child.graph_module, **insert_subgraph_kwargs)

    # Log successful subgraph insertion
    logging.info(f"Successfully inserted subgraph with {len(insert_subgraph_kwargs['subgraph_nodes'])} nodes into child {child.id} from parent {parent.id}")
    # Visualize the graph after crossover (only if visualization is enabled)  
    if visualize_graphs:
        visualize_graph(child.graph_module, "model_graph_after_crossover_highlighted", os.path.join(crossover_visualization_dir, f"{random_int}_{child.id}_graph_after_crossover_highlighted.svg"), highlight_nodes=new_node_names)

def random_subgraph(graph_module: torch.fx.GraphModule, num_nodes: int):
    """
    This function returns a random subgraph of the given graph.

    Args:
        graph_module: The graph module to get the subgraph from.
        num_nodes: The desired number of nodes in the subgraph (not guaranteed).

    Returns:
        A tuple of the candidate nodes, input boundary nodes, and output boundary nodes.
    """
    all_nodes = list(graph_module.graph.nodes)
    anchor_node = random.choice(all_nodes)
    rejected_anchors = 0
    while not _is_allowed_subgraph_node_type(anchor_node):
        logging.warning(f"picked node with non-allowed type or name: {anchor_node.op} {anchor_node.name}")
        anchor_node = random.choice(all_nodes)
        rejected_anchors += 1
        if rejected_anchors > 50:  # Safety check to avoid infinite loop
            raise ValueError(f"Too many rejected anchor nodes ({rejected_anchors}), may indicate graph structure issue")
    
    logging.debug(f"Selected anchor node: {anchor_node.name} (type: {anchor_node.op}) after rejecting {rejected_anchors} candidates")
    subgraph_nodes = {anchor_node}
    frontier_nodes = [anchor_node]
    while frontier_nodes and len(subgraph_nodes) < num_nodes:
        current_node = frontier_nodes.pop()
        candidate_nodes = set()
        for neighbor_node in (*current_node.all_input_nodes, *current_node.users):
            if neighbor_node not in subgraph_nodes and _is_allowed_subgraph_node_type(neighbor_node):
                candidate_nodes.add(neighbor_node)
        
        if len(subgraph_nodes) + len(candidate_nodes) <= num_nodes:
            for candidate_node in candidate_nodes:
                subgraph_nodes.add(candidate_node)
                frontier_nodes.append(candidate_node)

    # Find boundary nodes that are within the subgraph
    input_mapping, output_mapping = {}, {}
    subgraph_nodes_list = list(subgraph_nodes)
    def _add_to_subgraph(node):
        subgraph_nodes.add(node)
        subgraph_nodes_list.append(node)
    for node in subgraph_nodes_list:
        input_mapping[node], output_mapping[node] = [], []
        for arg in node.args:
            if isinstance(arg, torch.fx.Node):
                if arg in subgraph_nodes:
                    input_mapping[node].append(arg)
                elif _has_float_dtype(arg):
                    input_mapping[node].append(None)  # placeholder for target graph replacement arg
                elif _is_allowed_subgraph_node_type(arg):  # if neighbor node and has no shape, add it to the subgraph
                    _add_to_subgraph(arg)
                    input_mapping[node].append(arg)  # because now it's in the subgraph
            else:
                input_mapping[node].append(arg)
        if all(arg is not None for arg in input_mapping[node]):
            del input_mapping[node]  # if all node inputs are in the subgraph, we don't need to keep the mapping
        
        for user_node in node.users:
            if user_node in subgraph_nodes:
                output_mapping[node].append(user_node)
            elif _has_float_dtype(user_node):
                output_mapping[node].append(None)  # placeholder for target graph replacement user
            elif _is_allowed_subgraph_node_type(user_node):  # if neighbor node and has no shape, add it to the subgraph
                _add_to_subgraph(user_node)
                output_mapping[node].append(user_node)  # because now it's in the subgraph
        if all(user_node is not None for user_node in output_mapping[node]):
            del output_mapping[node]  # if all node outputs are in the subgraph, we don't need to keep the mapping

        if (node in input_mapping or node in output_mapping) and not _has_float_dtype(node):
            if node in input_mapping:
                del input_mapping[node]
                for arg in node.all_input_nodes:
                    _add_to_subgraph(arg)
            else:
                del output_mapping[node]
                for user_node in node.users:
                    _add_to_subgraph(user_node)

    return subgraph_nodes, input_mapping, output_mapping

def _is_allowed_subgraph_node_type(node: torch.fx.Node):
    # Reject placeholders / outputs
    if node.op in ("placeholder", "output"):
        return False

    # Reject training-specific helper nodes
    if "cross_entropy" in node.name or "targets" in node.name:
        return False

    # Reject Embedding modules (includes torch.nn.modules.sparse.Embedding)
    if node.op == "call_module":
        submodule = node.graph.owning_module.get_submodule(node.target)
        if isinstance(submodule, torch.nn.Embedding):
            return False

    return True 

def _has_float_dtype(node: torch.fx.Node):
    """Check if a node has float dtype (float32 or float64)."""
    if not node_has_shape(node) or not _is_allowed_subgraph_node_type(node):
        return False
    dtype = node.meta["tensor_meta"].dtype
    float_dtypes = [torch.float32, torch.float64, torch.bfloat16]
    return dtype in float_dtypes

def find_subgraph_connections(
    target_graph: torch.fx.Graph,
    input_mapping: dict[torch.fx.Node, list[torch.fx.Node|None]],
    output_mapping: dict[torch.fx.Node, list[torch.fx.Node|None]]
):
    """
    Finds compatible connection points between a subgraph and a target graph.
    
    Args:
        target_graph: The graph to insert the subgraph into
        input_mapping: Dict mapping subgraph input boundary node -> target graph args. A None arg implies we need to select a compatible target arg.
        output_mapping: Dict mapping subgraph output boundary node -> target graph users. A None user implies we need to select a compatible target user.
    
    Returns:
        A tuple of (input_mapping, topo_target_input_nodes, output_mapping)
        input_mapping: Dict mapping subgraph input boundary node -> list of args in either the target graph or the subgraph.
        topo_target_input_nodes: List of target args for the input boundary nodes, in topological order.
        output_mapping: Dict mapping subgraph output boundary node -> list of users in either the target graph or the subgraph.
    """
    target_graph_nodes = list(target_graph.nodes)
    
    def are_nodes_compatible(node1, node2):
        # Skip placeholder and output nodes
        if node1.op in ["placeholder", "output"] or node2.op in ["placeholder", "output"]:
            return False
            
        # Check tensor metadata
        if not _has_float_dtype(node1) or not _has_float_dtype(node2):
            return False
        
        if node1.meta["tensor_meta"].dtype != node2.meta["tensor_meta"].dtype:
            return False

        # Ensure batch dimension matches
        if node1.meta["tensor_meta"].shape[0] != node2.meta["tensor_meta"].shape[0]:
            return False
            
        return True

    def get_candidates(boundary_nodes):
        all_candidates = {}
        for node in boundary_nodes:  # TODO do we need each candidate list to be for an arg index?
            candidates = [n for n in target_graph_nodes if are_nodes_compatible(node, n)]
            if candidates:
                all_candidates[node] = candidates
            else:
                logging.warning(f"no candidates found for node: {node}")
        return all_candidates
    
    input_mapping, _ = _select_random_mapping(input_mapping, get_candidates(input_mapping))
    target_input_nodes = set(node for nodes in input_mapping.values() for node in nodes)
    
    output_mapping, topo_target_input_nodes = _select_random_mapping(
        output_mapping,
        get_candidates(output_mapping),
        target_graph,
       target_input_nodes
    )
    return input_mapping, topo_target_input_nodes, output_mapping

def _select_random_mapping(
    boundary_nodes: dict[torch.fx.Node, list[torch.fx.Node|None]],
    candidates_dict: dict[torch.fx.Node, list[torch.fx.Node]],
    target_graph: torch.fx.Graph|None = None,
    target_input_nodes: set[torch.fx.Node]|None = None
):
    """
    Randomly selects compatible target node(s) for each boundary node, avoiding clashes (and incorrect topological order if selecting output nodes).
    Args:
        boundary_nodes: Set of subgraph boundary nodes.
        candidates_dict: Dict mapping boundary nodes to lists of compatible target nodes.
        target_graph (optional): The target graph. Should be provided if target_input_nodes is provided.
        target_input_nodes (optional): Set of target input nodes. Should be provided if target_graph is provided.
    Returns:
        Dict mapping subgraph boundary node -> selected target node(s).
        Topologically ordered list of target input nodes.
    """
    visited_nodes, visited_target_input_nodes = _traverse_and_extract_target_inputs(target_graph, target_input_nodes) if target_input_nodes else ([], [])
    visited_nodes_set = set(visited_nodes)

    used_candidates = set()
    for node, args_or_users in boundary_nodes.items():
        for i, arg_or_user in enumerate(args_or_users):  # TODO if these are users (meaning output_mapping), we don't necessarily need the same number of users in the target graph... But we are forcing it to be the same here.
            if arg_or_user is not None:
                continue
            candidates = [c for c in candidates_dict.get(node, []) if c not in used_candidates and c not in visited_nodes_set]
            if candidates:
                selected = random.choice(candidates)
                used_candidates.add(selected)
                boundary_nodes[node][i] = selected
            else:
                raise ValueError(f"no candidates found for node: {node}")

    return boundary_nodes, visited_target_input_nodes

def _traverse_and_extract_target_inputs(target_graph: torch.fx.Graph, target_input_nodes: set[torch.fx.Node]):
    node_list = target_graph.nodes
    visited_nodes, visited_target_input_nodes = [], []
    for node in node_list:
        visited_nodes.append(node)
        if node not in target_input_nodes:
            continue

        visited_target_input_nodes.append(node)
        if set(visited_target_input_nodes) == target_input_nodes:
            break

    return visited_nodes, visited_target_input_nodes

def insert_subgraph(
    target_graph_module: torch.fx.GraphModule,
    subgraph_nodes: set[torch.fx.Node],
    input_mapping: dict[torch.fx.Node, list[torch.fx.Node|list[torch.fx.Node]]],
    topo_target_input_nodes: list[torch.fx.Node],  # TODO ideally we can just sort the input_mapping to be topographical instead of needing this list
    output_mapping: dict[torch.fx.Node, list[torch.fx.Node|list[torch.fx.Node]]],
):
    """
    Inserts a subgraph into the target graph.
    Args:
        target_graph: The FX graph to insert into.
        subgraph_nodes: Set of nodes in the subgraph.
        input_mapping: Dict mapping subgraph input boundary node -> target node(s). If it's a list, it means we need to select one of the target nodes.
        topo_target_input_nodes: List of target nodes for the input boundary nodes, in topological order.
        output_mapping: Dict mapping subgraph output boundary node -> target node(s). If it's a list, it means we need to select one of the target nodes.
    Returns:
        Modified target_graph.
    """
    new_node_names = set()
    old_to_new = {}
    ordered_subgraph = _kanh_algo(subgraph_nodes)
    logging.debug(f"ordered_subgraph: {ordered_subgraph}")
    logging.debug(f"input_mapping: {input_mapping}")
    logging.debug(f"output_mapping: {output_mapping}")

    # Insert nodes in topological order and adapt shapes
    for i, node in enumerate(ordered_subgraph):
        new_module_name, new_attr_name = None, None
        if node.op == "call_module":
            new_module_name = get_unique_name(target_graph_module, node.target)
            # Create a deep copy to avoid sharing parameters with the parent
            original_module = node.graph.owning_module.get_submodule(node.target)
            copied_module = copy.deepcopy(original_module)  # TODO unsure if this is necessary
            if hasattr(copied_module, "reset_parameters"):  # TODO before these two lines were added, the model trained so fast. Interesting finding we can maybe explore more.
                copied_module.reset_parameters()
            # TODO do we need to also force it to the same device?
            target_graph_module.add_submodule(new_module_name, m=copied_module)
            new_node_names.add(new_module_name)
        elif node.op == "get_attr":
            new_attr_name = get_unique_name(target_graph_module, node.target)
            original_attr_value = getattr(node.graph.owning_module, node.target)
            setattr(target_graph_module, new_attr_name, copy.deepcopy(original_attr_value))
            new_node_names.add(new_attr_name)

        after_node = old_to_new[ordered_subgraph[i-1]] if i > 0 else topo_target_input_nodes[-1]
        if node in input_mapping:  # Handle input boundary nodes
            target_inputs = []
            for arg in input_mapping[node]:
                target_inputs.append(old_to_new[arg] if arg in subgraph_nodes and isinstance(arg, torch.fx.Node) else arg)

            assert len(target_inputs) == len(node.args)
            new_node = _insert_node(
                target_graph_module,
                after_node=after_node,
                node=node,
                new_args=tuple(target_inputs),
                new_module_name=new_module_name,
                new_attr_name=new_attr_name
            )
            
            # Adapt shape if needed
            for j, target_input in enumerate(target_inputs):
                target_graph_module, _ = adapt_node_shape(
                    target_graph_module,
                    node=target_input,
                    current_size=target_input.meta["tensor_meta"].shape[1:],
                    target_size=node.args[j].meta["tensor_meta"].shape[1:],
                    target_user=new_node
                )
        else:
            new_args = []
            for arg in node.args:
                if isinstance(arg, torch.fx.Node):
                    new_args.append(old_to_new[arg])
                elif isinstance(arg, tuple):
                    new_args.append(tuple(old_to_new[a] if isinstance(a, torch.fx.Node) else a for a in arg))
                else:
                    new_args.append(arg)
            new_args = tuple(new_args)
            new_node = _insert_node(target_graph_module, after_node, node, new_args, new_module_name, new_attr_name)

        if new_node:
            new_node_names.add(new_node.name)
        old_to_new[node] = new_node

    # For each output boundary node, replace the input of the mapped target node
    for sub_out, users in output_mapping.items():
        logging.debug(f"sub_out: {sub_out}")
        for user in users:
            if user in subgraph_nodes:
                logging.debug(f"user already in subgraph_nodes: {user}")  # TODO instead of skipping, we might not even need to add it in the first place, in random_subgraph. I think I did this in case we cared about order of users, but I don't think we do...?
                continue
            new_out_node = old_to_new[sub_out]
            # Replace the input of user with new_out_node
            # TODO should we do a random selection instead of the first one with a shape?
            for i, arg in enumerate(user.args):
                if hasattr(arg, "meta") and "tensor_meta" in arg.meta:
                    break
            else:
                raise ValueError("no tensor_meta found for any args of user", user)

            tensor_meta = user.args[i].meta["tensor_meta"]
            try:
                first_arg_shape = tensor_meta.shape
            except:
                first_arg_shape = tensor_meta[i].shape  # Note: split nodes have a tuple of shapes. Maybe this is hacky?

            new_args = tuple([new_out_node, *user.args[1:]])  # TODO do we need to track arg indices? This is just replacing the first arg. But our current node compatibility check doesn't look at args at all, just tensor shapes... I'm confused.
            user.args = new_args

            target_graph_module, _ = adapt_node_shape(
                target_graph_module,
                node=new_out_node,
                current_size=sub_out.meta["tensor_meta"].shape[1:],
                target_size=first_arg_shape[1:],
                target_user=user
            )

    logging.debug(f"old_to_new: {old_to_new}")
    
    target_graph_module.graph.lint()
    target_graph_module.recompile()

    # Shape propagation
    try:
        ShapeProp(target_graph_module).propagate(target_graph_module.example_input)
    except Exception:
        logging.exception("error propagating shapes")
        logging.debug("\nTarget graph nodes with shapes:")
        for node in target_graph_module.graph.nodes:
            if hasattr(node, "meta") and "tensor_meta" in node.meta and hasattr(node.meta["tensor_meta"], "shape"):
                logging.debug(f"{node.name}: {node.meta['tensor_meta'].shape}")
            else:
                logging.debug(f"{node.name}: No shape info")

    return target_graph_module, new_node_names

def _kanh_algo(subgraph_nodes: set[torch.fx.Node]) -> list[torch.fx.Node]:
    # Build dependency graph (only within subgraph_nodes)
    in_degree = {node: 0 for node in subgraph_nodes}
    dependents = defaultdict(list)
    for node in subgraph_nodes:
        for input_node in node.all_input_nodes:
            if input_node in subgraph_nodes:
                in_degree[node] += 1
                dependents[input_node].append(node)

    # Initialize queue with nodes having in-degree 0
    topo_queue = deque([node for node in subgraph_nodes if in_degree[node] == 0])
    topo_order = []
    while topo_queue:
        node = topo_queue.popleft()
        topo_order.append(node)
        for dependent in dependents[node]:
            in_degree[dependent] -= 1
            if in_degree[dependent] == 0:
                topo_queue.append(dependent)
    return topo_order

def _insert_node(target_graph: torch.fx.GraphModule, after_node: torch.fx.Node, node: torch.fx.Node, new_args, new_module_name, new_attr_name):
    logging.debug(f"inserting after: {after_node}")
    def _insert_call(func, target):
        with target_graph.graph.inserting_after(after_node):
            return func(target, args=new_args, kwargs=node.kwargs)

    if node.op == "call_module":
        return _insert_call(target_graph.graph.call_module, new_module_name if new_module_name else node.target)
    elif node.op == "call_function":
        return _insert_call(target_graph.graph.call_function, node.target)
    elif node.op == "call_method":
        return _insert_call(target_graph.graph.call_method, node.target)
    elif node.op == "get_attr":
        with target_graph.graph.inserting_after(after_node):
            return target_graph.graph.get_attr(new_attr_name if new_attr_name else node.target)
    else:
        raise ValueError("unsupported node type", node, node.op)
