import copy
import json
import logging
import os

import torch
from torch.fx import Graph

from ..evolution import Evolution
from .individual import NeuralNetworkIndividual
from .visualization import visualize_graph
from .variation.utils import print_graph_debug_info

class NeuralNetworkEvolution(Evolution):
    def _pre_evaluation(self, individual: NeuralNetworkIndividual):
        n_params = sum(p.numel() for p in individual.graph_module.parameters())
        logging.debug(f"Individual {individual.id} has parameter count: {n_params:,}")
        individual.param_count = n_params  # TODO use this in fitness calculation, we should minimize this

    def _handle_evaluation_error(self, individual: NeuralNetworkIndividual):
        print_graph_debug_info(individual.graph_module)
        individual.graph_module = individual.graph_module.to('cpu')
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _log_individual(self, individual: NeuralNetworkIndividual):
        experiment_individuals_path = os.path.join(self.kwargs["experiment_path"], "individuals")
        train_configs_path = os.path.join(experiment_individuals_path, "train_configs")
        graphs_path = os.path.join(experiment_individuals_path, "graphs")
        models_path = os.path.join(experiment_individuals_path, "models")

        for path in [train_configs_path, graphs_path, models_path]:
            os.makedirs(path, exist_ok=True)
        
        try:
            logging.info(f"Individual {individual.id} has fitness {individual.fitness}")
            if train_configs_path and graphs_path and models_path:
                train_config_filepath = os.path.join(train_configs_path, f"{individual.id}_train_config.json")
                with open(train_config_filepath, "w") as train_config_file:
                    json.dump(individual.train_config.to_dict(), train_config_file, indent=4)

                graph_filepath = os.path.join(graphs_path, f"{individual.id}_graph.svg") if self.visualize_graphs else None
                if self.visualize_graphs and graph_filepath:
                    visualize_graph(individual.graph_module, "model_graph", graph_filepath)

                model_filepath = os.path.join(models_path, f"{individual.id}_model.pt")
                torch.save(individual.graph_module, model_filepath)

                artifacts = {
                    "train_config": train_config_filepath if os.path.exists(train_config_filepath) else None,
                    "graph_svg": graph_filepath if graph_filepath and os.path.exists(graph_filepath) else None,
                    "model_state": model_filepath if os.path.exists(model_filepath) else None,
                }
            else:
                artifacts = {}

            self.experiment_recorder.record_individual_evaluation(
                individual,
                generation=self.generation,
                artifacts=artifacts,
            )
        except Exception:
            logging.exception(f"Error logging/saving individual {individual.id}")
            self.experiment_recorder.record_individual_evaluation(
                individual,
                generation=self.generation,
                artifacts={},
            )

    def _copy_individual(self, individual: NeuralNetworkIndividual) -> NeuralNetworkIndividual:
        child = copy.deepcopy(individual)

        # reset all the weights
        graph: Graph = child.graph_module.graph
        log_msg = f"Resetting parameters for individual {individual.id}'s nodes: "
        for node in graph.nodes:
            if node.op == "call_module":
                submodule = child.graph_module.get_submodule(node.target)
                # If the submodule has a reset_parameters method, call it
                if hasattr(submodule, "reset_parameters"):
                    log_msg += f"{node.name}, "
                    submodule.reset_parameters()
        logging.debug(log_msg)

        return child
