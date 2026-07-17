import copy
import math
import unittest
from types import SimpleNamespace

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset

from src.mingpt_altered.trainer import Trainer
from src.nn.core import get_graph


class GradTrackingTests(unittest.TestCase):
    def setUp(self):
        model = nn.Sequential(nn.Linear(2, 2), nn.ReLU())
        with torch.no_grad():
            model[0].weight.fill_(1)
            model[0].bias.zero_()
        self.graph = get_graph(model, example_input=torch.ones(1, 2))
        self.linear_name = next(node.name for node in self.graph.graph.nodes if node.op == "call_module")

    def test_updates_ema_for_weighted_node_only(self):
        output = self.graph(torch.ones(1, 2)).sum()
        output.backward()
        self.graph.update_grad_stats(0.5)
        first = self.graph.node_grad_stats[self.linear_name]
        self.assertAlmostEqual(first["ema_norm"], math.sqrt(6), places=6)
        self.assertEqual(first["updates"], 1)

        for param in self.graph.parameters():
            param.grad = torch.zeros_like(param)
        self.graph.update_grad_stats(0.5)
        self.assertEqual(self.graph.node_grad_stats[self.linear_name]["updates"], 2)
        self.assertAlmostEqual(self.graph.node_grad_stats[self.linear_name]["ema_norm"], math.sqrt(6) / 2, places=6)
        self.assertEqual(set(self.graph.node_grad_stats), {self.linear_name})

    def test_deepcopy_and_sync_are_independent(self):
        self.graph.node_grad_stats[self.linear_name] = {"ema_norm": 3.0, "updates": 4}
        child = copy.deepcopy(self.graph)
        child.node_grad_stats[self.linear_name]["ema_norm"] = 1.0
        self.assertEqual(self.graph.node_grad_stats[self.linear_name]["ema_norm"], 3.0)

        child.node_grad_stats["deleted"] = {"ema_norm": 2.0, "updates": 1}
        child.sync_grad_stats()
        self.assertNotIn("deleted", child.node_grad_stats)

    def test_sync_removes_deleted_nodes_and_initializes_new_weighted_nodes(self):
        linear = next(node for node in self.graph.graph.nodes if node.name == self.linear_name)
        input_node = next(node for node in self.graph.graph.nodes if node.op == "placeholder")
        linear.replace_all_uses_with(input_node)
        self.graph.graph.erase_node(linear)
        self.graph.delete_all_unused_submodules()
        self.graph.sync_grad_stats()
        self.assertNotIn(self.linear_name, self.graph.node_grad_stats)

        output = next(node for node in self.graph.graph.nodes if node.op == "output")
        self.graph.add_submodule("extra", nn.Linear(2, 2))
        with self.graph.graph.inserting_before(output):
            extra = self.graph.graph.call_module("extra", args=(input_node,))
        output.args = (extra,)
        self.graph.recompile()
        self.graph.sync_grad_stats()
        self.assertEqual(self.graph.node_grad_stats[extra.name], {"ema_norm": 0.0, "updates": 0})

    def test_trainer_records_grad_stats_before_the_step(self):
        config = SimpleNamespace(
            device="cpu", training_total_batches=1, batch_size=1, num_workers=0,
            learning_rate=0.001, betas=(0.9, 0.95), weight_decay=0.0,
            grad_norm_clip=1.0, grad_ema_decay=0.95,
        )
        trainer = Trainer(config, self.graph, TensorDataset(torch.ones(1, 2), torch.zeros(1, dtype=torch.long)))
        trainer.run()
        self.assertEqual(self.graph.node_grad_stats[self.linear_name]["updates"], 1)


if __name__ == "__main__":
    unittest.main()
