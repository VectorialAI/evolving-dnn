import copy

import torch
import torch.fx

class NeuralNetworkIndividualGraphModule(torch.fx.GraphModule):
    def __init__(self, graph_module: torch.fx.GraphModule, example_input: torch.Tensor|None = None):
        super().__init__(graph_module, graph_module.graph)
        self.example_input = example_input
        # Running gradient importance for directly parameterized FX nodes.
        self.node_grad_stats = {}
        self.sync_grad_stats()

    def _weighted_nodes(self):
        for node in self.graph.nodes:
            if node.op == "call_module":
                module = self.get_submodule(node.target)
                if any(param.requires_grad for param in module.parameters(recurse=False)):
                    yield node, module

    def sync_grad_stats(self):
        # Reconcile only after graph changes; training updates existing entries in place.
        current = {node.name for node, _ in self._weighted_nodes()}
        self.node_grad_stats = {
            name: dict(self.node_grad_stats.get(name, {"ema_norm": 0.0, "updates": 0}))
            for name in current
        }

    def update_grad_stats(self, decay: float):
        node_norms = []
        for node, module in self._weighted_nodes():
            grads = [param.grad.detach() for param in module.parameters(recurse=False) if param.grad is not None]
            if not grads:
                continue
            node_norms.append((node.name, torch.stack([grad.norm() for grad in grads]).norm()))
        if not node_norms:
            return
        # Transfer scalar norms once rather than synchronizing the GPU per node.
        norms = torch.stack([norm for _, norm in node_norms]).cpu().tolist()
        for (name, _), norm in zip(node_norms, norms):
            stats = self.node_grad_stats.setdefault(name, {"ema_norm": 0.0, "updates": 0})
            stats["ema_norm"] = norm if not stats["updates"] else decay * stats["ema_norm"] + (1 - decay) * norm
            stats["updates"] += 1

    def configure_optimizers(self, train_config):  # COPIED FROM karpathy/mingpt/model.py
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, )
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name
                # random note: because named_modules and named_parameters are recursive
                # we will see the same tensors p many many times. but doing it this way
                # allows us to know which parent module any tensor p belongs to...
                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": train_config.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=train_config.learning_rate, betas=train_config.betas)
        return optimizer

    def __deepcopy__(self, memo):
        example_input = self.example_input
        temp = super().__deepcopy__(memo)
        temp.example_input = example_input  # this is not a deep copy of the tensor, but it's fine for now
        temp.node_grad_stats = copy.deepcopy(self.node_grad_stats, memo)
        temp.sync_grad_stats()
        return temp
